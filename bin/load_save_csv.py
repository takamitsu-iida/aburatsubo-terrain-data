#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeeperのGPSデータのCSVファイルを入出力するスクリプトです。


"""

#
# 標準ライブラリのインポート
#
import logging
import sys

from pathlib import Path
from typing import Dict, List, Tuple, Dict  # , Any, Optional

# WSL1 固有の numpy 警告を抑制
# https://github.com/numpy/numpy/issues/18900
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, module=r"numpy.*", message=r"Signature b")

#
# 外部ライブラリのインポート
#
try:
    # データフレーム操作
    import pandas as pd

    # データを整形して表示
    from tabulate import tabulate

except ImportError as e:
    logging.error(f"必要なライブラリがインストールされていません: {e}")
    sys.exit(1)

# このファイルへのPathオブジェクト
app_path = Path(__file__)

# このファイルがあるディレクトリ
app_dir = app_path.parent

# このファイルの名前から拡張子を除いてプログラム名を得る
app_name = app_path.stem

# アプリケーションのホームディレクトリはこのファイルからみて一つ上
app_home = app_path.parent.joinpath('..').resolve()

# データを格納しているディレクトリ
data_dir = app_home.joinpath("data")

#
# ここからスクリプト
#

def read_file_lines(file_path: Path, callback: callable) -> list[str] | None:
    lines = []
    try:
        with file_path.open() as f:
            for line in f:
                line = line.rstrip()
                lines.append(line)
                if callback:
                    callback(line)
    except IOError as e:
        logging.exception(f"ファイルの読み込みに失敗しました: {file_path} - {str(e)}")
        return None
    return lines


def line_callback(line: str) -> None:
    print(line)


def read_csv(file_path: Path) -> Tuple[List[List[float]], Dict[str, Dict[str, float]]]:
    """
    CSVファイルを読み込み、データのリストおよびlat, lon, depthの最大・最小値を返す。
    epochがあってもなくても大丈夫なようにする

    Returns:
        data: List[List[float]]
        stats: Dict[str, Dict[str, float]]
            例:
              {
                'lat': {'min': ..., 'max': ...},
                'lon': {'min': ..., 'max': ...},
                'depth': {'min': ..., 'max': ...}
              }
    """
    data = []
    min_lat = float('inf')
    max_lat = float('-inf')
    min_lon = float('inf')
    max_lon = float('-inf')
    min_depth = float('inf')
    max_depth = float('-inf')
    min_epoch = float('inf')
    max_epoch = float('-inf')

    with file_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            split_values = line.split(',')

            if len(split_values) == 3:
                try:
                    lat, lon, depth = [float(x) for x in split_values[:3]]
                    epoch = 0.0  # epochがない場合は0にする
                except ValueError:
                    continue
            elif len(split_values) == 4:
                try:
                    lat, lon, depth, epoch = [float(x) for x in split_values[:4]]
                except ValueError:
                    continue
            else:
                continue

            data.append([lat, lon, depth, epoch])
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
            min_depth = min(min_depth, depth)
            max_depth = max(max_depth, depth)

            min_epoch = min(min_epoch, epoch)
            max_epoch = max(max_epoch, epoch)

    stats = {
        'lat': {'min': min_lat, 'max': max_lat},
        'lon': {'min': min_lon, 'max': max_lon},
        'depth': {'min': min_depth, 'max': max_depth},
        'epoch': {'min': min_epoch, 'max': max_epoch}
    }

    tabulate_headers = ["", "lat", "lon", "depth"]
    tabulate_table = [
        ["min", stats['lat']['min'], stats['lon']['min'], stats['depth']['min']],
        ["max", stats['lat']['max'], stats['lon']['max'], stats['depth']['max']],
    ]

    logging.info(f"read_csv() {file_path}\n{tabulate(tabulate_table, headers=tabulate_headers, floatfmt='.6f')}\n")

    return data, stats


def load_csv(input_data_path: Path) -> pd.DataFrame | None:
    """
    CSVファイルを読み込み、DataFrameとして返す。

    Args:
        input_data_path: Path to the input CSV file.

    Returns:
        pd.DataFrame | None: 読み込んだデータフレーム、もしくは読み込みに失敗した場合はNoneを返す
    """
    try:
        # CSVファイルに列名の行は存在しない前提なのでheader=Noneを指定して読み込む
        df = pd.read_csv(input_data_path, header=None)

        # データフレームに列名を定義
        if df.shape[1] == 3:
            df.columns = ["lat", "lon", "depth"]
        elif df.shape[1] == 4:
            df.columns = ["lat", "lon", "depth", "epoch"]
        else:
            return None
    except Exception as e:
        return None
    return df


def save_csv(df: pd.DataFrame, output_file_path: Path) -> None:
    # データフレームをCSVファイルに保存する
    try:
        df.to_csv(output_file_path, index=False, header=False)
    except Exception as e:
        logging.error(f"CSVファイルの保存に失敗しました：{str(e)}")


def save_points_as_csv(points: List[Dict[str, float]], output_file_path: Path) -> None:
    """
    座標のリストをCSVファイルに保存する。

    Args:
        points: List of dictionaries containing {'lat': ..., 'lon': ..., 'depth': ...}.
        output_file_path: Path to the output CSV file.
    """

    must_keys = ['lat', 'lon', 'depth']

    try:
        with output_file_path.open('w') as f:
            for p in points:
                if not all(k in p for k in must_keys):
                    continue
                line = f"{p['lat']},{p['lon']},{p['depth']}"
                if 'epoch' in p:
                    line += f",{p['epoch']}"
                line += "\n"
                f.write(line)
    except Exception as e:
        logging.error(f"CSVファイルの保存に失敗しました：{str(e)}")