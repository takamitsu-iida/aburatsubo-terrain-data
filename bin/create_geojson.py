#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSVデータの凸包をGeoJSON形式で保存します。

"""

#
# 標準ライブラリのインポート
#
import logging
import json
import sys
import csv

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable

# WSL1 固有の numpy 警告を抑制
# https://github.com/numpy/numpy/issues/18900
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, module=r"numpy.*", message=r"Signature b")

#
# 外部ライブラリのインポート
#
try:
    from shapely.geometry import Point, MultiPoint, mapping
    from shapely.ops import transform
    import pyproj
except ImportError as e:
    logging.error("必要なライブラリがインストールされていません。")
    logging.error("pip install shapely pyproj を実行してください。")
    sys.exit(1)

# このファイルへのPathオブジェクト
app_path: Path = Path(__file__)

# このファイルがあるディレクトリ
app_dir: Path = app_path.parent

# このファイルの名前から拡張子を除いてプログラム名を得る
app_name: str = app_path.stem

# アプリケーションのホームディレクトリはこのファイルからみて一つ上
app_home: Path = app_path.parent.joinpath('..').resolve()

#
# ログ設定
#

# ログファイルの名前
log_file: str = app_path.with_suffix('.log').name

# ログファイルを置くディレクトリ
log_dir: Path = app_home.joinpath('log')
log_dir.mkdir(exist_ok=True)

# ログファイルのパス
log_path: Path = log_dir.joinpath(log_file)

# ロギングの設定
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# フォーマット
formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 標準出力へのハンドラ
stdout_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)

# ログファイルのハンドラ
file_handler: logging.FileHandler = logging.FileHandler(
    log_dir.joinpath(log_file), 'a+', encoding='utf-8'
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


def read_csv_points(csv_path: Path) -> List[Tuple[float, float, float]]:
    """
    CSVファイルから(lat, lon, depth)のデータを読み込む

    Args:
        csv_path: CSVファイルのパス

    Returns:
        (lat, lon, depth)のタプルのリスト
    """
    points: List[Tuple[float, float, float]] = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # ヘッダー行の有無を自動判定
            reader = csv.reader(f)
            first_row = next(reader)

            # 最初の行が数値でない場合はヘッダーとみなす
            try:
                lat = float(first_row[0])
                lon = float(first_row[1])
                depth = float(first_row[2])
                points.append((lat, lon, depth))
            except (ValueError, IndexError):
                logger.info("ヘッダー行を検出しました。スキップします。")

            # 残りの行を読み込む
            for row in reader:
                try:
                    lat = float(row[0])
                    lon = float(row[1])
                    depth = float(row[2])
                    points.append((lat, lon, depth))
                except (ValueError, IndexError) as e:
                    logger.warning(f"無効な行をスキップしました: {row}")
                    continue

        logger.info(f"{len(points)}個の点を読み込みました")
        return points

    except FileNotFoundError:
        logger.error(f"ファイルが見つかりません: {csv_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}")
        sys.exit(1)


def create_convex_hull(points: List[Tuple[float, float, float]]) -> Dict[str, Any]:
    """
    点群から凸包を作成し、GeoJSON形式で返す

    Args:
        points: (lat, lon, depth)のタプルのリスト

    Returns:
        GeoJSON形式の辞書
    """
    # (lon, lat)の順序でPointオブジェクトを作成（GeoJSONの座標順序に従う）
    shapely_points = [Point(lon, lat) for lat, lon, depth in points]

    # MultiPointオブジェクトを作成
    multi_point = MultiPoint(shapely_points)

    logger.info(f"凸包を計算中...")

    # 凸包を計算
    convex_hull = multi_point.convex_hull

    # GeoJSON形式に変換
    geojson_geometry = mapping(convex_hull)

    # GeoJSONのFeatureCollectionを作成
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geojson_geometry,
                "properties": {
                    "name": "Convex Hull",
                    "description": "CSVデータから作成された凸包",
                    "point_count": len(points)
                }
            }
        ]
    }

    logger.info(f"凸包の頂点数: {len(geojson_geometry.get('coordinates', [[]])[0])}")

    return geojson


def save_geojson(geojson: Dict[str, Any], output_path: Path) -> None:
    """
    GeoJSONデータをファイルに保存

    Args:
        geojson: GeoJSON形式の辞書
        output_path: 出力ファイルのパス
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        logger.info(f"GeoJSONファイルを保存しました: {output_path}")

    except Exception as e:
        logger.error(f"ファイルの保存中にエラーが発生しました: {e}")
        sys.exit(1)


# ================================================================================
# メイン処理
# ================================================================================

if __name__ == '__main__':

    # 入力ファイル
    csv_filename: str = 'ALL_depth_map_data_202510_de_dd_ol_ip_mf.csv'
    csv_filepath: Path = app_home.joinpath('data', csv_filename)

    # 出力ファイル名
    output_filename: str = 'convex.json'
    output_path: Path = app_home.joinpath('data', output_filename)

    def main() -> None:
        logger.info("=" * 80)
        logger.info("凸包GeoJSON生成処理を開始します")
        logger.info("=" * 80)

        # CSVファイルから点データを読み込む
        logger.info(f"入力ファイル: {csv_filepath}")
        points = read_csv_points(csv_filepath)

        if len(points) < 3:
            logger.error("凸包を作成するには最低3点必要です")
            sys.exit(1)

        # 凸包を作成
        geojson = create_convex_hull(points)

        # GeoJSONファイルとして保存
        logger.info(f"出力ファイル: {output_path}")
        save_geojson(geojson, output_path)

        logger.info("=" * 80)
        logger.info("処理が完了しました")
        logger.info("=" * 80)

    main()