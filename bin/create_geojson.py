#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSVデータの凸包をGeoJSON形式で保存します。

"""

SCRIPT_DESCRIPTION: str = 'Create GeoJSON convex hull from CSV data'

#
# 標準ライブラリのインポート
#
import argparse
import logging
import json
import sys

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable

#
# ローカルライブラリのインポート
#
from load_save_csv import read_csv_points

#
# 外部ライブラリのインポート
#
try:
    from shapely.geometry import Point, MultiPoint, mapping
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

# データディレクトリ
data_dir: Path = app_home.joinpath("data")

#
# ログ設定
#

# ログファイルの名前
log_file = f"{app_name}.log"

# ログファイルを置くディレクトリ
log_dir = app_home.joinpath("log")
log_dir.mkdir(exist_ok=True)

# ロギングの設定
# レベルはこの順で下にいくほど詳細になる
#   logging.CRITICAL
#   logging.ERROR
#   logging.WARNING --- 初期値はこのレベル
#   logging.INFO
#   logging.DEBUG
#
# ログの出力方法
# logger.debug("debugレベルのログメッセージ")
# logger.info("infoレベルのログメッセージ")
# logger.warning("warningレベルのログメッセージ")

# ロガーを取得
logger = logging.getLogger(__name__)

# ルートロガーへの伝播を無効化
logger.propagate = False

# ログレベル設定
logger.setLevel(logging.INFO)

# フォーマット
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 標準出力へのハンドラ
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)

# ログファイルのハンドラ
file_handler = logging.FileHandler(log_dir.joinpath(log_file), 'a+', encoding='utf-8')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

#
# ここからスクリプト
#

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
                    "name": "Lat Lon Convex Hull",
                    "description": "Convex hull of input points",
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

    def main() -> None:

        # 引数処理
        parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
        parser.add_argument('--input', type=str, required=True, help='dataディレクトリ直下の入力CSVファイル名')
        parser.add_argument('--output', type=str, required=True, help='dataディレクトリ直下の出力CSVファイル名')
        args = parser.parse_args()

        # 引数が何も指定されていない場合はhelpを表示して終了
        if not any(vars(args).values()):
            parser.print_help()
            return

        # 保存先のファイル名が指定されていない場合は終了
        if not args.output:
            logger.error("出力ファイル名が指定されていません。")
            return

        # 入力ファイルのパス
        input_file_path = Path(data_dir, args.input)
        if not input_file_path.exists():
            logger.error(f"入力ファイルが存在しません: {input_file_path}")
            return

        # 出力ファイルの名前とパス
        output_filename = args.output
        output_file_path = Path(data_dir, output_filename)

        logger.info("凸包GeoJSON生成処理を開始します")

        # CSVファイルから点データを読み込む
        logger.info(f"入力ファイル: {input_file_path}")
        points = read_csv_points(input_file_path)

        if points is None:
            logger.error("CSVファイルの読み込みに失敗しました")
            return
        if len(points) < 3:
            logger.error("凸包を作成するには最低3点必要です")
            return

        # 凸包を作成
        geojson = create_convex_hull(points)

        # GeoJSONファイルとして保存
        logger.info(f"出力ファイル: {output_file_path}")
        save_geojson(geojson, output_file_path)

    main()