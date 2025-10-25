#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeeperのGPSデータを入力として受け取り、四分木を使ってデータを集約した新しいCSVファイルを作成するスクリプト。


"""

# スクリプトを引数無しで実行したときのヘルプに使うデスクリプション
SCRIPT_DESCRIPTION = 'Aggregate Deeper GPS data using Quadtree'

#
# 標準ライブラリのインポート
#
import logging
import os
import sys

from pathlib import Path

# WSL1 固有の numpy 警告を抑制
# https://github.com/numpy/numpy/issues/18900
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, module=r"numpy.*", message=r"Signature b")

#
# 外部ライブラリのインポート
#
try:
    import matplotlib.pyplot as plt
    from tabulate import tabulate
    from geopy.distance import great_circle, distance
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    sys.exit(1)

#
# 独自モジュールのインポート
#
from qtree import Quadtree, QuadtreeNode


# このファイルへのPathオブジェクト
app_path = Path(__file__)

# このファイルがあるディレクトリ
app_dir = app_path.parent

# このファイルの名前から拡張子を除いてプログラム名を得る
app_name = app_path.stem

# アプリケーションのホームディレクトリはこのファイルからみて一つ上
app_home = app_path.parent.joinpath('..').resolve()

# ディレクトリ
data_dir = app_home.joinpath("data")

#
# ログ設定
#

# ログファイルの名前
log_file = app_path.with_suffix('.log').name

# ログファイルを置くディレクトリ
log_dir = app_home.joinpath('log')
log_dir.mkdir(exist_ok=True)

# ログファイルのパス
log_path = log_dir.joinpath(log_file)

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
USE_FILE_HANDLER = True
if USE_FILE_HANDLER:
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file), 'a+')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

#
# ここからスクリプト
#


if __name__ == '__main__':


    def main():
        data_filename = "data.csv"
        data_path = data_dir.joinpath(data_filename)
        if not data_path.exists():
            logger.error("File not found: %s" % data_path)
            return

        data, data_stats = read_csv(data_path)

        #           lat         lon      depth
        # ---  ---------  ----------  ---------
        # min  35.157248  139.598684   1.082000
        # max  35.173733  139.621782  47.539000

        # 中央座標を求める
        mid_lat = (data_stats['lat']['min'] + data_stats['lat']['max']) / 2
        mid_lon = (data_stats['lon']['min'] + data_stats['lon']['max']) / 2

        # 四分木の境界を正方形で設定
        square_size = max(
            mid_lat - data_stats['lat']['min'],
            data_stats['lat']['max'] - mid_lat,
            mid_lon - data_stats['lon']['min'],
            data_stats['lon']['max'] - mid_lon
        )
        bounds = (mid_lat - square_size, mid_lon - square_size, mid_lat + square_size, mid_lon + square_size)
        quadtree = Quadtree(bounds)

        for row in data:
            lat, lon, depth = row
            point = {'lat': lat, 'lon': lon, 'depth': depth}
            quadtree.insert(point)

        # 四分木の統計情報を取得
        stats = quadtree.stats()

        # 四分木の統計情報を表示する
        table = [
            ["total_nodes", stats.get("total_nodes", 0)],
            ["leaf_nodes", stats.get("leaf_nodes", 0)],
            ["max_level", stats.get("max_level", 0)],
            ["deepest_nodes_count", stats.get("deepest_nodes_count", 0)],
            ["leaf_points_max", stats.get("leaf_points_max", 0)],
            ["deepest_node_size(m)", f"{stats.get('deepest_node_size_m', 0):.3f}"]
        ]
        headers = ["", "value"]
        logger.info(f"Quadtree stats\n{tabulate(table, headers=headers, numalign='right')}")

        # 最も深いノードについては、そのノード内の点の平均値に置き換える
        deepest_level = stats.get("max_level", 0)
        for node in quadtree.get_leaf_nodes():
            if node.level == deepest_level and len(node.points) > 1:
                avg_point = node.average()
                node.points = [avg_point]

        aggregated_points = []
        for node in quadtree.get_leaf_nodes():
            aggregated_points.extend(node.points)
        logger.info(f"Aggregated points count: {len(aggregated_points)}")


        # 四分木の可視化画像を保存
        output_image_path = log_dir.joinpath("quadtree_visualization.png")
        save_quadtree_image(quadtree, output_image_path)

        output_path = app_home.joinpath("static/data/aggregated_data.csv")
        with open(output_path, 'w') as f:
            f.write("lat,lon,depth\n")
            for p in aggregated_points:
                f.write(f"{p['lat']},{p['lon']},{p['depth']}\n")
        logger.info(f"Aggregated data saved to: {output_path}")

    #
    # 実行
    #

    main()
