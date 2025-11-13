#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeeperのGPSデータを入力として受け取り、集約と補間処理を行うスクリプトです。

"""

# スクリプトを引数無しで実行したときのヘルプに使うデスクリプション
SCRIPT_DESCRIPTION = 'Interpolate Deeper GPS data using Quadtree'

#
# 標準ライブラリのインポート
#
import argparse
import logging
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
    import numpy as np
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    sys.exit(1)

#
# ローカルファイルからインポート
#
try:
    from load_save_csv import load_csv
    from qtree import Quadtree, create_quadtree_from_df
except ImportError as e:
    logging.error(f"module import error: {e}")
    sys.exit(1)

# このファイルへのPathオブジェクト
app_path = Path(__file__)

# このファイルがあるディレクトリ
app_dir = app_path.parent

# このファイルの名前から拡張子を除いてプログラム名を得る
app_name = app_path.stem

# アプリケーションのホームディレクトリはこのファイルからみて一つ上
app_home = app_path.parent.joinpath('..').resolve()

# データディレクトリ
data_dir = app_home.joinpath("data")

# 出力画像ファイルを保存するディレクトリ
image_dir = app_home.joinpath("img")
image_dir.mkdir(exist_ok=True)

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
file_handler = logging.FileHandler(log_dir.joinpath(log_file), 'a+', encoding='utf-8')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

#
# ここからスクリプト
#


# データ補間
# ポイントのない葉ノードについては、隣接ノードに値があればその平均値で埋める
def interpolate_empty_leaf_nodes(quadtree: Quadtree):
    deepest_level = quadtree.get_deepest_level()
    empty_leaf_nodes = quadtree.get_empty_leaf_nodes()
    for node in empty_leaf_nodes:

        if node.level < deepest_level - 3:
            # 深さが浅いノードは無視
            continue

        # 隣接ノードに格納されているポイントを収集
        neighbor_points = []

        # ノードの中心座標
        mid_lat, mid_lon = node.center

        # 8方向の隣接ノードをチェック
        directions = [
            (mid_lat - node.lat_length, mid_lon - node.lon_length),  # NW
            (mid_lat - node.lat_length, mid_lon),                    # N
            (mid_lat - node.lat_length, mid_lon + node.lon_length),  # NE
            (mid_lat, mid_lon - node.lon_length),                    # W
            (mid_lat, mid_lon + node.lon_length),                    # E
            (mid_lat + node.lat_length, mid_lon - node.lon_length),  # SW
            (mid_lat + node.lat_length, mid_lon),                    # S
            (mid_lat + node.lat_length, mid_lon + node.lon_length)   # SE
        ]

        # 8方向の隣接ノードを取得
        node_count = 0
        for d_lat, d_lon in directions:
            neighbor_node = quadtree.get_leaf_node_by_point(d_lat, d_lon)
            if neighbor_node and len(neighbor_node.points) > 0:
                neighbor_points.extend(neighbor_node.points)
                node_count += 1

        # 平均値で埋める
        #if len(neighbor_points) > 3:
        #  avg_lat, avg_lon, avg_depth = average_interpolate_value(mid_lat, mid_lon, neighbor_points)
        #  node.points.append({'lat': avg_lat, 'lon': avg_lon, 'depth': avg_depth})

        # IDW補間 (inverse distance weighted algorithm)
        if node_count > 2 and len(neighbor_points) > 3:
            avg_lat, avg_lon, avg_depth = idw_interpolate_value(mid_lat, mid_lon, neighbor_points)
            node.points.append({'lat': avg_lat, 'lon': avg_lon, 'depth': avg_depth})

        # TODO: 平均ではなくクリギングなどの高度な補間手法を使う


def average_interpolate_value(neighbor_points):
    """
    単純平均で値を補間する
    target_lat, target_lon: 補間したい座標
    neighbor_points: {'lat', 'lon', 'depth'}を持つ辞書のリスト
    戻り値: (lat, lon, depth) の補間値
    """
    if not neighbor_points:
        return None, None, None
    avg_lat = sum(p['lat'] for p in neighbor_points) / len(neighbor_points)
    avg_lon = sum(p['lon'] for p in neighbor_points) / len(neighbor_points)
    avg_depth = sum(p['depth'] for p in neighbor_points) / len(neighbor_points)
    return avg_lat, avg_lon, avg_depth


def idw_interpolate_value(target_lat, target_lon, neighbor_points, power=2):
    """
    inverse distance weighted (IDW) algorithm で値を補間する
    target_lat, target_lon: 補間したい座標
    neighbor_points: {'lat', 'lon', 'depth'}を持つ辞書のリスト
    power: 距離のべき乗（通常は2）
    戻り値: (lat, lon, depth) の補間値
    """
    weights = []
    values = []
    for p in neighbor_points:
        dist = ((target_lat - p['lat'])**2 + (target_lon - p['lon'])**2)**0.5
        if dist == 0:
            # 距離0ならその値をそのまま使う
            return p['lat'], p['lon'], p['depth']
        w = 1.0 / (dist ** power)
        weights.append(w)
        values.append(p['depth'])
    if not weights:
        return None, None, None
    depth_idw = np.average(values, weights=weights)
    avg_lat = np.average([p['lat'] for p in neighbor_points], weights=weights)
    avg_lon = np.average([p['lon'] for p in neighbor_points], weights=weights)
    return avg_lat, avg_lon, depth_idw




if __name__ == '__main__':

    def main():
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

        # 入力CSVファイルをPandasのデータフレームとして読み込む
        df = load_csv(input_file_path)
        if df is None:
            logger.error(f"CSVファイルの読み込みに失敗しました: {input_file_path}")
            return

        # STEP1
        # 細かい領域で四分木を作成して、データを集約する

        # 四分木を作成
        Quadtree.MAX_POINTS = 3        # デフォルトは6
        Quadtree.MIN_GRID_WIDTH = 2.0  # デフォルトは2.0メートル
        quadtree = create_quadtree_from_df(df)
        # 四分木の統計情報を表示
        logger.info(f"Initial Quadtree stats\n{quadtree.get_stats_text()}\n")

        # 最も深いレベルにあるリーフノードのポイントを平均化して集約
        quadtree.aggregate_deepest_node_points()
        logger.info("Aggregated deepest node points.")
        logger.info(f"Post-aggregation Quadtree stats\n{quadtree.get_stats_text()}\n")


        # STEP2
        # 空白の葉ノードについては、3個以上の隣接ノードに値があれば補間して埋める

        interpolate_empty_leaf_nodes(quadtree)

        # STEP3
        # 大きめの領域で四分木を作成して、密度の薄いノードを対象に補間する
        Quadtree.MAX_POINTS = 6         # デフォルトは6
        Quadtree.MIN_GRID_WIDTH = 20.0  # 10～20mの領域に収まる
        quadtree.rebuild()
        logger.info(f"Rebuilt Quadtree stats\n{quadtree.get_stats_text()}\n")

        # 最も深いレベルを除く、ポイントを持つ全てのリーフノードを取得
        non_deepest_nodes = [
            node for node in quadtree.get_nonempty_leaf_nodes()
            if node.level < quadtree.get_deepest_level()
        ]
        logger.info(f"Interpolating {len(non_deepest_nodes)} non-deepest leaf nodes...")

        # これらノードが持つポイントを取り出して、N, W, E, Sの4点を追加して補間する
        # 4m四方の領域を考慮する
        directions = [[0.0, 4.0], [4.0, 0.0], [0.0, -4.0], [-4.0, 0.0]]  # 北、東、南、西
        for node in non_deepest_nodes:
            for point in node.points:
                lat = point['lat']
                lon = point['lon']
                # 4方向に4m移動した点を追加、深さはその点の値を使う
                for d in directions:
                    new_lat = lat + d[0] * quadtree.lat_per_meter
                    new_lon = lon + d[1] * quadtree.lon_per_meter
                    new_point = {'lat': new_lat, 'lon': new_lon, 'depth': point['depth']}
                    quadtree.insert(new_point)
        logger.info("Inserted N, E, S, W points for interpolation.")
        logger.info(f"Post-insertion Quadtree stats\n{quadtree.get_stats_text()}\n")


        # STEP4
        # ポイントが増えたので、もう一度細かい領域で四分木を作成して、データを集約する

        Quadtree.MAX_POINTS = 3        # デフォルトは6
        Quadtree.MIN_GRID_WIDTH = 2.0  # デフォルトは2.0メートル
        quadtree.rebuild()

        # 最も深いレベルにあるリーフノードのポイントを平均化して集約
        quadtree.aggregate_deepest_node_points()
        logger.info("Aggregated deepest node points.")
        logger.info(f"Post-aggregation Quadtree stats\n{quadtree.get_stats_text()}\n")

        # 点群をCSVファイルに保存する
        quadtree.save_to_csv(output_file_path)
        logger.info(f"Points data saved to: {output_file_path}")

    #
    # 実行
    #
    main()
