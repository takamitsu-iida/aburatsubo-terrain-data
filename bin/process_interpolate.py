#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeeperのGPSデータを入力として受け取り、補間処理を行うスクリプトです。


"""

# スクリプトを引数無しで実行したときのヘルプに使うデスクリプション
SCRIPT_DESCRIPTION = 'Interpolate Deeper GPS data using Quadtree'

#
# 標準ライブラリのインポート
#
import argparse
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
    import pandas as pd
    import matplotlib.pyplot as plt

    import numpy as np
    from pykrige.ok import OrdinaryKriging
    from scipy.interpolate import griddata
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    sys.exit(1)

#
# ローカルファイルからインポート
#
try:
    from load_save_csv import load_csv, save_points_as_csv
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
file_handler = logging.FileHandler(os.path.join(log_dir, log_file), 'a+')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

#
# ここからスクリプト
#


def __process_kriging(df: pd.DataFrame):

    N_TILES_PER_SIDE = 5 # 1辺あたりのタイル数 (合計 5x5 = 25領域に分割)

    # 全体の範囲を計算
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    lat_min, lat_max = df['lat'].min(), df['lat'].max()

    # 領域の境界を定義
    lon_bins = np.linspace(lon_min, lon_max, N_TILES_PER_SIDE + 1)
    lat_bins = np.linspace(lat_min, lat_max, N_TILES_PER_SIDE + 1)

    # 結果格納用のリスト
    interpolated_results = []
    all_grid_coords = [] # 補間した格子点の座標を結合するために使用

    print("\n--- 領域ごとのクリギング処理を開始 ---")

    for i in range(N_TILES_PER_SIDE):
        for j in range(N_TILES_PER_SIDE):
            # 領域 (タイル) の境界
            current_lon_min, current_lon_max = lon_bins[i], lon_bins[i+1]
            current_lat_min, current_lat_max = lat_bins[j], lat_bins[j+1]

            # 該当領域のデータ点を抽出
            tile_data = df[
                (df['lon'] >= current_lon_min) & (df['lon'] < current_lon_max) &
                (df['lat'] >= current_lat_min) & (df['lat'] < current_lat_max)
            ].copy()

            if len(tile_data) < 4:
                # クリギングに必要な最低限の点数（通常は3点以上）がない場合はスキップ
                # または、単純な補間（例: 平均値）で埋める
                # print(f"領域 ({i}, {j}) はデータ点数が少なすぎます ({len(tile_data)}点)。スキップします。")
                continue

            lon_tile, lat_tile, depth_tile = tile_data['lon'].values, tile_data['lat'].values, tile_data['depth'].values
            print(f"処理中: 領域 ({i}, {j}), データ点数: {len(tile_data)}")

            # 3-1. クリギングモデルの構築
            # 異方性の設定は、必要に応じてここに追加します
            try:
                OK = OrdinaryKriging(
                    lon_tile, lat_tile, depth_tile,
                    variogram_model='spherical',
                    verbose=False,
                    enable_plotting=False
                )

                # 3-2. 補間するグリッドの定義
                # 各領域で均一なメッシュを作成 (例: 20x20)
                GRID_POINTS = 20
                tile_grid_lon = np.linspace(current_lon_min, current_lon_max, GRID_POINTS)
                tile_grid_lat = np.linspace(current_lat_min, current_lat_max, GRID_POINTS)

                # 3-3. 補間の実行
                z_tile, ss_tile = OK.execute('grid', tile_grid_lon, tile_grid_lat)

                # 結果を DataFrame に格納
                tile_grid_x, tile_grid_y = np.meshgrid(tile_grid_lon, tile_grid_lat)

                # 緯度・経度と補間結果を結合してリストに追加
                interpolated_results.append(pd.DataFrame({
                    'lon': tile_grid_x.ravel(),
                    'lat': tile_grid_y.ravel(),
                    'depth_kriging': z_tile[0].ravel()
                }))

            except Exception as e:
                print(f"領域 ({i}, {j}) でクリギングエラーが発生しました: {e}")
                pass # エラーが発生した領域はスキップまたは単純補間で処理

    # ----------------------------------------------------
    # 4. 結果の結合
    # ----------------------------------------------------
    if interpolated_results:
        final_result_df = pd.concat(interpolated_results).drop_duplicates(subset=['lon', 'lat']).reset_index(drop=True)
        print("\n--- 補間結果の結合が完了しました ---")
        print(f"最終グリッドデータ件数: {len(final_result_df)}")

        # 可視化の準備
        # griddataを使って、結合された不規則な点を規則的なグリッドに再補間（表示用）
        # ただし、クリギングで得た値の精度が落ちるため、本来はクリギングの結果をそのまま利用すべき

        # 最終的なメッシュの作成 (可視化用)
        VIS_POINTS = 100
        vis_lon = np.linspace(lon_min, lon_max, VIS_POINTS)
        vis_lat = np.linspace(lat_min, lat_max, VIS_POINTS)

        # griddata（線形補間）で最終結果を可視化用のメッシュに変換
        # 注意: ここは表示の滑らかさを得るためであり、クリギングの結果が最良です
        final_depth_grid = griddata(
            final_result_df[['lon', 'lat']].values,
            final_result_df['depth_kriging'].values,
            (vis_lon[None,:], vis_lat[:,None]),
            method='linear'
        )

        # 5. 可視化
        plt.figure(figsize=(10, 8))
        plt.title("Tiled Kriging Interpolation Map")

        im = plt.contourf(vis_lon, vis_lat, final_depth_grid, levels=50, cmap='viridis')
        plt.colorbar(im, label='Interpolated Depth')

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()
    else:
        print("\n補間可能なデータ領域がありませんでした。")


def aggregate_leaf_nodes(quadtree: Quadtree):
    # 最も深いノードについては、そのノード内のポイントの平均値に置き換える
    deepest_level = quadtree.get_deepest_level()
    nodes = [node for node in quadtree.get_leaf_nodes() if node.level == deepest_level and len(node.points) > 1]
    for node in nodes:
        avg_point = node.average()
        node.points = [avg_point]

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
        for d_lat, d_lon in directions:
            neighbor_node = quadtree.get_leaf_node_by_point(d_lat, d_lon)
            if neighbor_node and len(neighbor_node.points) > 0:
                neighbor_points.extend(neighbor_node.points)

        # 平均値で埋める
        #if len(neighbor_points) > 3:
        #  avg_lat, avg_lon, avg_depth = average_interpolate_value(mid_lat, mid_lon, neighbor_points)
        #  node.points.append({'lat': avg_lat, 'lon': avg_lon, 'depth': avg_depth})

        # IDW補間 (inverse distance weighted algorithm)
        if len(neighbor_points) > 3:
            avg_lat, avg_lon, avg_depth = idw_interpolate_value(mid_lat, mid_lon, neighbor_points)
            node.points.append({'lat': avg_lat, 'lon': avg_lon, 'depth': avg_depth, 'epoch': 0})

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

        # 四分木を初期化
        quadtree = create_quadtree_from_df(df)

        # 四分木の統計情報を表示する
        logger.info(f"Initial Quadtree stats\n{quadtree.get_stats_text()}\n")

        # データ集約
        # 最も深いノードについては、そのノード内のポイントの平均値に置き換えることでデータを減らす
        aggregate_leaf_nodes(quadtree)

        # 集約後の四分木の統計情報を表示する
        logger.info(f"Aggregated Quadtree stats\n{quadtree.get_stats_text()}\n")

        # データ補間
        # ポイントのない葉ノードについては、隣接ノードに値があればその平均値で埋める
        interpolate_empty_leaf_nodes(quadtree)

        # 補間した四分木の統計情報を表示する
        logger.info(f"Extended Quadtree stats\n{quadtree.get_stats_text()}\n")

        # 点群をファイルに保存する
        points = []
        for node in quadtree.get_leaf_nodes():
            points.extend(node.points)
        logger.info(f"Points count: {len(points)}")

        # CSVファイルに保存する
        save_points_as_csv(points, output_file_path)
        logger.info(f"Points data saved to: {output_file_path}")

        # 四分木の可視化画像を保存する
        #output_image_filename = f"{input_file_path.stem}_qtree.png"
        #output_image_path = image_dir.joinpath(output_image_filename)
        #save_quadtree_image(quadtree=quadtree, filename=output_image_path, draw_points=False)

    #
    # 実行
    #
    main()
