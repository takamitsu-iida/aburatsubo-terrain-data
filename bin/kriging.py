#!/usr/bin/env python

# 四分木を実装したスクリプトです。

#
# 標準ライブラリのインポート
#
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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 空間補間・近傍探索
    from scipy.spatial import cKDTree

    # クラスタリング
    from sklearn.cluster import DBSCAN

    # クリギング補間
    from pykrige.ok import OrdinaryKriging

except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    sys.exit(1)

#
# ローカルファイルからインポート
#
try:
    from load_save_csv import load_csv, save_csv
except ImportError as e:
    print(f"ローカルモジュールのインポートに失敗しました: {e}")
    sys.exit(1)


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


def split_region_by_quadtree(df, min_points=30, min_size_m=30, max_depth=8):
    """
    四分木で空間分割し、各リーフノードごとに十分な点があり、かつ領域サイズがmin_size_m以上ならDataFrameとして返す
    """
    from collections import deque
    from geopy.distance import distance

    class QuadNode:
        def __init__(self, lat1, lon1, lat2, lon2, depth=0):
            self.bounds = (lat1, lon1, lat2, lon2)
            self.depth = depth
            self.children = []
            self.indices = []

        def size_m(self):
            # 領域の幅・高さ（メートル）
            lat1, lon1, lat2, lon2 = self.bounds
            width = distance((lat1, lon1), (lat1, lon2)).meters
            height = distance((lat1, lon1), (lat2, lon1)).meters
            return min(width, height)

    lat1, lat2 = df['lat'].min(), df['lat'].max()
    lon1, lon2 = df['lon'].min(), df['lon'].max()
    root = QuadNode(lat1, lon1, lat2, lon2)
    root.indices = list(range(len(df)))

    result_nodes = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        # 領域サイズがmin_size_m未満なら分割しない
        if node.depth >= max_depth or len(node.indices) <= min_points or node.size_m() <= min_size_m:
            result_nodes.append(node)
            continue
        mid_lat = (node.bounds[0] + node.bounds[2]) / 2
        mid_lon = (node.bounds[1] + node.bounds[3]) / 2
        quads = [
            (node.bounds[0], node.bounds[1], mid_lat, mid_lon),  # SW
            (node.bounds[0], mid_lon, mid_lat, node.bounds[3]),  # SE
            (mid_lat, node.bounds[1], node.bounds[2], mid_lon),  # NW
            (mid_lat, mid_lon, node.bounds[2], node.bounds[3]),  # NE
        ]
        for qlat1, qlon1, qlat2, qlon2 in quads:
            idx = [
                i for i in node.indices
                if (qlat1 <= df.iloc[i]['lat'] < qlat2) and (qlon1 <= df.iloc[i]['lon'] < qlon2)
            ]
            if idx:
                child = QuadNode(qlat1, qlon1, qlat2, qlon2, node.depth + 1)
                child.indices = idx
                node.children.append(child)
                queue.append(child)
    clusters = [df.iloc[node.indices].reset_index(drop=True) for node in result_nodes if len(node.indices) >= 4]
    return clusters


def grid_and_kriging(df, grid_size_m=10):
    # 領域範囲
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    min_lon, max_lon = df['lon'].min(), df['lon'].max()
    mean_lat = df['lat'].mean()
    METERS_PER_DEG_LAT = 111000.0
    METERS_PER_DEG_LON = 111320.0 * np.cos(np.deg2rad(mean_lat))
    # グリッド生成
    width_m = (max_lon - min_lon) * METERS_PER_DEG_LON
    height_m = (max_lat - min_lat) * METERS_PER_DEG_LAT
    n_x = max(2, int(width_m // grid_size_m))
    n_y = max(2, int(height_m // grid_size_m))
    grid_lon = np.linspace(min_lon, max_lon, n_x)
    grid_lat = np.linspace(min_lat, max_lat, n_y)
    grid_x = grid_lon * METERS_PER_DEG_LON
    grid_y = grid_lat * METERS_PER_DEG_LAT
    # クリギング
    x = df['lon'].values * METERS_PER_DEG_LON
    y = df['lat'].values * METERS_PER_DEG_LAT
    z = df['depth'].values
    OK = OrdinaryKriging(x, y, z, variogram_model='linear', verbose=False, enable_plotting=False)
    z_grid, ss = OK.execute('grid', grid_x, grid_y)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    result_df = pd.DataFrame({
        'lat': grid_yy.flatten() / METERS_PER_DEG_LAT,
        'lon': grid_xx.flatten() / METERS_PER_DEG_LON,
        'depth': z_grid.flatten()
    })
    return result_df



def visualize_kriged_df(kriged_df, image_dir, filename="kriging_interpolated_depth.png"):
    """
    kriged_df（lon, lat, depthを持つDataFrame）を散布図として画像ファイルに保存する関数
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(kriged_df['lon'], kriged_df['lat'], c=kriged_df['depth'], cmap='viridis', s=20)
    plt.colorbar(sc, ax=ax, label='Depth')
    ax.set_title("Kriging Interpolated Depth Map (High Density Region)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal', adjustable='box')

    output_image_path = image_dir.joinpath(filename)
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close(fig)
    logger.info(f"Image saved: {output_image_path}")



def visualize_all_kriged_df(kriged_dfs, image_dir, filename="kriging_interpolated_depth_all.png"):
    """
    複数のkriged_df（lon, lat, depthを持つDataFrame）をまとめて散布図として画像ファイルに保存する関数
    """
    # 全クラスタの補間結果を結合
    all_df = pd.concat(kriged_dfs, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(all_df['lon'], all_df['lat'], c=all_df['depth'], cmap='viridis', s=20)
    plt.colorbar(sc, ax=ax, label='Depth')
    ax.set_title("Kriging Interpolated Depth Map (All Regions)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal', adjustable='box')

    output_image_path = image_dir.joinpath(filename)
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close(fig)
    logger.info(f"Image saved: {output_image_path}")



def save_kriged_df_to_csv(kriged_dfs, output_csv_path):
    """
    複数のkriged_df（lon, lat, depthを持つDataFrame）をまとめてCSVファイルに保存する関数
    """
    all_df = pd.concat(kriged_dfs, ignore_index=True)
    all_df.to_csv(output_csv_path, index=False)
    logger.info(f"CSV saved: {output_csv_path}")


if __name__ == '__main__':

    def main():

        data_filename = "ALL_depth_map_data_202510_de_dd_ol.csv"
        data_path = data_dir.joinpath(data_filename)
        if not data_path.exists():
            logger.error("File not found: %s" % data_path)
            return

        # read_csv()を使った方が軽いが、PandasのDataFrameで処理を統一する
        df = load_csv(data_path)
        if df is None:
            logger.error(f"データの読み込みに失敗しました: {data_path}")
            return

        # 四分木で領域分割
        clusters = split_region_by_quadtree(df, min_points=30, max_depth=6)

        logger.info(f"Quadtree leaf clusters: {len(clusters)}")
        kriged_dfs = []
        for i, dense_df in enumerate(clusters):
            logger.info(f"Leaf {i+1}: points = {len(dense_df)}")
            kriged_df = grid_and_kriging(dense_df, grid_size_m=10)
            kriged_dfs.append(kriged_df)

        # 全体で一つの可視化画像を作成
        visualize_all_kriged_df(kriged_dfs, image_dir, filename="kriging_interpolated_depth_all.png")

        # CSVファイルとして保存
        output_csv_path = data_dir.joinpath("kriging_data.csv")
        save_kriged_df_to_csv(kriged_dfs, output_csv_path)

    #
    # 実行
    #
    main()
