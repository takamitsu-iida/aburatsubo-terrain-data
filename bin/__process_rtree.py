#!/usr/bin/env python

# R-Treeを使ってデータを処理するスクリプトです。

#
# 標準ライブラリのインポート
#
import logging
import math
import os
import sys

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# WSL1 固有の numpy 警告を抑制
# https://github.com/numpy/numpy/issues/18900
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, module=r"numpy.*", message=r"Signature b")

#
# 外部ライブラリのインポート
#
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
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

def draw_mbr(ax, bounds, color, alpha=0.3, label=None):
    """最小境界矩形 (MBR) を描画するヘルパー関数"""
    min_x, min_y, max_x, max_y = bounds
    rect = plt.Rectangle(
        (min_x, min_y),
        max_x - min_x,
        max_y - min_y,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        fill=True,
        linewidth=2,
        linestyle='--',
        label=label
    )
    ax.add_patch(rect)


if __name__ == "__main__":


    def main():
        # 1. データ読み込み
        csv_path = Path(__file__).parent.parent.joinpath("data/ALL_depth_map_data_202510_dd_ol.csv")
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] == 3:
            df.columns = ["lat", "lon", "depth"]
        elif df.shape[1] == 4:
            df.columns = ["lat", "lon", "depth", "time"]
            del df["time"]

        # ルートMBR
        min_lat, max_lat = df["lat"].min(), df["lat"].max()
        min_lon, max_lon = df["lon"].min(), df["lon"].max()
        root_mbr = (min_lon, min_lat, max_lon, max_lat)

        # クラスタ数（中間ノード数）を指定
        N_CLUSTERS = 50
        coords = df[["lon", "lat"]].values
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
        labels = kmeans.fit_predict(coords)

        # 各クラスタごとにMBRを計算
        cluster_mbrs = []
        for i in range(N_CLUSTERS):
            cluster_points = coords[labels == i]
            if len(cluster_points) == 0:
                continue
            min_lon, min_lat = cluster_points.min(axis=0)
            max_lon, max_lat = cluster_points.max(axis=0)
            cluster_mbrs.append((min_lon, min_lat, max_lon, max_lat))

        # 3. 可視化
        fig, ax = plt.subplots(figsize=(10, 8))

        # ルートMBR
        draw_mbr(ax, root_mbr, color='grey', alpha=0.2, label='Root Node MBR')

        # クラスタMBR（中間ノード）
        cluster_colors = ['purple', 'orange', 'green', 'blue', 'red', 'cyan', 'magenta', 'yellow']
        for idx, mbr in enumerate(cluster_mbrs):
            draw_mbr(ax, mbr, color=cluster_colors[idx % len(cluster_colors)], alpha=0.3, label=f'Cluster MBR {idx+1}')

        # GPSデータ点
        ax.scatter(df["lon"], df["lat"], c=labels, cmap="tab10", s=10, label="GPS Data Points")

        # 装飾（英語）
        ax.set_title("R-Tree Spatial Index Visualization (Clustered Intermediate Nodes)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')
        # ax.legend(loc='lower left')

        output_image_path = Path(__file__).parent.parent.joinpath("img/rtree_mbr.png")
        plt.tight_layout()
        plt.savefig(output_image_path)
        logger.info(f"Image saved: {output_image_path}")

    def draw_mbr_hierarchy_test():
        """
        R-Treeの概念を示すために、階層的なMBR（最小境界矩形）を描画するスクリプト
        """

        # 1. データ点（GPSログ）の生成
        np.random.seed(42)

        # 4つの主要なクラスターをシミュレート
        points_a = np.random.uniform(2, 4, (10, 2))
        points_b = np.random.uniform(6, 8, (10, 2))
        points_c = np.random.uniform(2, 4, (5, 2)) + [5, 0]
        points_d = np.random.uniform(6, 8, (5, 2)) + [5, 0]
        all_points = np.vstack([points_a, points_b, points_c, points_d])

        # 2. R-Treeの階層的なMBRを仮想的に定義
        # MBRの色と透明度を設定
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
        alphas = [0.2, 0.4, 0.6]

        # --- リーフノード層 (レベル 1 MBR) ---
        # 各クラスタを囲むMBR
        mbr1_a = (points_a[:, 0].min(), points_a[:, 1].min(), points_a[:, 0].max(), points_a[:, 1].max())
        mbr1_b = (points_b[:, 0].min(), points_b[:, 1].min(), points_b[:, 0].max(), points_b[:, 1].max())
        mbr1_c = (points_c[:, 0].min(), points_c[:, 1].min(), points_c[:, 0].max(), points_c[:, 1].max())
        mbr1_d = (points_d[:, 0].min(), points_d[:, 1].min(), points_d[:, 0].max(), points_d[:, 1].max())
        mbrs_level1 = [mbr1_a, mbr1_b, mbr1_c, mbr1_d]


        # --- 中間ノード層 (レベル 2 MBR) ---
        # MBR_A: mbr1_aとmbr1_cを包含
        coords_a_c = np.vstack([points_a, points_c])
        mbr2_a_c = (coords_a_c[:, 0].min(), coords_a_c[:, 1].min(), coords_a_c[:, 0].max(), coords_a_c[:, 1].max())
        # MBR_B: mbr1_bとmbr1_dを包含
        coords_b_d = np.vstack([points_b, points_d])
        mbr2_b_d = (coords_b_d[:, 0].min(), coords_b_d[:, 1].min(), coords_b_d[:, 0].max(), coords_b_d[:, 1].max())
        mbrs_level2 = [mbr2_a_c, mbr2_b_d]

        # --- ルートノード層 (レベル 3 MBR) ---
        # 全データを包含するMBR (データ全体の範囲)
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        mbr3_root = (min_x, min_y, max_x, max_y)


        # 3. 描画
        fig, ax = plt.subplots(figsize=(10, 8))

        # 3.1. ルートMBR (データ全体を包含)
        draw_mbr(ax, mbr3_root, color='grey', alpha=alphas[2], label='Root Node MBR')

        # 3.2. 中間ノードMBR
        draw_mbr(ax, mbr2_a_c, color='purple', alpha=alphas[1], label='Intermediate Node MBR (Group 1)')
        draw_mbr(ax, mbr2_b_d, color='darkorange', alpha=alphas[1], label='Intermediate Node MBR (Group 2)')

        # 3.3. リーフノードMBR
        draw_mbr(ax, mbr1_a, color=colors[0], alpha=alphas[0])
        draw_mbr(ax, mbr1_b, color=colors[1], alpha=alphas[0])
        draw_mbr(ax, mbr1_c, color=colors[2], alpha=alphas[0])
        draw_mbr(ax, mbr1_d, color=colors[3], alpha=alphas[0])

        # 3.4. 点データ（GPSログ）の描画
        ax.plot(all_points[:, 0], all_points[:, 1], 'ko', markersize=4, label='GPS Data Points')

        # 4. 図の装飾（英語表示に変更）
        ax.set_title("R-Tree Spatial Index Concept (MBR Hierarchy)")
        ax.set_xlabel("Latitude (Lat)")
        ax.set_ylabel("Longitude (Lon)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='lower left')

        # 画像として保存
        # matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # デフォルト英語フォント
        # matplotlib.rcParams['font.family'] = 'Liberation Sans'  # Ubuntu標準フォント
        plt.rcParams['font.family'] = 'sans-serif'

        output_image_path = image_dir.joinpath("rtree_mbr.png")
        plt.tight_layout()
        plt.savefig(output_image_path)
        logger.info(f"画像を保存しました: {output_image_path}")


    main()