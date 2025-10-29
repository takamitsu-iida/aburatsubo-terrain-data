#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeeperのGPSデータを入力として受け取り、水深の値を滑らかにするフィルタを適用した新しいCSVファイルを作成するスクリプト。


"""

# スクリプトを引数無しで実行したときのヘルプに使うデスクリプション
SCRIPT_DESCRIPTION = 'apply median filter to Deeper GPS data'

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
    import pandas as pd

    # KDTreeを使って高速に近傍点を探索する
    from scipy.spatial import cKDTree

except ImportError as e:
    logging.error("必要なモジュールがインストールされていません。pandasおよびscikit-learnをインストールしてください。")
    sys.exit(1)

#
# ローカルファイルからインポート
#
try:
    from load_save_csv import load_csv, save_csv
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

# ディレクトリ
data_dir = app_home.joinpath("data")

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
file_handler = logging.FileHandler(log_dir.joinpath(log_file), 'a+')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

#
# ここからスクリプト
#

def median_filter_outlier(df: pd.DataFrame, radius_m: float = 5.0) -> pd.DataFrame:
    """
    メディアンフィルタで深度の異常値を修正する(平準化する)
    radius_m: 近傍探索の半径（メートル単位）
    """
    # 経度をメートル座標に変換
    METERS_PER_DEG_LAT = 111000.0

    # 緯度をメートル座標に変換（df['lat']の平均値を使って経度1度あたりの距離を計算）
    mean_lat = df['lat'].mean()
    METERS_PER_DEG_LON = 111320.0 * np.cos(np.deg2rad(mean_lat))

    x = df['lon'].values * METERS_PER_DEG_LON
    y = df['lat'].values * METERS_PER_DEG_LAT

    # 2次元の座標配列を作成
    coords = np.column_stack([x, y])

    # KD-Treeを構築
    tree = cKDTree(coords)

    # depth列を直接置き換える
    new_depths = np.empty(len(df))
    for i in range(len(df)):
        neighbor_indices = tree.query_ball_point(coords[i], r=radius_m)
        neighbor_depths = df.loc[neighbor_indices, 'depth']
        median_val = np.median(neighbor_depths)
        new_depths[i] = median_val

    df['depth'] = new_depths

    return df


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

        input_file_path = Path(data_dir, args.input)
        if not input_file_path.exists():
            logger.error(f"入力ファイルが存在しません: {input_file_path}")
            return

        output_filename = args.output
        output_file_path = Path(data_dir, output_filename)

        # 入力CSVファイルをPandasのデータフレームとして読み込む
        df = load_csv(input_file_path)
        if df is None:
            logger.error("CSVファイルの読み込みに失敗しました。")
            return

        #
        # メディアンフィルタで深さを滑らかにする（異常値を修正する）
        #
        df = median_filter_outlier(df, radius_m=5.0)
        logger.info(f"describe() --- メディアンフィルタ適用後\n{df.describe().to_markdown()}\n")

        #
        # データフレームをCSVファイルに保存する
        #
        save_csv(df, output_file_path)
        logger.info(f"メディアンフィルタを適用したデータフレームを保存しました: {output_filename}")

    #
    # 実行
    #
    main()
