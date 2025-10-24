#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeeperのGPSデータを入力として受け取り、異常値を排除した新しいCSVファイルを作成するスクリプト。


"""

# スクリプトを引数無しで実行したときのヘルプに使うデスクリプション
SCRIPT_DESCRIPTION = 'drop outliers from Deeper GPS data'

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

    from sklearn.neighbors import LocalOutlierFactor  # process_outlier()
    from sklearn.ensemble import IsolationForest  # process_outlier()

except ImportError as e:
    logging.error("必要なモジュールがインストールされていません。pandasおよびscikit-learnをインストールしてください。")
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
USE_FILE_HANDLER = True
if USE_FILE_HANDLER:
    file_handler = logging.FileHandler(log_dir.joinpath(log_file), 'a+')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

#
# ここからスクリプト
#

def snap_to_one_meter_grid(latitude, longitude):
    """
    GPS座標を概算で1m単位のグリッドに吸着させる関数。

    精度を優先しないため、特定の緯度（例：北緯35度付近）での
    緯度・経度の概算のメートル距離を使用する。

    概算値:
    - 緯度1度あたり：約 111,000 メートル
    - 経度1度あたり（北緯35度付近）：約 91,287 メートル
    """

    # 概算の変換係数 (1度あたりのメートル数)
    # 緯度方向 (北緯・南緯に関わらずほぼ一定): 約111,000 m/度
    METERS_PER_DEG_LAT = 111000.0

    # 経度方向 (緯度によって変化する。ここでは北緯35度付近を想定): 約91,287 m/度
    # 経度1度あたりの距離 = 111,320 * cos(緯度)
    # 例：cos(35度) * 111,320 ≈ 91,287
    METERS_PER_DEG_LON = 91287.0

    # --- 1. メートル単位の座標に変換 (基準点からの相対距離として扱う) ---

    # 非常に大きな値になるのを避けるため、適当な「原点」を設定して相対距離を計算します。
    # ここでは、座標自体を相対値として使い、メートルに変換します。

    # 緯度をメートルに変換
    y_meters = latitude * METERS_PER_DEG_LAT

    # 経度をメートルに変換
    x_meters = longitude * METERS_PER_DEG_LON

    # --- 2. 1m単位で丸める (グリッドへの吸着) ---

    # メートル座標を最も近い整数値 (1m単位) に丸める
    snapped_x_meters = np.round(x_meters)
    snapped_y_meters = np.round(y_meters)

    # --- 3. 再度、緯度・経度に逆変換 ---

    # 丸めたメートル座標を元の緯度・経度に戻す
    snapped_latitude = snapped_y_meters / METERS_PER_DEG_LAT
    snapped_longitude = snapped_x_meters / METERS_PER_DEG_LON

    return snapped_latitude, snapped_longitude


def local_outlier_factor(df: pd.DataFrame, n_neighbors: int = 20) -> np.ndarray:
    """Local Outlier Factor (LOF)を使用して外れ値を検出する

    Args:
        df (pd.DataFrame): 入力データフレーム。'lat'および'lon'列を含む必要があります。
        n_neighbors (int, optional): 近傍点の数. デフォルトは20.

    Returns:        np.ndarray: 各サンプルの予測結果。1は正常、-1は外れ値を示す。
    """

    # n_neighborsはデータ数が数百件以下のような少ない場合は5～10が推奨され、
    # 数千件以上のような多い場合は20～50が推奨される。
    # scikit-learnのデフォルトは20
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html

    # LOFモデルの作成と適合
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)

    # 特徴量として緯度と経度を使用
    X = df[["lat", "lon"]]

    # 学習させる
    lof.fit(X)

    # 予測を行う
    predicted = lof.fit_predict(X)

    # 予測結果を返す
    return predicted


def isolation_forest(df: pd.DataFrame) -> np.ndarray:
    isof = IsolationForest(
        contamination='auto',
        n_estimators=100,
        random_state=42,
    )
    X = df[["lat", "lon"]]
    # 学習させる
    isof.fit(X)
    predicted = isof.predict(X)
    return predicted


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

        # CSVファイルをPandasのデータフレームとして読み込む
        try:
            df = pd.read_csv(input_file_path)
            logger.info(f"describe()\n{df.describe().to_markdown()}")
        except Exception as e:
            logger.error(f"CSVファイルの読み込みに失敗しました：{str(e)}")
            return

        # 外れ値を検出する
        predicted = local_outlier_factor(df, n_neighbors=20)

        # 外れ値を除去したデータフレームを作成する
        df_cleaned = df[predicted == 1].reset_index(drop=True)

        logger.info(f"describe() --- 外れ値を削除後\n{df_cleaned.describe().to_markdown()}\n")

        # 外れ値を除去したデータフレームをCSVファイルに保存する
        try:
            df_cleaned.to_csv(output_file_path, index=False)
            logger.info(f"外れ値を除去したデータフレームを保存しました: {output_filename}")
        except Exception as e:
            logger.error(f"CSVファイルの保存に失敗しました：{str(e)}")

    #
    # 実行
    #
    main()
