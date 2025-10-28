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

    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.ensemble import IsolationForest

    # KDTreeを使って高速に近傍点を探索する
    from scipy.spatial import cKDTree

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
file_handler = logging.FileHandler(log_dir.joinpath(log_file), 'a+')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

#
# ここからスクリプト
#

def local_outlier_factor(df: pd.DataFrame, n_neighbors: int = 20, features: list = ["lat", "lon"]) -> pd.DataFrame:
    """Local Outlier Factor (LOF)を使用して外れ値を検出し、除去したデータフレームを返す関数。

    Args:
        df (pd.DataFrame): 入力データフレーム。'lat'および'lon'列を含む必要があります。
        n_neighbors (int, optional): 近傍点の数. デフォルトは20.
        features (list, optional): 使用する特徴量のリスト. デフォルトは["lat", "lon"].

    Returns:
    """

    # n_neighborsはデータ数が数百件以下のような少ない場合は5～10が推奨され、
    # 数千件以上のような多い場合は20～50が推奨される。
    # scikit-learnのデフォルトは20
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html

    # LOFモデルの作成と適合
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)

    # 特徴量を指定してデータを抽出（デフォルトは緯度と経度）
    X = df[features]

    # 学習させる
    lof.fit(X)

    # 予測を行う
    # 各サンプルの予測結果。1は正常、-1は外れ値を示す。
    predicted: np.ndarray = lof.fit_predict(X)

    # 外れ値を除去したデータフレームに置き換える
    df = df[predicted == 1].reset_index(drop=True)

    # 結果を返す
    return df


def isolation_forest(df: pd.DataFrame, features: list = ["lat", "lon"]) -> pd.DataFrame:
    """
    Isolation Forestを使用して外れ値を検出し、除去したデータフレームを返す関数。

    Args:
        df (pd.DataFrame): 入力データフレーム。'lat'および'lon'列を含む必要があります。
        features (list, optional): 使用する特徴量のリスト. デフォルトは["lat", "lon"].

    Returns: pd.DataFrame: 外れ値を除去したデータフレーム。
    """
    isof = IsolationForest(
        contamination='auto',
        n_estimators=100,
        random_state=42,
    )

    # 特徴量を指定してデータを抽出（デフォルトは緯度と経度）
    X = df[features]

    # 学習させる
    isof.fit(X)

    # 予測を行う
    # 各サンプルの予測結果。1は正常、-1は外れ値を示す。
    predicted: np.ndarray = isof.predict(X)

    # 外れ値を除去したデータフレームに置き換える
    df = df[predicted == 1].reset_index(drop=True)

    # 予測結果を返す
    return df


def spatial_depth_outlier(df, radius_m=10.0, z_thresh=3.0, min_neighbors=3) -> pd.DataFrame:
    """
    (lat, lon)でcKDTreeを使い、半径radius_mメートル以内の近傍点のdepth分布から外れ値を判定、除去する
    """
    # 経度1度をメートル座標に変換
    METERS_PER_DEG_LAT = 111000.0

    # 緯度1度をメートル座標に変換
    # 緯度によって変わるので、df['lat']の平均値を使って経度1度あたりの距離を計算する
    mean_lat = df['lat'].mean()
    METERS_PER_DEG_LON = 111320.0 * np.cos(np.deg2rad(mean_lat))

    # 座標をメートル単位に変換
    x = df['lon'].values * METERS_PER_DEG_LON
    y = df['lat'].values * METERS_PER_DEG_LAT
    coords = np.column_stack([x, y])

    tree = cKDTree(coords)
    depths = df["depth"].values

    outlier_mask = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        neighbor_indices = tree.query_ball_point(coords[i], r=radius_m)
        neighbor_indices = [idx for idx in neighbor_indices if idx != i]  # 自分自身を除く

        # 円の中に含まれる近傍点が少ない場合は異常値判定しない
        if len(neighbor_indices) <= min_neighbors:
            outlier_mask[i] = True
            continue

        neighbor_depths = depths[neighbor_indices]
        mean = neighbor_depths.mean()
        std = neighbor_depths.std()
        if std == 0:
            outlier_mask[i] = True
        else:
            z = abs(depths[i] - mean) / std
            outlier_mask[i] = z < z_thresh

    df = df[outlier_mask].reset_index(drop=True)
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

        # CSVファイルをPandasのデータフレームとして読み込む
        try:
            df = pd.read_csv(input_file_path)
            logger.info(f"describe() --- 入力データ\n{df.describe().to_markdown()}\n")
        except Exception as e:
            logger.error(f"CSVファイルの読み込みに失敗しました：{str(e)}")
            return

        #
        # ["lat", "lon"]を特徴量としてLOFで外れ値を検出する(LOF)
        # これにより、位置情報が非常に離れている点を除去する
        #
        df = local_outlier_factor(df, n_neighbors=20, features=["lat", "lon"])
        logger.info(f"describe() --- 位置の外れ値を削除後\n{df.describe().to_markdown()}\n")

        #
        # 水深の異常値を検出、除去する
        #
        df = spatial_depth_outlier(df, radius_m=10.0, z_thresh=3.0, min_neighbors=3)
        logger.info(f"describe() --- 水深の外れ値を削除後\n{df.describe().to_markdown()}\n")

        #
        # メディアンフィルタで深さを滑らかにする（異常値を修正する）
        #
        #df = median_filter_outlier(df, radius_m=5.0)
        #logger.info(f"describe() --- メディアンフィルタ適用後\n{df.describe().to_markdown()}\n")

        #
        # 外れ値を除去したデータフレームをCSVファイルに保存する
        #
        try:
            df.to_csv(output_file_path, index=False)
            logger.info(f"外れ値を除去したデータフレームを保存しました: {output_filename}")
        except Exception as e:
            logger.error(f"CSVファイルの保存に失敗しました：{str(e)}")

    #
    # 実行
    #
    main()
