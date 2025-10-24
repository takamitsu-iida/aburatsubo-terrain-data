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
import os
import sys

from pathlib import Path


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
# 外部ライブラリのインポート
#
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.neighbors import LocalOutlierFactor  # process_outlier()
    from sklearn.ensemble import IsolationForest  # process_outlier()

except ImportError as e:
    logger.exception(e)
    sys.exit(1)

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





if __name__ == '__main__':

    def rbf_example():
        from scipy.stats.qmc import Halton
        rng = np.random.default_rng()
        xobs = 2 * Halton(2, seed=rng).random(100) - 1
        yobs = np.sum(xobs, axis=1) * np.exp(-6 * np.sum(xobs**2, axis=1))
        xgrid = np.mgrid[-1:1:50j, -1:1:50j]
        xflat = xgrid.reshape(2, -1).T

        print(xgrid)

        yflat = RBFInterpolator(xobs, yobs)(xflat)

        ygrid = yflat.reshape(50, 50)
        fig, ax = plt.subplots()
        ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
        p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
        fig.colorbar(p)
        plt.savefig("3d.png")

    def generate_test_grid(df: pd.DataFrame):
        # テスト用座標
        TEST_COORD_NW = (35.1636, 139.6082)
        TEST_COORD_SE = (35.1622, 139.6099)

        extracted = df.query("lat > {} and lat < {} and lon > {} and lon < {}".format(
            TEST_COORD_SE[0], TEST_COORD_NW[0], TEST_COORD_NW[1], TEST_COORD_SE[1]))
        lat = extracted["lat"]
        lat = (lat - lat.mean()) / lat.std()
        lon = extracted["lon"]
        lon = (lon - lon.mean()) / lon.std()
        depth = extracted["depth"]
        lat_lon = np.stack([lat, lon], -1)
        rbf = RBFInterpolator(
            lat_lon, depth, kernel='thin_plate_spline', epsilon=2.0, neighbors=10)

        # RBF補間を作成
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html#scipy.interpolate.RBFInterpolator
        # rbf = RBFInterpolator(extracted[["lat", "lon"]], extracted["depth"], kernel='gaussian', epsilon=1.0, neighbors=1000)

        """
        linear : -r
        thin_plate_spline : r**2 * log(r)
        cubic : r**3
        quintic : -r**5
        multiquadric : -sqrt(1 + r**2)
        inverse_multiquadric : 1/sqrt(1 + r**2)
        inverse_quadratic : 1/(1 + r**2)
        gaussian : exp(-r**2)

        Default is 'thin_plate_spline'
        """

        # x, yの各軸に関して、等間隔の配列を作成する
        xi = np.linspace(lon.min(), lon.max(), 100)
        yi = np.linspace(lat.min(), lat.max(), 100)

        # メッシュグリッドに変換する
        xi, yi = np.meshgrid(xi, yi)
        xi_yi = np.stack([xi.ravel(), yi.ravel()], -1)

        # xi, yi の位置の zi を計算する
        zi = rbf(xi_yi)

        # ==============================================================================
        # 3 次元グラフ
        # ==============================================================================
        fig = plt.figure(figsize=(18, 7), dpi=200)  # 画像を作成する
        el = 55                                    # 視点高さを設定する

        ax = fig.add_subplot(231, projection='3d')  # 2 × 3 の 1 枚目に描画する

        # 曲面のプロット
        ax.plot_surface(xi, yi, zi)                      # サーフェスの描画
        ax.view_init(elev=el, azim=10)                   # ビューの設定
        ax.set_title('elev = ' + str(el) + ', deg = 10')  # タイトルの設定
        ax.set_xlabel('xi')                              # 軸ラベルの設定
        ax.set_ylabel('yi')                              # 軸ラベルの設定
        ax.set_zlabel('zi')                              # 軸ラベルの設定

        plt.savefig("3d.png")

    def read_file(filename, callback):
        try:
            with open(filename) as f:
                for line in f:
                    line = line.rstrip()
                    callback(line)
        except IOError as e:
            logger.exception(e)

    def line_callback(line):
        print(line)

    def load_csv(data_path):
        try:
            return pd.read_csv(data_path)
        except:
            return None

    def print_summary(df: pd.DataFrame):
        print("head")
        print(df.head(3))
        print("")
        print("tail")
        print(df.tail(3))
        print("")
        print("describe")
        # to_markdown() requires tabulate module
        print(df.describe().to_markdown())
        print("")


    def process_duplicated(df: pd.DataFrame):
        dfx = df[["lat", "lon", "depth"]]

        # keep=Falseにすると重複行は全て削除。デフォルトはkeep='first'
        dfx_uniq = df.drop_duplicates(subset=["lat", "lon"], keep=False)
        print("unique coordinate")
        print(dfx_uniq.describe().to_markdown())
        print("")

        dfx_duplicated = dfx.groupby(["lat", "lon"])[
            "depth"].mean().reset_index()
        print("duplicated coordinate")
        print(dfx_duplicated.describe().to_markdown())
        print("")

        df = pd.concat([dfx_uniq, dfx_duplicated]).reset_index(drop=True)
        print("combined data frame")
        print(df.describe().to_markdown())
        print("")

        return df

    def process_outlier(df: pd.DataFrame):

        def local_outlier_factor(n_neighbors=20):
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            X = df[["lat", "lon"]]
            lof.fit(X)
            predicted = lof.fit_predict(X)
            return predicted

        def isolation_forest():
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

        # return isolation_forest()
        return local_outlier_factor()

    def main() -> None:

        # 引数処理
        parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
        parser.add_argument('--input', type=str, required=True, help='dataディレクトリ直下のCSVファイル名')
        args = parser.parse_args()

        # 引数が何も指定されていない場合はhelpを表示して終了
        if not any(vars(args).values()):
            parser.print_help()
            return

        input_file_path = Path(data_dir, args.input)
        if not input_file_path.exists():
            logger.error(f"入力ファイルが存在しません: {input_file_path}")
            return

        # 拡張子を除いた名前
        input_basename = input_file_path.stem

        # 出力ファイル名は入力ファイル名に_dedupを付加したものとする
        output_filename = f"{input_basename}_dedup.csv"

        # 出力先のファイルパス
        output_file_path = Path(data_dir, output_filename)





        # オリジナルデータをファイルから読む

        # 古いデータはこれ
        # filename = "bathymetry_data.csv"

        # 新しいデータはこのファイル
        filename = "ALL_depth_map_data_202408.csv"
        file_path = os.path.join(data_dir, filename)
        df = load_csv(file_path)

        # CSVファイルには列名がないので、データフレームに列名を定義
        df.columns = ["lat", "lon", "depth", "time"]

        # 時刻の列は不要なので削除する
        del df["time"]

        # データのサマリを表示
        print("original data")
        print_summary(df)


        # 座標を1m単位に吸着させる
        stats = Stats(df)
        df = align_coord(df, stats)
        print("align coord")
        print(df.describe().to_markdown())
        print("")

        # 重複した座標のデータを削除する
        df = process_duplicated(df)

        # 重複を削除した状態の散布図を保存
        save_scatter(df, title="drop duplicated", filename="scatter_02.png")

        # 外れ値を除く処理を施す
        pred = process_outlier(df)

        # 外れ値データの散布図を保存
        outlier = df.iloc[np.where(pred < 0)]
        save_scatter(outlier, title="outlier", filename="scatter_03.png")

        # dfを外れ値データを除いたデータに置き換える
        df = df.iloc[np.where(pred > 0)]
        save_scatter(df, title="drop outlier", filename="scatter_04.png")

        # 重複した座標のデータを削除する
        df = process_duplicated(df)


        print("final data")
        print(df.describe().to_markdown())

        # CSVファイルに保存、ファイル名はdata.csvで固定
        # 外部からみたこのファイルのURLはこれ
        # https://takamitsu-iida.github.io/aburatsubo-terrain-data/data/data.csv
        df.to_csv(os.path.join(data_dir, "data.csv"), index=False)

        return 0

    # 実行
    sys.exit(main())
