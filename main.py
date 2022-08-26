#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DeeperのGPSデータを加工するPythonスクリプト
"""

__author__ = "takamitsu-iida"
__version__ = "0.1"
__date__ = "2022/08/19"

#
# 標準ライブラリのインポート
#
import io
import logging
import math
import os
import sys


def here(path=''):
    """相対パスを絶対パスに変換して返却します"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


# アプリケーションのホームディレクトリは一つ上
app_home = here(".")

# 自身の名前から拡張子を除いてプログラム名を得る
app_name = os.path.splitext(os.path.basename(__file__))[0]

# ディレクトリ
data_dir = os.path.join(app_home, "data")
img_dir = os.path.join(app_home, "img")

# libフォルダにおいたpythonスクリプトをインポートできるようにするための処理
# このファイルの位置から一つ
if not here("./lib") in sys.path:
    sys.path.append(here("./lib"))

#
# ログ設定
#

# ログファイルの名前
log_file = app_name + ".log"

# ログファイルを置くディレクトリ
log_dir = os.path.join(app_home, "log")
try:
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
except OSError:
    pass

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

# default setting
logging.basicConfig()

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
# 外部ライブラリのインポート
#
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import RBFInterpolator
    from sklearn.neighbors import LocalOutlierFactor  # process_outlier()
    from sklearn.ensemble import IsolationForest  # process_outlier()
except ImportError as e:
    logger.exception(e)
    sys.exit(1)

#
# ここからスクリプト
#

# Earth radius in km
EARTH_RADIUS = 6378.137

# grid unit in m
GRID_UNIT = 1.0

# 画像サイズ（単位：インチ）
FIG_SIZE = (9, 6)


class Stats:
    def __init__(self, df: pd.DataFrame) -> None:
        self.mean_lat = df["lat"].mean()
        self.min_lat = df["lat"].min()
        self.max_lat = df["lat"].max()
        self.min_lon = df["lon"].min()
        self.max_lon = df["lon"].max()
        #self.min_depth = df["depth"].min()
        #self.max_depth = df["depth"].max()

        # この場所で地球を水平(holizontal)に輪切りにしたときの半径(m)と円の長さ
        h_r = EARTH_RADIUS * 1000 * math.cos(degree2radian(self.mean_lat))
        h_circle = 2 * math.pi * h_r

        # この円において弧の長さが1mになる角度、つまり経度の差
        self.lon_unit = 360 / h_circle

        # 地球を北極から南極に縦に輪切りにしたときの円周(m)
        v_circle = 2 * math.pi * EARTH_RADIUS * 1000

        # この円において弧の長さが1mになる角度、つまり経度の差
        self.lat_unit = 360 / v_circle

        # 南北方向の距離(m)
        self.distance_south_north = 2 * math.pi * EARTH_RADIUS * 1000 * (self.max_lat - self.min_lat) / 360
        self.distance_south_north = math.floor(self.distance_south_north)

        # 東西方向の距離(m)
        self.distance_west_east = h_circle * (self.max_lon - self.min_lon) / 360
        self.distance_west_east = math.floor(self.distance_west_east)

    def __str__(self) -> str:
        with io.StringIO() as s:
            print("lat unit: {}".format(self.lat_unit), file=s)
            print("lon unit: {}".format(self.lon_unit), file=s)
            print("south-north distance (m): {}".format(self.distance_south_north), file=s)
            print("west-east distance (m): {}".format(self.distance_west_east), file=s)
            print("", file=s)
            return s.getvalue()


def degree2radian(degree: float) -> float:
    return degree * math.pi / 180


# normalize x into rounded value
def round_unit(x: float, unit: float) -> float:
    return round(x / unit) * unit


if __name__ == '__main__':

    def rbf_example():
        from scipy.stats.qmc import Halton
        rng = np.random.default_rng()
        xobs = 2*Halton(2, seed=rng).random(100) - 1
        yobs = np.sum(xobs, axis=1)*np.exp(-6*np.sum(xobs**2, axis=1))
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


    def generate_test_grid(df:pd.DataFrame):
        # テスト用座標
        TEST_COORD_NW = (35.1636, 139.6082)
        TEST_COORD_SE = (35.1622, 139.6099)

        extracted = df.query("lat > {} and lat < {} and lon > {} and lon < {}".format(TEST_COORD_SE[0], TEST_COORD_NW[0], TEST_COORD_NW[1], TEST_COORD_SE[1]))
        lat = extracted["lat"]
        lat = (lat - lat.mean())/lat.std()
        lon = extracted["lon"]
        lon = (lon - lon.mean())/lon.std()
        depth = extracted["depth"]
        lat_lon = np.stack([lat, lon], -1)
        rbf = RBFInterpolator(lat_lon, depth, kernel='thin_plate_spline', epsilon=2.0, neighbors=10)


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

        #==============================================================================
        # 3 次元グラフ
        #==============================================================================
        fig = plt.figure(figsize=(18, 7), dpi=200) # 画像を作成する
        el = 55                                    # 視点高さを設定する

        ax = fig.add_subplot(231, projection='3d') # 2 × 3 の 1 枚目に描画する

        # 曲面のプロット
        ax.plot_surface(xi, yi, zi)                      # サーフェスの描画
        ax.view_init(elev=el, azim=10)                   # ビューの設定
        ax.set_title('elev = ' + str(el) + ', deg = 10') # タイトルの設定
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
        print(df.describe().to_markdown())  # to_markdown() requires tabulate module
        print("")


    def save_scatter(df: pd.DataFrame, title="", filename="") -> None:
        if not filename:
            return
        df_plt = df.plot.scatter(x="lon", y="lat", title=title, grid=True, figsize=FIG_SIZE, s=0.5)
        df_plt.set_xlabel("lon")
        df_plt.set_ylabel("lat")
        plt.savefig(os.path.join(img_dir, filename))
        plt.clf()


    def process_duplicated(df: pd.DataFrame):
        dfx = df[["lat", "lon", "depth"]]

        dfx_uniq = df.drop_duplicates(subset=["lat", "lon"], keep=False)
        print("unique coordinate")
        print(dfx_uniq.describe().to_markdown())
        print("")

        dfx_duplicated = dfx.groupby(["lat", "lon"])["depth"].mean().reset_index()
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

    def align_coord(df: pd.DataFrame) -> pd.DataFrame:
        # 1メートルの間隔で各座標を吸着させる
        stats = Stats(df)
        lat_unit = stats.lat_unit
        lon_unit = stats.lon_unit

        f_lat = lambda x: round(x / lat_unit) * lat_unit
        f_lon = lambda x: round(x / lon_unit) * lon_unit

        df["lat"] = df["lat"].map(f_lat)
        df["lon"] = df["lon"].map(f_lon)

        return df


    def main():
        """メイン関数
        Returns:
        int -- 正常終了は0、異常時はそれ以外を返却
        """

        # オリジナルデータをファイルから読む
        file_path = os.path.join(data_dir, "bathymetry_data.csv")
        df = load_csv(file_path)

        # CSVファイルには列名がないので、データフレームに列名を定義
        df.columns = ["lat", "lon", "depth", "time"]

        # 時刻の列は不要なので削除する
        del df["time"]

        # データのサマリを表示
        print("original data")
        print_summary(df)

        # オリジナルデータの散布図を保存
        save_scatter(df, title="original data", filename="scatter_01.png")

        # 座標を1m単位に吸着させる
        df = align_coord(df)
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

        return 0

    # 実行
    sys.exit(main())
