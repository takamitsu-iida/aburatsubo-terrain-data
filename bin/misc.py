
#
# 標準ライブラリのインポート
#
import io
import logging
import math
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
img_dir = app_home.joinpath("img")

#
# 外部ライブラリのインポート
#
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from scipy.interpolate import RBFInterpolator

    from sklearn.neighbors import LocalOutlierFactor  # process_outlier()
    from sklearn.ensemble import IsolationForest  # process_outlier()

except ImportError as e:
    logging.error(e)
    sys.exit(1)

#
# ここからスクリプト
#

# Earth radius in km
EARTH_RADIUS = 6378.137

# 画像サイズ（単位：インチ）
FIG_SIZE = (9, 6)


class Stats:
    def __init__(self, df: pd.DataFrame) -> None:
        self.mean_lat = df["lat"].mean()
        self.min_lat = df["lat"].min()
        self.max_lat = df["lat"].max()
        self.min_lon = df["lon"].min()
        self.max_lon = df["lon"].max()
        # self.min_depth = df["depth"].min()
        # self.max_depth = df["depth"].max()

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
        self.distance_south_north = 2 * math.pi * EARTH_RADIUS * \
            1000 * (self.max_lat - self.min_lat) / 360
        self.distance_south_north = math.floor(self.distance_south_north)

        # 東西方向の距離(m)
        self.distance_west_east = h_circle * \
            (self.max_lon - self.min_lon) / 360
        self.distance_west_east = math.floor(self.distance_west_east)

    def __str__(self) -> str:
        with io.StringIO() as s:
            print("lat unit: {}".format(self.lat_unit), file=s)
            print("lon unit: {}".format(self.lon_unit), file=s)
            print(
                "south-north distance (m): {}".format(self.distance_south_north), file=s)
            print("west-east distance (m): {}".format(self.distance_west_east), file=s)
            print("", file=s)
            return s.getvalue()


def degree2radian(degree: float) -> float:
    return degree * math.pi / 180


# normalize x into rounded value
def round_unit(x: float, unit: float) -> float:
    return round(x / unit) * unit








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

def save_scatter(df: pd.DataFrame, title="", filename="") -> None:
    if not filename:
        return
    df_plt = df.plot.scatter(
        x="lon", y="lat", title=title, grid=True, figsize=FIG_SIZE, s=0.5)
    df_plt.set_xlabel("lon")
    df_plt.set_ylabel("lat")
    plt.savefig(os.path.join(img_dir, filename))
    plt.clf()

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


#
# snap_to_one_meter_gridを使った方がよい
#
def align_coord(df: pd.DataFrame, stats: Stats) -> pd.DataFrame:
    # 1メートルの間隔で各座標を吸着させる

    lat_unit = stats.lat_unit
    lon_unit = stats.lon_unit

    def f_lat(x): return round(x / lat_unit) * lat_unit
    def f_lon(x): return round(x / lon_unit) * lon_unit

    df["lat"] = df["lat"].map(f_lat)
    df["lon"] = df["lon"].map(f_lon)

    return df


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



def main():

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

    # オリジナルデータの散布図を保存
    save_scatter(df, title="original data", filename="scatter_01.png")

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
