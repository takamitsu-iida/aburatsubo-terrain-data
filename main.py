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
        self.min_lat = df["lat"].min()
        self.max_lat = df["lat"].max()
        self.min_lon = df["lon"].min()
        self.max_lon = df["lon"].max()
        self.max_depth = df["depth"].max()

        # 南北方向の距離(m)
        self.distance_south_north = 2 * math.pi * EARTH_RADIUS * 1000 * (self.max_lat - self.min_lat) / 360
        self.distance_south_north = math.floor(self.distance_south_north)

        # 東西方向の距離(m)
        radius = EARTH_RADIUS * 1000 * math.cos(degree2radian(self.min_lat))
        self.distance_west_east = 2 * math.pi * radius * (self.max_lon - self.min_lon) / 360
        self.distance_west_east = math.floor(self.distance_west_east)

        # グリッド単位
        self.grid_lon = (self.max_lon - self.min_lon) / (self.distance_west_east / GRID_UNIT)
        self.grid_lat = (self.max_lat - self.min_lat) / (self.distance_south_north / GRID_UNIT)


def degree2radian(degree: float) -> float:
    return degree * math.pi / 180


# normalize x into rounded value
def round_unit(x: float, unit: float) -> float:
    return round(x / unit) * unit


if __name__ == '__main__':

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


    def process_duplicated(df: pd.DataFrame):
        dfx = df[["lat", "lon", "depth"]]

        dfx_uniq = df.drop_duplicates(subset=["lat", "lon"], keep=False)
        print("unique coordinate")
        print(dfx_uniq.describe().to_markdown())
        print("")

        dfx_duplicated = dfx.groupby(["lat", "lon"])["depth"].mean().reset_index()
        print("uplicated coordinate")
        print(dfx_duplicated.describe().to_markdown())
        print("")

        df = pd.concat([dfx_uniq, dfx_duplicated]).reset_index(drop=True)
        print("combilned data frame")
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

    def norm_coord(df: pd.DataFrame):
        # 全行をforで取り出すので処理が重い
        for _, row in df.iterrows():
            row["lat"] = row["lat"] - 35.0
            row["lon"] = row["lon"] - 139.0

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

        # 時刻の列はいらない
        df = df[["lat", "lon", "depth"]]

        # データのサマリを表示
        print_summary(df)

        # オリジナルデータの散布図を保存
        df_plt = df.plot.scatter(x="lon", y="lat", title="original data", grid=True, figsize=FIG_SIZE)
        df_plt.set_xlabel("lon")
        df_plt.set_ylabel("lat")
        plt.savefig(os.path.join(img_dir, "scatter_01.png"))
        plt.clf()

        # 重複した座標のデータを削除する
        df = process_duplicated(df)

        # 重複を削除した散布図を保存
        df_plt = df.plot.scatter(x="lon", y="lat", title="drop duplicated", grid=True, figsize=FIG_SIZE)
        df_plt.set_xlabel("lon")
        df_plt.set_ylabel("lat")
        plt.savefig(os.path.join(img_dir, "scatter_02.png"))
        plt.clf()

        # 外れ値を除く処理を施す
        pred = process_outlier(df)

        # 外れ値データの散布図を保存
        outlier = df.iloc[np.where(pred < 0)]
        print("外れ値")
        print_summary(outlier)

        df_plt = outlier.plot.scatter(x="lon", y="lat", title="outlier", grid=True, figsize=FIG_SIZE)
        df_plt.set_xlabel("lon")
        df_plt.set_ylabel("lat")
        plt.savefig(os.path.join(img_dir, "scatter_03.png"))
        plt.clf()

        # dfを外れ値データを除いたデータに置き換える
        df = df.iloc[np.where(pred > 0)]
        print("外れ値を除外")
        print_summary(df)

        # 外れ値データを除いた散布図を保存
        df_plt = df.plot.scatter(x="lon", y="lat", title="drop outlier", grid=True, figsize=FIG_SIZE)
        df_plt.set_xlabel("lon")
        df_plt.set_ylabel("lat")
        plt.savefig(os.path.join(img_dir, "scatter_04.png"))
        plt.clf()

        # 統計量から東西距離、南北距離を割り出す
        stats = Stats(df)
        print("south-north distance (m): {}".format(stats.distance_south_north))
        print("west-east distance (m): {}".format(stats.distance_west_east))
        print("")

        # 座標を正規化する
        #df = norm_coord(df)
        # print_summary(df)

        return 0

    # 実行
    sys.exit(main())
