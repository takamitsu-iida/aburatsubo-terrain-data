#!/usr/bin/env python

"""

OpenStreetMap (OSM) のタイルの座標は、
地図を構成する小さな正方形の画像タイルの位置を一意に特定するためのものです。
これらのタイルは、特定のズームレベルにおいて世界全体を網羅するようにグリッド状に並べられています。

OSMタイルの座標は、主に以下の3つの要素で構成されます。

Z ズームレベル
X X座標
Y Y座標


Z
    意味: 地図の拡大率を表します。ズームレベルが高くなるほど、地図は拡大され、より詳細な情報が表示されるようになります。
    値の範囲:
        通常は0から19や20（またはそれ以上）の整数値で表されます。
        Z=0 は、世界全体が1枚のタイルで表現される、最も広範囲なズームレベルです。
        Z=1 になると、世界は縦横2x2の4枚のタイルで表現されます。
        Z=n になると、世界は縦横 2^n x 2^n 枚のタイルで構成されます。
X
    意味: 西から東方向へのタイルの位置 を表します。
    値の範囲:
        各ズームレベルにおいて、X座標は 0 から (2^Z - 1) までの整数値を取ります。
        例えば、Z=0 ではXは0のみ。
        Z=1 ではXは 0 または 1。
        Z=14 の場合、Xは 0 から (2^{14}-1) = 16383 までの整数になります。
    基準:
        グリニッジ子午線（経度0度）がX座標の基準となります。
        最も西端（経度-180度）がX座標 0 に近くなり、最も東端（経度+180度）が最大値に近くなります。

Y
    意味: 北から南方向へのタイルの位置 を表します。
    値の範囲:
        各ズームレベルにおいて、Y座標は 0 から (2^Z - 1) までの整数値を取ります。
        例えば、Z=0 ではYは0のみ。
        Z=1 ではYは 0 または 1。
        Z=14 の場合、Yは 0 から (2^{14}-1) = 16383 までの整数になります。
    基準:
      メルカトル図法の特性上、赤道がY座標の基準となります。最も北端（北緯85.0511度付近）がY座標 0 に近くなり、最も南端（南緯85.0511度付近）が最大値に近くなります。
        メルカトル図法の特性上、赤道がY座標の基準となります。
        最も北端（北緯85.0511度付近）がY座標 0 に近くなり、最も南端（南緯85.0511度付近）が最大値に近くなります。

"""

#
# 標準ライブラリのインポート
#
import logging
import math
import sys

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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

def latlon_to_tile(lat_deg, lon_deg, zoom):
    """
    経度緯度とズームレベルからタイルのXYZ座標を計算します。
    OpenStreetMapのSlippy map tilenamesの計算式に基づいています。
    """
    lat_rad = math.radians(lat_deg)
    n = 2 ** zoom
    x_tile = int(n * ((lon_deg + 180) / 360))
    y_tile = int(n * (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2)
    return x_tile, y_tile


def tile_to_bounds(x_tile, y_tile, zoom):
    """
    タイルのXYZ座標からそのタイルの南西角と北東角の経度緯度を計算します。
    """
    def n_to_lon(x, n):
        return x / n * 360.0 - 180.0

    def n_to_lat(y, n):
        a = math.pi * (1 - 2 * y / n)
        return math.degrees(math.atan(math.sinh(a)))

    n = 2 ** zoom
    lon_west = n_to_lon(x_tile, n)
    lat_north = n_to_lat(y_tile, n)
    lon_east = n_to_lon(x_tile + 1, n)
    lat_south = n_to_lat(y_tile + 1, n)

    return (lat_south, lon_west, lat_north, lon_east) # (min_lat, min_lon, max_lat, max_lon)



if __name__ == '__main__':

    def main():

        # 東京タワーの緯度経度
        tokyo_tower_lat = 35.658581
        tokyo_tower_lon = 139.745433

        # レベル14
        zoom_level = 14

        # 例: 東京タワーのタイル座標を計算
        x, y = latlon_to_tile(tokyo_tower_lat, tokyo_tower_lon, zoom_level)
        print(f"東京タワー ({tokyo_tower_lat}, {tokyo_tower_lon}) のズームレベル {zoom_level} でのタイル座標: x={x}, y={y}")

        # 例: 上記で計算したタイル座標の範囲を計算
        tile_bounds = tile_to_bounds(x, y, zoom_level)
        print(f"タイル ({x}, {y}) のズームレベル {zoom_level} での経度緯度範囲: {tile_bounds}")


    #
    # 実行
    #
    main()
