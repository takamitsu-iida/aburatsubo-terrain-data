#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""rtreeを動かす練習スクリプト
"""

__author__ = "takamitsu-iida"
__version__ = "0.1"
__date__ = "2024/10/10"

#
# 標準ライブラリのインポート
#
import logging
import os
import sys


def here(path=''):
    """相対パスを絶対パスに変換して返却します"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


# アプリケーションのホームディレクトリは一つ上
app_home = here("..")

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
# ここからスクリプト
#

class Area:
    def __init__(self, x1, y1, x2, y2, level):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.level = level
        self.__points = []
        self.fixed = False

    def append(self, p):
        """ 領域にデータ点を追加 """
        self.__points.append(p)

    def points(self):
        """ 領域に属しているデータ点を返す """
        return self.__points

    def is_fixed(self):
        """ 分割が終わっているかどうか """
        return self.fixed

    def set_fixed(self):
        """ 分割が終わったフラグを立てる """
        self.fixed = True

    def cover(self, p):
        """ あるデータ点pがこの領域にカバーされるかどうか """
        if self.x1 < p[0] and self.y1 < p[1] and self.x2 >= p[0] and self.y2 >= p[1]:
            return True
        else:
            return False


def divide(area, level):
    division = []

    """ 分割後の各辺の長さ """
    xl = (area.x2 - area.x1)/2
    yl = (area.y2 - area.y1)/2

    """ 分割後の領域を生成 """
    for dx in [0, 1]:
        for dy in [0, 1]:
            sub_area = Area(area.x1+dx*xl, area.y1+dy*yl, area.x1+(1+dx)*xl, area.y1+(1+dy)*yl, level)
            division.append(sub_area)

    """ 分割前の領域に属すデータ点を分割後の領域にアサイン """
    for p in area.points():
        for sub_area in division:
            if sub_area.cover(p):
                sub_area.append(p)
                break

    return division


def quadtree(initial, maxpoints, maxdivision):
    areas = [initial]

    """ 引数で与えられたmaxdivision回だけ分割を繰り返す """
    for n in range(maxdivision):
        new_areas = []
        for i in range(len(areas)):
            if not areas[i].is_fixed():
                """ まだ分割が終わっていない領域に対して """
                if len(areas[i].points()) > maxpoints:
                    """ 領域に属すデータ点の数がmaxpoints個を超えていたら分割 """
                    division = divide(areas[i], n+1)
                    for d in division:
                        new_areas.append(d)
                else:
                    """ 領域に属すデータ点の数がmaxpoints個を超えていなかったらもう分割しない """
                    areas[i].set_fixed()
                    new_areas.append(areas[i])
            else:
                """ 分割が終わっていればそのまま """
                new_areas.append(areas[i])
        areas = new_areas

    return areas


if __name__ == '__main__':

    test_data = [
        [0.567603099626, 0.410160220857],
        [0.405568042222, 0.555583016695],
        [0.450289054963, 0.252870772505],
        [0.373657009068, 0.549501477427],
        [0.500192599714, 0.352420542886],
        [0.626796922, 0.422685113179],
        [0.527521290061, 0.483502904656],
        [0.407737520746, 0.570049935936],
        [0.578095278433, 0.6959689898],
        [0.271957975882, 0.450460115198],
        [0.56451369371, 0.491139311353],
        [0.532304354436, 0.486931064003],
        [0.553942716039, 0.51953331907],
        [0.627341495722, 0.396617894317],
        [0.454210189397, 0.570214499065],
        [0.327359895038, 0.582972137899],
        [0.422271372537, 0.560892624101],
        [0.443036148978, 0.434529240506],
        [0.644625936719, 0.542486338813],
        [0.447813648487, 0.575896033203],
        [0.534217713171, 0.636123087401],
        [0.348346109137, 0.312959224746],
        [0.484075186327, 0.289521849258],
        [0.588417643962, 0.387831556678],
        [0.613422176662, 0.665770829308],
        [0.60994411786, 0.399778040078],
        [0.425443751505, 0.402619561402],
        [0.504955932504, 0.610015349003],
        [0.402852203978, 0.382379275531],
        [0.387591801531, 0.452180343665]
    ]

    def read_csv(filepath):
        """ CSVファイルを読み込んでリストを返す """
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data.append([float(x) for x in line.strip().split(',')])
                except ValueError:
                    pass
        return data


    def dump_area(area):
        for a in area:
            print("%s %s %s %s" % (a.x1, a.y1, a.x2, a.y2), end=' ')
            for p in a.points():
                print(p, end=' ')
            print()


    def main():
        data_filename = "data.csv"
        data_path = os.path.join(data_dir, data_filename)
        if not os.path.exists(data_path):
            logger.error("File not found: %s" % data_path)
            return 1

        data = read_csv(data_path)

        # |       |             lat |             lon |        depth |
        # |:------|----------------:|----------------:|-------------:|
        # | count | 107440          | 107440          | 107440       |
        # | mean  |     35.1641     |    139.608      |     16.4776  |
        # | std   |      0.00237647 |      0.00435879 |      9.62367 |
        # | min   |     35.1572     |    139.599      |      1.082   |
        # | 25%   |     35.1628     |    139.604      |      8.409   |
        # | 50%   |     35.164      |    139.608      |     15.09    |
        # | 75%   |     35.1649     |    139.61       |     21.928   |
        # | max   |     35.1737     |    139.622      |     47.539   |

        x_min = 35.1572
        y_min = 139.599
        x_max = 35.1737
        y_max = 139.622
        maxpoints = 5
        maxdivision = 20

        """ 対象とする領域を生成 """
        initial = Area(x_min, y_min, x_max, y_max, 0)
        for d in data:
            initial.append(d)

        """ 分割 """
        area = quadtree(initial, maxpoints, maxdivision)

        """ 結果 """
        dump_area(area)

        return 0

    # 実行
    sys.exit(main())
