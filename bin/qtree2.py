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


class QuadtreeNode:
    MAX_POINTS = 4
    MAX_DEPTH = 10

    def __init__(self, bounds, depth, parent=None):
        self.bounds = bounds
        self.depth = depth
        self.points = []
        self.children = []
        self.parent = parent

    def is_leaf(self):
        return len(self.children) == 0

    def subdivide(self):
        lon1, lat1, lon2, lat2 = self.bounds
        mid_lon = (lon1 + lon2) / 2
        mid_lat = (lat1 + lat2) / 2

        self.children.append(QuadtreeNode(
            (lon1, lat1, mid_lon, mid_lat), self.depth + 1, self))
        self.children.append(QuadtreeNode(
            (mid_lon, lat1, lon2, mid_lat), self.depth + 1, self))
        self.children.append(QuadtreeNode(
            (lon1, mid_lat, mid_lon, lat2), self.depth + 1, self))
        self.children.append(QuadtreeNode(
            (mid_lon, mid_lat, lon2, lat2), self.depth + 1, self))

    def insert(self, point):
        if not self.contains(point):
            return False

        if self.is_leaf():
            if len(self.points) < QuadtreeNode.MAX_POINTS or self.depth >= QuadtreeNode.MAX_DEPTH:
                self.points.append(point)
                return True
            else:
                self.subdivide()

        for child in self.children:
            if child.insert(point):
                return True

        return False

    def contains(self, point):
        lon1, lat1, lon2, lat2 = self.bounds
        return lon1 <= point['lon'] < lon2 and lat1 <= point['lat'] < lat2

    def intersects(self, range):
        lon1, lat1, lon2, lat2 = self.bounds
        r_lon1, r_lat1, r_lon2, r_lat2 = range
        return not (r_lon1 >= lon2 or r_lon2 <= lon1 or r_lat1 >= lat2 or r_lat2 <= lat1)

    def query(self, range):
        found_points = []
        if not self.intersects(range):
            return found_points

        for point in self.points:
            if self.contains_point_in_range(point, range):
                found_points.append(point)

        if not self.is_leaf():
            for child in self.children:
                found_points.extend(child.query(range))

        return found_points

    def contains_point_in_range(self, point, range):
        r_lon1, r_lat1, r_lon2, r_lat2 = range
        return r_lon1 <= point['lon'] < r_lon2 and r_lat1 <= point['lat'] < r_lat2

    def get_nodes_at_depth(self, depth):
        nodes = []
        if self.depth == depth:
            nodes.append(self)
        elif self.depth < depth:
            for child in self.children:
                nodes.extend(child.get_nodes_at_depth(depth))
        return nodes

    def get_leaf_nodes(self):
        if self.is_leaf():
            return [self]
        else:
            leaf_nodes = []
            for child in self.children:
                leaf_nodes.extend(child.get_leaf_nodes())
            return leaf_nodes


class Quadtree:

    def __init__(self, bounds):
        self.root = QuadtreeNode(bounds, 0)

    def insert(self, point):
        self.root.insert(point)

    def query(self, range):
        return self.root.query(range)

    def get_nodes_at_depth(self, depth):
        return self.root.get_nodes_at_depth(depth)

    def get_leaf_nodes(self):
        return self.root.get_leaf_nodes()


if __name__ == '__main__':

    import random

    test_data = [
        {'lon': 0.567603099626, 'lat': 0.410160220857},
        {'lon': 0.373657009068, 'lat': 0.549501477427},
        {'lon': 0.500192599714, 'lat': 0.352420542886},
        {'lon': 0.626796922, 'lat': 0.422685113179},
        {'lon': 0.527521290061, 'lat': 0.483502904656},
        {'lon': 0.1, 'lat': 0.1},
        {'lon': 0.2, 'lat': 0.2},
        {'lon': 0.3, 'lat': 0.3},
        {'lon': 0.4, 'lat': 0.4},
        {'lon': 0.5, 'lat': 0.5},
        {'lon': 0.6, 'lat': 0.6},
        {'lon': 0.7, 'lat': 0.7},
        {'lon': 0.8, 'lat': 0.8},
        {'lon': 0.9, 'lat': 0.9},
        {'lon': 0.15, 'lat': 0.15},
        {'lon': 0.25, 'lat': 0.25},
        {'lon': 0.35, 'lat': 0.35},
        {'lon': 0.45, 'lat': 0.45},
        {'lon': 0.55, 'lat': 0.55},
        {'lon': 0.65, 'lat': 0.65},
        {'lon': 0.75, 'lat': 0.75},
        {'lon': 0.85, 'lat': 0.85},
        {'lon': 0.95, 'lat': 0.95},
    ]

    def test_query():
        bounds = (0, 0, 1, 1)
        quadtree = Quadtree(bounds)

        for point in test_data:
            quadtree.insert(point)

        bounds = (0, 0, 1, 1)
        quadtree = Quadtree(bounds)

        for point in test_data:
            quadtree.insert(point)

        # テストパターン1: 範囲 (0.5, 0.5, 1, 1)
        range1 = (0.5, 0.5, 1, 1)
        found_points1 = quadtree.query(range1)
        expected_points1 = [
            {'lon': 0.5, 'lat': 0.5},
            {'lon': 0.6, 'lat': 0.6},
            {'lon': 0.7, 'lat': 0.7},
            {'lon': 0.8, 'lat': 0.8},
            {'lon': 0.9, 'lat': 0.9},
            {'lon': 0.55, 'lat': 0.55},
            {'lon': 0.65, 'lat': 0.65},
            {'lon': 0.75, 'lat': 0.75},
            {'lon': 0.85, 'lat': 0.85},
            {'lon': 0.95, 'lat': 0.95},
        ]
        assert set(tuple(p.items()) for p in found_points1) == set(
            tuple(p.items()) for p in expected_points1), f"Test 1 failed: {found_points1}"

        # テストパターン2: 範囲 (0, 0, 0.5, 0.5)
        range2 = (0, 0, 0.5, 0.5)
        found_points2 = quadtree.query(range2)
        expected_points2 = [
            {'lon': 0.1, 'lat': 0.1},
            {'lon': 0.2, 'lat': 0.2},
            {'lon': 0.3, 'lat': 0.3},
            {'lon': 0.4, 'lat': 0.4},
            {'lon': 0.15, 'lat': 0.15},
            {'lon': 0.25, 'lat': 0.25},
            {'lon': 0.35, 'lat': 0.35},
            {'lon': 0.45, 'lat': 0.45},
        ]
        assert set(tuple(p.items()) for p in found_points2) == set(
            tuple(p.items()) for p in expected_points2), f"Test 2 failed: {found_points2}"

        # テストパターン3: 範囲 (0.25, 0.25, 0.75, 0.75)
        range3 = (0.25, 0.25, 0.75, 0.75)
        found_points3 = quadtree.query(range3)
        expected_points3 = [
            {'lon': 0.25, 'lat': 0.25},
            {'lon': 0.567603099626, 'lat': 0.410160220857},
            {'lon': 0.373657009068, 'lat': 0.549501477427},
            {'lon': 0.500192599714, 'lat': 0.352420542886},
            {'lon': 0.626796922, 'lat': 0.422685113179},
            {'lon': 0.527521290061, 'lat': 0.483502904656},
            {'lon': 0.3, 'lat': 0.3},
            {'lon': 0.4, 'lat': 0.4},
            {'lon': 0.5, 'lat': 0.5},
            {'lon': 0.6, 'lat': 0.6},
            {'lon': 0.35, 'lat': 0.35},
            {'lon': 0.45, 'lat': 0.45},
            {'lon': 0.55, 'lat': 0.55},
            {'lon': 0.65, 'lat': 0.65},
            {'lon': 0.7, 'lat': 0.7},
        ]
        assert set(tuple(p.items()) for p in found_points3) == set(
            tuple(p.items()) for p in expected_points3), f"Test 3 failed: {found_points3}"

        # テストパターン4: 範囲 (0.1, 0.1, 0.2, 0.2)
        range4 = (0.1, 0.1, 0.2, 0.2)
        found_points4 = quadtree.query(range4)
        expected_points4 = [
            {'lon': 0.1, 'lat': 0.1},
            {'lon': 0.15, 'lat': 0.15},
        ]
        assert set(tuple(p.items()) for p in found_points4) == set(
            tuple(p.items()) for p in expected_points4), f"Test 4 failed: {found_points4}"

        print("All query tests passed!")

    def generate_test_data(num_points):
        return [{'lon': random.uniform(0, 1), 'lat': random.uniform(0, 1)} for _ in range(num_points)]

    def test_get_nodes_at_depth():
        test_data = generate_test_data(1000)  # 多くのデータを生成
        bounds = (0, 0, 1, 1)
        quadtree = Quadtree(bounds)

        for point in test_data:
            quadtree.insert(point)

        # get_nodes_at_depthのテスト
        nodes_at_depth_1 = quadtree.get_nodes_at_depth(1)
        assert len(
            nodes_at_depth_1) == 4, f"Test get_nodes_at_depth failed: {len(nodes_at_depth_1)} nodes found at depth 1"

        nodes_at_depth_2 = quadtree.get_nodes_at_depth(2)
        assert len(
            nodes_at_depth_2) == 16, f"Test get_nodes_at_depth failed: {len(nodes_at_depth_2)} nodes found at depth 2"

        print("All get_nodes_at_depth tests passed!")

    def main():
        test_query()
        test_get_nodes_at_depth()
        return 0

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

    def _main():
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

        return 0

    # 実行
    sys.exit(main())
