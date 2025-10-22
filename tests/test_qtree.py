# tests/test_qtree2.py
# テストコード for qtree2.py
# pytestを使用
# 実行方法: pytest tests/test_qtree.py


import sys
import os
import random

# qtree2.pyのパスを追加
BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin'))
if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)

from qtree import Quadtree, QuadtreeNode

def test_query():
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
    bounds = (0, 0, 1, 1)
    quadtree = Quadtree(bounds)
    for point in test_data:
        quadtree.insert(point)

    # テストパターン1
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
    assert set(tuple(p.items()) for p in found_points1) == set(tuple(p.items()) for p in expected_points1)

    # テストパターン2
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
    assert set(tuple(p.items()) for p in found_points2) == set(tuple(p.items()) for p in expected_points2)

    # テストパターン3
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
    assert set(tuple(p.items()) for p in found_points3) == set(tuple(p.items()) for p in expected_points3)

    # テストパターン4
    range4 = (0.1, 0.1, 0.2, 0.2)
    found_points4 = quadtree.query(range4)
    expected_points4 = [
        {'lon': 0.1, 'lat': 0.1},
        {'lon': 0.15, 'lat': 0.15},
    ]
    assert set(tuple(p.items()) for p in found_points4) == set(tuple(p.items()) for p in expected_points4)



def test_get_nodes_at_depth():
    def generate_test_data(num_points):
        return [{'lon': random.uniform(0, 1), 'lat': random.uniform(0, 1)} for _ in range(num_points)]

    test_data = generate_test_data(1000)
    bounds = (0, 0, 1, 1)
    quadtree = Quadtree(bounds)
    for point in test_data:
        quadtree.insert(point)

    nodes_at_depth_1 = quadtree.get_nodes_at_depth(1)
    assert len(nodes_at_depth_1) == 4

    nodes_at_depth_2 = quadtree.get_nodes_at_depth(2)
    assert len(nodes_at_depth_2) == 16