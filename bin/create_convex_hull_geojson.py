#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSVデータの凸包をGeoJSON形式で保存します。



# デフォルトのalpha値で実行
create_convex_hull_geojson.py --input input.csv --output output.geojson

# alpha値を調整（小さいほど詳細な境界）
create_convex_hull_geojson.py --input input.csv --output output.geojson --alpha 0.005

# alpha値を大きくすると凸包に近づく
create_convex_hull_geojson.py --input input.csv --output output.geojson --alpha 0.1

"""

SCRIPT_DESCRIPTION: str = 'Create GeoJSON convex hull from CSV data'

#
# 標準ライブラリのインポート
#
import argparse
import logging
import json
import sys

from pathlib import Path
from typing import List, Tuple, Dict, Any  #, Callable, Optional

#
# ローカルライブラリのインポート
#
from load_save_csv import read_csv_points

#
# 外部ライブラリのインポート
#

# WSL1 固有の numpy 警告を抑制
# https://github.com/numpy/numpy/issues/18900
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, module=r"numpy.*", message=r"Signature b")

try:
    import numpy as np
    import matplotlib.pyplot as plt

    from shapely.geometry import Point, MultiPoint, mapping, LineString, Polygon
    from shapely.ops import unary_union
    from scipy.interpolate import griddata
    from scipy.spatial import Delaunay

except ImportError as e:
    logging.error("必要なライブラリがインストールされていません。")
    sys.exit(1)

# このファイルへのPathオブジェクト
app_path: Path = Path(__file__)

# このファイルがあるディレクトリ
app_dir: Path = app_path.parent

# このファイルの名前から拡張子を除いてプログラム名を得る
app_name: str = app_path.stem

# アプリケーションのホームディレクトリはこのファイルからみて一つ上
app_home: Path = app_path.parent.joinpath('..').resolve()

# データディレクトリ
data_dir: Path = app_home.joinpath("data")

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

def create_convex_hull_geometry(points: List[Tuple[float, float, float]]) -> Tuple[Dict[str, Any], Tuple[float, float]]:
    """
    点群から凸包を作成し、GeoJSON形式で返す

    Args:
        points: (lat, lon, depth)のタプルのリスト

    Returns:
        GeoJSON形式の辞書と中心座標(lat, lon)のタプル
    """
    # (lon, lat)の順序でPointオブジェクトを作成（GeoJSONの座標順序に従う）
    shapely_points = [Point(lon, lat) for lat, lon, depth in points]

    # MultiPointオブジェクトを作成
    multi_point = MultiPoint(shapely_points)

    logger.info(f"凸包を計算中...")

    # 凸包を計算
    convex_hull = multi_point.convex_hull

    # 凸包の中心座標を計算
    centroid = convex_hull.centroid
    center_lon = centroid.x
    center_lat = centroid.y

    logger.info(f"凸包の中心座標: lat={center_lat:.6f}, lon={center_lon:.6f}")

    # GeoJSON形式に変換
    geojson_geometry = mapping(convex_hull)

    logger.info(f"凸包の頂点数: {len(geojson_geometry.get('coordinates', [[]])[0])}")

    return geojson_geometry, (center_lat, center_lon)



def create_contours_from_points(points: List[Tuple[float, float, float]],
                                 levels: List[float] = None,
                                 grid_resolution: int = 100) -> List[Dict[str, Any]]:
    """
    点群から等高線を生成し、GeoJSON Feature のリストを返す

    Args:
        points: (lat, lon, depth)のタプルのリスト
        levels: 等高線のレベル（深度）のリスト。Noneの場合は自動設定
        grid_resolution: グリッドの解像度

    Returns:
        GeoJSON Feature のリスト
    """
    # データを配列に変換
    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])
    depths = np.array([p[2] for p in points])

    # グリッドを作成
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    grid_lat = np.linspace(lat_min, lat_max, grid_resolution)
    grid_lon = np.linspace(lon_min, lon_max, grid_resolution)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    logger.info(f"グリッド補間中... (解像度: {grid_resolution}x{grid_resolution})")

    # グリッド補間
    grid_depth = griddata(
        (lons, lats),
        depths,
        (grid_lon_mesh, grid_lat_mesh),
        method='cubic'
    )

    # 等高線レベルの設定
    if levels is None:
        depth_min, depth_max = np.nanmin(depths), np.nanmax(depths)
        levels = np.linspace(depth_min, depth_max, 10)

    logger.info(f"等高線を生成中... (レベル数: {len(levels)})")

    # 等高線を生成
    contour_features = []

    fig, ax = plt.subplots()
    cs = ax.contour(grid_lon_mesh, grid_lat_mesh, grid_depth, levels=levels)

    # collectionsの取得方法を変更
    try:
        collections = cs.collections
    except AttributeError:
        # 新しいmatplotlibではallsegsを使用
        collections = [cs.allsegs[i] if i < len(cs.allsegs) else [] for i in range(len(levels))]
        use_allsegs = True
    else:
        use_allsegs = False

    # 各等高線をGeoJSON Featureに変換
    for level_idx, level in enumerate(levels):
        if use_allsegs:
            # allsegsを使用する場合
            segments = cs.allsegs[level_idx] if level_idx < len(cs.allsegs) else []
            for segment in segments:
                if len(segment) < 2:
                    continue

                # GeoJSON座標は [lon, lat] の順序
                coordinates = [[float(v[0]), float(v[1])] for v in segment]

                # LineStringとして追加
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "depth": float(level),
                        "type": "contour"
                    }
                }
                contour_features.append(feature)
        else:
            # collectionsを使用する場合
            collection = collections[level_idx]
            for path in collection.get_paths():
                vertices = path.vertices
                if len(vertices) < 2:
                    continue

                # GeoJSON座標は [lon, lat] の順序
                coordinates = [[float(v[0]), float(v[1])] for v in vertices]

                # LineStringとして追加
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "depth": float(level),
                        "type": "contour"
                    }
                }
                contour_features.append(feature)

    plt.close(fig)

    logger.info(f"等高線を {len(contour_features)} 本生成しました")

    return contour_features




def create_boundary_geometry(points: List[Tuple[float, float, float]], alpha: float = 0.01) -> Tuple[Dict[str, Any], Tuple[float, float]]:
    """
    点群から境界（concave hull）を作成し、GeoJSON形式で返す

    Args:
        points: (lat, lon, depth)のタプルのリスト
        alpha: 境界の詳細度パラメータ（小さいほど詳細な境界、大きいほど凸包に近い）

    Returns:
        GeoJSON形式の辞書と中心座標(lat, lon)のタプル
    """
    # (lon, lat)の配列を作成（GeoJSONの座標順序に従う）
    coords = np.array([(lon, lat) for lat, lon, depth in points])

    logger.info(f"境界を計算中... (alpha={alpha})")

    # ドロネー三角分割を実行
    tri = Delaunay(coords)

    # 境界エッジを抽出（外周の辺のみ）
    edges = {}  # edge -> triangle count

    for simplex in tri.simplices:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            edge = tuple(sorted([simplex[i], simplex[j]]))
            edges[edge] = edges.get(edge, 0) + 1

    # 外周のエッジ（1つの三角形にしか属さない辺）を抽出
    boundary_edges = [edge for edge, count in edges.items() if count == 1]

    logger.info(f"境界エッジ数: {len(boundary_edges)}")

    if not boundary_edges:
        logger.warning("境界エッジの抽出に失敗しました。凸包を使用します。")
        shapely_points = [Point(lon, lat) for lat, lon, depth in points]
        multi_point = MultiPoint(shapely_points)
        boundary = multi_point.convex_hull
    else:
        # エッジから連続したパスを構築
        from collections import defaultdict
        graph = defaultdict(list)

        for i, j in boundary_edges:
            graph[i].append(j)
            graph[j].append(i)

        # 外周パスを構築
        if not graph:
            logger.warning("グラフの構築に失敗しました。凸包を使用します。")
            shapely_points = [Point(lon, lat) for lat, lon, depth in points]
            multi_point = MultiPoint(shapely_points)
            boundary = multi_point.convex_hull
        else:
            # 開始点を選択
            start = next(iter(graph.keys()))
            path = [start]
            current = start
            visited = {start}

            # パスを辿る
            while len(path) < len(boundary_edges) + 1:
                neighbors = graph[current]

                # 未訪問の隣接点を探す
                next_nodes = [n for n in neighbors if n not in visited]

                if not next_nodes:
                    # 閉じたパスの場合
                    if len(path) > 2 and start in neighbors:
                        break
                    # 行き止まりの場合は最も近い点を探す
                    logger.warning("パスが途切れました。利用可能な点からパスを再構築します。")
                    break

                next_node = next_nodes[0]
                path.append(next_node)
                visited.add(next_node)
                current = next_node

            if len(path) >= 3:
                # 座標リストを作成
                boundary_coords = [coords[i].tolist() for i in path]

                # 閉じたポリゴンにする
                if boundary_coords[0] != boundary_coords[-1]:
                    boundary_coords.append(boundary_coords[0])

                try:
                    boundary = Polygon(boundary_coords)

                    # ポリゴンが有効か確認
                    if not boundary.is_valid:
                        logger.warning("生成されたポリゴンが無効です。凸包を使用します。")
                        shapely_points = [Point(lon, lat) for lat, lon, depth in points]
                        multi_point = MultiPoint(shapely_points)
                        boundary = multi_point.convex_hull
                except Exception as e:
                    logger.warning(f"ポリゴンの作成に失敗しました: {e}。凸包を使用します。")
                    shapely_points = [Point(lon, lat) for lat, lon, depth in points]
                    multi_point = MultiPoint(shapely_points)
                    boundary = multi_point.convex_hull
            else:
                logger.warning("十分な点が見つかりませんでした。凸包を使用します。")
                shapely_points = [Point(lon, lat) for lat, lon, depth in points]
                multi_point = MultiPoint(shapely_points)
                boundary = multi_point.convex_hull

    # 境界の中心座標を計算
    centroid = boundary.centroid
    center_lon = centroid.x
    center_lat = centroid.y

    logger.info(f"境界の中心座標: lat={center_lat:.6f}, lon={center_lon:.6f}")
    logger.info(f"境界の面積: {boundary.area:.6f}")

    # GeoJSON形式に変換
    geojson_geometry = mapping(boundary)

    if geojson_geometry['type'] == 'Polygon':
        logger.info(f"境界の頂点数: {len(geojson_geometry.get('coordinates', [[]])[0])}")
    else:
        logger.info(f"境界の形状タイプ: {geojson_geometry['type']}")

    return geojson_geometry, (center_lat, center_lon)

# ================================================================================
# メイン処理
# ================================================================================

if __name__ == '__main__':

    def main() -> None:

        # 引数処理
        parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
        parser.add_argument('--input', type=str, required=True, help='dataディレクトリ直下の入力CSVファイル名')
        parser.add_argument('--output', type=str, required=True, help='dataディレクトリ直下の出力CSVファイル名')
        parser.add_argument('--name', type=str, default="Convex Hull", help='GeoJSONのFeatureのnameプロパティ')
        parser.add_argument('--description', type=str, default="", help='GeoJSONのFeatureのdescriptionプロパティ')
        parser.add_argument('--contour-levels', type=str, help='等高線レベル（カンマ区切り、例: -10,-20,-30,-40,-50）')
        parser.add_argument('--grid-resolution', type=int, default=100, help='グリッド解像度（デフォルト: 100）')
        parser.add_argument('--alpha', type=float, default=0.01, help='境界形状のパラメータ（小さいほど詳細、デフォルト: 0.01）')

        args = parser.parse_args()

        # 入力ファイルが指定されていない場合はヘルプを表示して終了
        if not args.input:
            parser.print_help()
            return

        # 保存先のファイル名が指定されていない場合は終了
        if not args.output:
            parser.print_help()
            return

        # 入力ファイルのパス
        input_file_path = Path(data_dir, args.input)
        if not input_file_path.exists():
            logger.error(f"入力ファイルが存在しません: {input_file_path}")
            return

        # 出力ファイルのパス
        output_filename = args.output
        output_file_path = Path(data_dir, output_filename)

        logger.info("凸包GeoJSON生成処理を開始します")

        # CSVファイルから点データを読み込む
        logger.info(f"入力ファイル: {input_file_path}")
        points = read_csv_points(input_file_path)

        if points is None:
            logger.error("CSVファイルの読み込みに失敗しました")
            return
        if len(points) < 3:
            logger.error("凸包を作成するには最低3点必要です")
            return

        # 凸包を作成
        geojson_geometry, (center_lat, center_lon) = create_convex_hull_geometry(points)

        # 境界を作成（凸包から変更）
        # geojson_geometry, (center_lat, center_lon) = create_boundary_geometry(points, alpha=args.alpha)

        if geojson_geometry is None:
            logger.error("境界の作成に失敗しました")
            return

        # GeoJSONのFeatureCollectionを作成
        features = [
            {
                "type": "Feature",
                "geometry": geojson_geometry,
                "properties": {
                    "name": args.name,
                    "description": args.description if args.description else f"(lat, lon) boundary from {input_file_path.name}",
                    "source": input_file_path.name,
                    "link": "./index-bathymetric-data-dev.html",
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "type": "boundary"
                }
            }
        ]

        # 等高線レベルの解析
        contour_levels = None
        if args.contour_levels:
            try:
                contour_levels = [float(x.strip()) for x in args.contour_levels.split(',')]
                logger.info(f"指定された等高線レベル: {contour_levels}")
            except ValueError:
                logger.warning("等高線レベルのパースに失敗しました。自動設定を使用します。")

        # 等高線を生成
        contour_features = create_contours_from_points(
            points,
            levels=contour_levels,
            grid_resolution=args.grid_resolution
        )

        # 等高線のFeaturesを追加
        features.extend(contour_features)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        # GeoJSONファイルを保存
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, ensure_ascii=False, indent=2)
            logger.info(f"GeoJSONファイルを保存しました: {output_file_path}")
        except Exception as e:
            logger.error(f"ファイルの保存中にエラーが発生しました: {e}")
            sys.exit(1)

    #
    # 実行
    #
    main()
