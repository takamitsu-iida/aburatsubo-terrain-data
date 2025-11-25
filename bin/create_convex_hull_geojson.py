#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSVデータの凸包をGeoJSON形式で保存します。

"""

SCRIPT_DESCRIPTION: str = 'Create Boundary GeoJSON from CSV'

# デフォルト設定
DEFAULT_AREA_NAME = "Area no name"
DEFAULT_ALPHA = 0.01
DEFAULT_GRID_RESOLUTION = 100
DEFAULT_CONTOUR_LEVELS = 11  # 自動生成時のレベル数

# リンク先
DEFAULT_LINK = "./index-bathymetric-data-dev.html"

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

def create_boundary_geometry(points: List[Tuple[float, float, float]], alpha: float = 0.01) -> Tuple[Dict[str, Any], Tuple[float, float]]:
    """
    点群から境界（concave hull）を作成し、GeoJSON形式で返す

    ドロネー三角分割の外周エッジを抽出して境界を構築します。
    失敗した場合は凸包にフォールバックします。

    Args:
        points: (lat, lon, depth)のタプルのリスト
        alpha: 境界の詳細度パラメータ（現在未使用、将来の拡張用）

    Returns:
        Tuple[Dict[str, Any], Tuple[float, float]]:
            - GeoJSON形式の辞書（Polygon geometry）
            - 中心座標(lat, lon)のタプル

    Raises:
        なし（エラー時は凸包にフォールバック）

    Example:
        >>> points = [(35.1, 139.5, -10.0), (35.2, 139.6, -15.0), ...]
        >>> geom, (lat, lon) = create_boundary_geometry(points)
        >>> print(geom['type'])
        'Polygon'
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


def create_contours_and_polygons_from_points(points: List[Tuple[float, float, float]],
                                              levels: List[float] = None,
                                              grid_resolution: int = 100) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    点群から等高線と水深ポリゴンを同時に生成し、GeoJSON Feature のリストを返す

    同じcontourオブジェクトから生成することで、等高線とポリゴンのズレを防ぎます。
    グリッド補間 → contourf/contour → GeoJSON変換の順に処理します。

    Args:
        points: (lat, lon, depth)のタプルのリスト
        levels: 等深線のレベル（深度）のリスト。
                Noneの場合は深度の最小値〜最大値を11段階に自動分割
        grid_resolution: グリッド補間の解像度（ピクセル数）
                        大きいほど詳細だがメモリと時間を消費

    Returns:
        Tuple[List[Dict], List[Dict]]:
            - contour_features: 等高線のGeoJSON Featureリスト（LineString）
            - polygon_features: 水深ポリゴンのGeoJSON Featureリスト（Polygon）

    Note:
        - matplotlib の contourf.allsegs/contour.allsegs を使用
        - allsegs が利用できない場合は空のリストを返す

    Example:
        >>> points = [(35.1, 139.5, -10.0), ...]
        >>> contours, polygons = create_contours_and_polygons_from_points(points)
        >>> len(contours), len(polygons)
        (50, 100)
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

    # 等深線レベルの設定
    if levels is None:
        depth_min, depth_max = np.nanmin(depths), np.nanmax(depths)
        # 10段階に分割
        levels = np.linspace(depth_min, depth_max, 11)

    logger.info(f"等高線と水深ポリゴンを生成中... (レベル数: {len(levels)})")

    contour_features = []
    polygon_features = []

    fig, ax = plt.subplots()

    # contourfとcontourを同時に生成
    cs_filled = ax.contourf(grid_lon_mesh, grid_lat_mesh, grid_depth, levels=levels, extend='neither')
    cs_lines = ax.contour(grid_lon_mesh, grid_lat_mesh, grid_depth, levels=levels)

    # 水深ポリゴンを生成（contourf）
    if hasattr(cs_filled, 'allsegs'):
        logger.info("水深ポリゴンを生成中...")

        for level_idx in range(len(cs_filled.allsegs)):
            segments = cs_filled.allsegs[level_idx]

            # レベルの範囲を計算
            if level_idx < len(levels) - 1:
                depth_min = levels[level_idx]
                depth_max = levels[level_idx + 1]
            else:
                continue

            depth_avg = (depth_min + depth_max) / 2

            for segment in segments:
                if len(segment) < 3:
                    continue

                # GeoJSON座標は [lon, lat] の順序
                coordinates = [[float(v[0]), float(v[1])] for v in segment]

                # 閉じたポリゴンにする
                if coordinates[0] != coordinates[-1]:
                    coordinates.append(coordinates[0])

                # Polygonとして追加
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coordinates]
                    },
                    "properties": {
                        "depth": float(depth_avg),
                        "depth_min": float(depth_min),
                        "depth_max": float(depth_max),
                        "type": "depth_polygon"
                    }
                }
                polygon_features.append(feature)
    else:
        logger.warning("allsegs属性が利用できません。水深ポリゴンを生成できませんでした。")

    # 等高線を生成（contour）
    if hasattr(cs_lines, 'allsegs'):
        logger.info("等高線を生成中...")

        for level_idx in range(len(cs_lines.allsegs)):
            segments = cs_lines.allsegs[level_idx]

            if level_idx < len(levels):
                level = levels[level_idx]
            else:
                continue

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
        logger.warning("allsegs属性が利用できません。等高線を生成できませんでした。")

    plt.close(fig)

    logger.info(f"水深ポリゴンを {len(polygon_features)} 個、等高線を {len(contour_features)} 本生成しました")

    return contour_features, polygon_features


# ================================================================================
# メイン処理
# ================================================================================

if __name__ == '__main__':

    def main():
        """メイン処理"""

        epilog="""
使用例:
# 基本的な使用
%(prog)s --input input.csv --output output.geojson

# 水深ポリゴンなし（ファイルサイズ削減）
%(prog)s --input input.csv --output output.geojson --no-depth-polygons

# 等高線レベルを指定
%(prog)s --input input.csv --output output.geojson --contour-levels "-10,-20,-30,-40,-50"

# 境界のみ（最小サイズ）
%(prog)s --input input.csv --output output.geojson --no-depth-polygons --no-contours
        """

        # 引数処理
        parser = argparse.ArgumentParser(
            description=SCRIPT_DESCRIPTION,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=epilog
        )

        # 必須引数
        required = parser.add_argument_group('必須引数')
        required.add_argument('--input', type=str, required=True, help='入力CSVファイル名（dataディレクトリ直下）')
        required.add_argument('--output', type=str, required=True, help='出力GeoJSONファイル名（dataディレクトリ直下）')

        # オプション引数（基本設定）
        basic = parser.add_argument_group('基本設定')
        basic.add_argument('--name', type=str, default=DEFAULT_AREA_NAME, help=f'エリアの名前（デフォルト: {DEFAULT_AREA_NAME}）')
        basic.add_argument('--description', type=str, default="", help='エリアの説明')

        # オプション引数（詳細設定）
        advanced = parser.add_argument_group('詳細設定')
        advanced.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, help=f'境界の詳細度（小さいほど詳細、デフォルト: {DEFAULT_ALPHA}）')
        advanced.add_argument('--grid-resolution', type=int, default=DEFAULT_GRID_RESOLUTION, help=f'グリッド解像度（デフォルト: {DEFAULT_GRID_RESOLUTION}）')
        advanced.add_argument('--contour-levels', type=str, help='等高線レベル（カンマ区切り、例: -10,-20,-30,-40,-50）')

        # オプション引数（出力制御）
        output_control = parser.add_argument_group('出力制御（ファイルサイズ削減）')
        output_control.add_argument('--no-depth-polygons', action='store_true', help='水深ポリゴンを生成しない')
        output_control.add_argument('--no-contours', action='store_true', help='等高線を生成しない')

        args = parser.parse_args()

        logger.info("=" * 60)
        logger.info("凸包GeoJSON生成処理を開始します")
        logger.info("=" * 60)

        # ファイルパスの構築
        input_file_path = Path(data_dir, args.input)
        output_file_path = Path(data_dir, args.output)

        # 入力ファイルの存在確認
        if not input_file_path.exists():
            logger.error(f"入力ファイルが存在しません: {input_file_path}")
            return

        logger.info(f"入力ファイル: {input_file_path}")
        logger.info(f"出力ファイル: {output_file_path}")

        # CSVファイルを読み込む
        points = read_csv_points(input_file_path)

        if not points:
            logger.error("データポイントが読み込まれませんでした")
            return

        logger.info(f"読み込んだデータポイント数: {len(points)}")

        # 境界を作成
        geojson_geometry, (center_lat, center_lon) = create_boundary_geometry(
            points,
            alpha=args.alpha
        )

        if geojson_geometry is None:
            logger.error("境界の作成に失敗しました")
            return

        # 基本Feature（境界）を作成
        features = [{
            "type": "Feature",
            "geometry": geojson_geometry,
            "properties": {
                "name": args.name,
                "description": args.description or f"Boundary from {input_file_path.name}",
                "source": input_file_path.name,
                "link": DEFAULT_LINK,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "type": "boundary"
            }
        }]

        # 等高線レベルの解析
        contour_levels = None
        if args.contour_levels:
            try:
                contour_levels = [float(x.strip()) for x in args.contour_levels.split(',')]
                logger.info(f"指定された等高線レベル: {contour_levels}")
            except ValueError:
                logger.warning("等高線レベルのパースに失敗しました。自動設定を使用します。")

        # 等高線と水深ポリゴンの生成判定
        generate_contours_or_polygons = not args.no_depth_polygons or not args.no_contours

        if generate_contours_or_polygons:
            logger.info("-" * 60)
            logger.info("等高線と水深ポリゴンの生成")
            logger.info("-" * 60)

            contour_features, depth_polygon_features = create_contours_and_polygons_from_points(
                points,
                levels=contour_levels,
                grid_resolution=args.grid_resolution
            )

            # 水深ポリゴンの追加
            if not args.no_depth_polygons:
                features.extend(depth_polygon_features)
                logger.info(f"✓ 水深ポリゴンを追加: {len(depth_polygon_features)} 個")
            else:
                logger.info("✗ 水深ポリゴンの生成をスキップ")

            # 等高線の追加
            if not args.no_contours:
                features.extend(contour_features)
                logger.info(f"✓ 等高線を追加: {len(contour_features)} 本")
            else:
                logger.info("✗ 等高線の生成をスキップ")
        else:
            logger.info("等高線と水深ポリゴンの生成をスキップしました")

        # GeoJSON作成
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        # ファイルに保存
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        # 結果サマリー
        logger.info("=" * 60)
        logger.info("生成完了")
        logger.info("=" * 60)
        logger.info(f"出力ファイル: {output_file_path}")
        logger.info(f"総Feature数: {len(features)}")
        logger.info(f"  - 境界: 1")
        logger.info(f"  - 水深ポリゴン: {len([f for f in features if f['properties']['type'] == 'depth_polygon'])}")
        logger.info(f"  - 等高線: {len([f for f in features if f['properties']['type'] == 'contour'])}")
        logger.info(f"ファイルサイズ: {output_file_path.stat().st_size / 1024:.1f} KB")
        logger.info("=" * 60)


    #
    # 実行
    #
    try:
        main()
    except KeyboardInterrupt:
        logger.info("処理がユーザーにより中断されました")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"予期せぬエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)
