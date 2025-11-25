#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSVデータの凸包をGeoJSON形式で保存します。

"""

SCRIPT_DESCRIPTION: str = 'Create Boundary GeoJSON from CSV'

# デフォルト設定
DEFAULT_AREA_NAME = "Area no name"

# alpha-shape境界パラメータ
# 小さいほど境界がタイト（陸地を避ける）、大きいほど緩い（広範囲をカバー）
# 座標が度単位なので0.001-0.01の範囲を推奨
DEFAULT_ALPHA = 0.001

# グリッド補間解像度（ピクセル数）
# 大きいほど等高線が詳細になるが、処理時間とメモリを消費
DEFAULT_GRID_RESOLUTION = 100

# 等高線レベルの自動生成時の分割数
DEFAULT_CONTOUR_LEVELS = 11

# 海底地形図を３次元表示するHTMLファイルへのリンク
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

# ================================================================================
# ジオメトリ生成関数
# ================================================================================

def create_boundary_geometry(points: List[Tuple[float, float, float]], alpha: float = 0.01) -> Tuple[Dict[str, Any], Tuple[float, float]]:
    """
    点群から境界（concave hull / alpha-shape）を作成し、GeoJSON形式で返す

    alpha-shapeアルゴリズムの仕組み:
    1. ドロネー三角分割で全ての点を三角形で結ぶ
    2. 各三角形の外接円半径を計算
    3. alpha値より小さい外接円の三角形のみを採用
    4. 採用された三角形を結合して境界を作成

    alphaパラメータの効果:
    - 小さい値（例: 0.001）→ タイトな境界、陸地を避ける
    - 大きい値（例: 0.01） → 緩い境界、広範囲をカバー

    Args:
        points: (lat, lon, depth)のタプルのリスト
        alpha: 境界の詳細度パラメータ（推奨: 0.001-0.01）
               ※度単位の座標系なので小さい値を使用

    Returns:
        Tuple[Dict[str, Any], Tuple[float, float]]:
            - GeoJSON形式の辞書（Polygon geometry）
            - 中心座標(lat, lon)のタプル
    """
    from shapely.ops import unary_union

    # (lon, lat)の配列を作成（GeoJSONは経度が先）
    coords = np.array([(lon, lat) for lat, lon, depth in points])

    logger.info(f"境界を計算中... (alpha={alpha})")

    try:
        # ステップ1: ドロネー三角分割
        tri = Delaunay(coords)

        # ステップ2: alpha-shape用の三角形フィルタリング
        triangles = []
        total_triangles = len(tri.simplices)
        accepted_triangles = 0

        for simplex in tri.simplices:
            # 三角形の3辺の長さを計算
            pts = coords[simplex]
            a = np.linalg.norm(pts[0] - pts[1])
            b = np.linalg.norm(pts[1] - pts[2])
            c = np.linalg.norm(pts[2] - pts[0])

            # ヘロンの公式で面積を計算
            s = (a + b + c) / 2
            if s * (s - a) * (s - b) * (s - c) <= 0:
                continue
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            if area == 0:
                continue

            # 外接円の半径を計算（R = abc / 4A）
            circum_r = (a * b * c) / (4 * area)

            # alpha判定: 外接円半径がalpha未満なら採用
            # 小さい三角形のみが残り、境界がタイトになる
            if circum_r < alpha:
                triangle = Polygon([coords[simplex[0]],
                                  coords[simplex[1]],
                                  coords[simplex[2]]])
                triangles.append(triangle)
                accepted_triangles += 1

        logger.info(f"三角形: {accepted_triangles}/{total_triangles} 個を使用 (alpha={alpha})")

        if not triangles:
            logger.warning("alpha-shapeの生成に失敗。凸包を使用します。")
            raise ValueError("No triangles")

        # ステップ3: 三角形を結合して境界ポリゴンを作成
        boundary = unary_union(triangles)

        # ステップ4: 最大のポリゴンのみを使用（小さな断片を除去）
        if boundary.geom_type == 'MultiPolygon':
            boundary = max(boundary.geoms, key=lambda p: p.area)

        # ステップ5: 外側の輪郭のみを取得（内側の穴を除去）
        if boundary.geom_type == 'Polygon':
            boundary = Polygon(boundary.exterior.coords)

        logger.info(f"alpha-shapeの生成に成功")

    except Exception as e:
        # フォールバック: 凸包を使用
        logger.warning(f"alpha-shape生成エラー: {e}。凸包を使用します。")
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
                                              grid_resolution: int = 100,
                                              generate_contours: bool = True,
                                              generate_polygons: bool = True,
                                              boundary_polygon = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        generate_contours: 等高線を生成するかどうか
        generate_polygons: 水深ポリゴンを生成するかどうか
        boundary_polygon: 境界ポリゴン（Shapely Polygon）。指定すると等高線・ポリゴンをクリッピング

    Returns:
        Tuple[List[Dict], List[Dict]]:
            - contour_features: 等高線のGeoJSON Featureリスト（LineString）
            - polygon_features: 水深ポリゴンのGeoJSON Featureリスト（Polygon）
    """

    # ==================== データ準備 ====================

    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])
    depths = np.array([p[2] for p in points])

    # ==================== グリッド作成 ====================

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    grid_lat = np.linspace(lat_min, lat_max, grid_resolution)
    grid_lon = np.linspace(lon_min, lon_max, grid_resolution)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    # ==================== グリッド補間（cubic spline） ====================

    logger.info(f"グリッド補間中... (解像度: {grid_resolution}x{grid_resolution})")

    grid_depth = griddata(
        (lons, lats),
        depths,
        (grid_lon_mesh, grid_lat_mesh),
        method='cubic'
    )

    # ==================== 等深線レベルの設定 ====================

    if levels is None:
        depth_min, depth_max = np.nanmin(depths), np.nanmax(depths)
        # 10段階に分割
        levels = np.linspace(depth_min, depth_max, 11)

    logger.info(f"等高線と水深ポリゴンを生成中... (レベル数: {len(levels)})")

    contour_features = []
    polygon_features = []

    fig, ax = plt.subplots()

    # ==================== 水深ポリゴン生成 ====================

    if generate_polygons:
        cs_filled = ax.contourf(grid_lon_mesh, grid_lat_mesh, grid_depth, levels=levels, extend='neither')

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

                    # ポリゴンを作成
                    poly = Polygon(coordinates)

                    # 境界でクリッピング
                    if boundary_polygon is not None:
                        try:
                            poly = poly.intersection(boundary_polygon)
                            if poly.is_empty:
                                continue
                        except Exception as e:
                            logger.warning(f"ポリゴンのクリッピングに失敗: {e}")
                            continue

                    # GeoJSON座標に変換
                    if poly.geom_type == 'Polygon':
                        clipped_coords = [[list(coord) for coord in poly.exterior.coords]]
                    elif poly.geom_type == 'MultiPolygon':
                        # MultiPolygonの場合は各ポリゴンを個別に追加
                        for sub_poly in poly.geoms:
                            clipped_coords = [[list(coord) for coord in sub_poly.exterior.coords]]
                            feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": clipped_coords
                                },
                                "properties": {
                                    "depth": float(depth_avg),
                                    "depth_min": float(depth_min),
                                    "depth_max": float(depth_max),
                                    "type": "depth_polygon"
                                }
                            }
                            polygon_features.append(feature)
                        continue
                    else:
                        continue

                    # Polygonとして追加
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": clipped_coords
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

    # ==================== 等深線生成 ====================

    if generate_contours:
        cs_lines = ax.contour(grid_lon_mesh, grid_lat_mesh, grid_depth, levels=levels)

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

                    # LineStringを作成
                    line = LineString(coordinates)

                    # 境界でクリッピング
                    if boundary_polygon is not None:
                        try:
                            line = line.intersection(boundary_polygon)
                            if line.is_empty:
                                continue
                        except Exception as e:
                            logger.warning(f"等高線のクリッピングに失敗: {e}")
                            continue

                    # GeoJSON座標に変換
                    if line.geom_type == 'LineString':
                        clipped_coords = [list(coord) for coord in line.coords]
                    elif line.geom_type == 'MultiLineString':
                        # MultiLineStringの場合は各ラインを個別に追加
                        for sub_line in line.geoms:
                            clipped_coords = [list(coord) for coord in sub_line.coords]
                            feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": clipped_coords
                                },
                                "properties": {
                                    "depth": float(level),
                                    "type": "contour"
                                }
                            }
                            contour_features.append(feature)
                        continue
                    else:
                        continue

                    # LineStringとして追加
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": clipped_coords
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
        logger.info("GeoJSON生成処理を開始します")
        logger.info("=" * 60)

        # ==================== STEP 1: ファイルパス設定 ====================

        input_file_path = Path(data_dir, args.input)
        output_file_path = Path(data_dir, args.output)

        # 入力ファイルの存在確認
        if not input_file_path.exists():
            logger.error(f"入力ファイルが存在しません: {input_file_path}")
            return

        logger.info(f"入力ファイル: {input_file_path}")
        logger.info(f"出力ファイル: {output_file_path}")

        # ==================== STEP 2: CSVファイル読み込み ====================

        points = read_csv_points(input_file_path)

        if not points:
            logger.error("データポイントが読み込まれませんでした")
            return

        logger.info(f"読み込んだデータポイント数: {len(points)}")

        # ==================== STEP 3: 境界ポリゴン作成 ====================

        logger.info("-" * 60)
        logger.info("境界ジオメトリの作成（alpha-shape）")
        logger.info("-" * 60)

        geojson_geometry, (center_lat, center_lon) = create_boundary_geometry(
            points,
            alpha=args.alpha
        )

        if geojson_geometry is None:
            logger.error("境界の作成に失敗しました")
            return

        # Shapely Polygonオブジェクトを作成（クリッピング用）
        from shapely.geometry import shape
        boundary_polygon = shape(geojson_geometry)


        # ==================== STEP 4: GeoJSON Feature作成 ====================

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

        # ==================== STEP 5: 等高線・水深ポリゴン生成（オプション） ====================

        contour_levels = None
        if args.contour_levels:
            try:
                contour_levels = [float(x.strip()) for x in args.contour_levels.split(',')]
                logger.info(f"指定された等高線レベル: {contour_levels}")
            except ValueError:
                logger.warning("等高線レベルのパースに失敗しました。自動設定を使用します。")

        # 等高線と水深ポリゴンの生成判定
        # どちらか一方でも生成する場合のみTrueにする
        need_contours = not args.no_contours
        need_polygons = not args.no_depth_polygons
        generate_contours_or_polygons = need_contours or need_polygons

        if generate_contours_or_polygons:
            logger.info("-" * 60)
            logger.info("等高線と水深ポリゴンの生成")
            logger.info("-" * 60)

            contour_features, depth_polygon_features = create_contours_and_polygons_from_points(
                points,
                levels=contour_levels,
                grid_resolution=args.grid_resolution,
                generate_contours=need_contours,
                generate_polygons=need_polygons,
                boundary_polygon=boundary_polygon  # 境界ポリゴンを渡す
            )

            # 水深ポリゴンの追加
            if need_polygons:
                features.extend(depth_polygon_features)
                logger.info(f"✓ 水深ポリゴンを追加: {len(depth_polygon_features)} 個")

            # 等高線の追加
            if need_contours:
                features.extend(contour_features)
                logger.info(f"✓ 等高線を追加: {len(contour_features)} 本")

        else:
            logger.info("等高線と水深ポリゴンの生成をスキップしました")

        # ==================== STEP 6: GeoJSON出力 ====================

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        # ファイルに保存
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        # ==================== 結果サマリー ====================

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
