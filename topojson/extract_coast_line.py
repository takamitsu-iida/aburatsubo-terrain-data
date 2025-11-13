#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
海岸線を形成している点を抽出するスクリプト

TopoJSON形式の地理データから海岸線の座標を抽出し、
指定した範囲内の点を一定間隔でサンプリングしてCSV形式で出力します。
また、結果を可視化して画像として保存します。
"""

#
# 標準ライブラリのインポート
#
import logging
import json
import sys

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable

# WSL1 固有の numpy 警告を抑制
# https://github.com/numpy/numpy/issues/18900
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, module=r"numpy.*", message=r"Signature b")

#
# 外部ライブラリのインポート
#
try:
    from shapely.geometry import LineString
    from shapely.ops import transform
    import pyproj
    import matplotlib.pyplot as plt
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

#
# ログ設定
#

# ログファイルの名前
log_file: str = app_path.with_suffix('.log').name

# ログファイルを置くディレクトリ
log_dir: Path = app_home.joinpath('log')
log_dir.mkdir(exist_ok=True)

# ログファイルのパス
log_path: Path = log_dir.joinpath(log_file)

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
logger: logging.Logger = logging.getLogger(__name__)

# ログレベル設定
logger.setLevel(logging.INFO)

# フォーマット
formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 標準出力へのハンドラ
stdout_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)

# ログファイルのハンドラ
file_handler: logging.FileHandler = logging.FileHandler(
    log_dir.joinpath(log_file), 'a+', encoding='utf-8'
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


# ================================================================================
# 座標変換の設定
# ================================================================================

# EPSG:4326 (WGS84緯度経度) から EPSG:3857 (Web メルカトル図法) への変換
# メルカトル図法ではメートル単位で距離計算ができる
proj_4326_to_3857: Callable[[float, float], Tuple[float, float]] = pyproj.Transformer.from_crs(
    "EPSG:4326", "EPSG:3857", always_xy=True
).transform

# EPSG:3857 から EPSG:4326 への逆変換
proj_3857_to_4326: Callable[[float, float], Tuple[float, float]] = pyproj.Transformer.from_crs(
    "EPSG:3857", "EPSG:4326", always_xy=True
).transform


# ================================================================================
# 関数定義
# ================================================================================

def is_within_bounds(
    lat: float,
    lon: float,
    se_lat: float,
    nw_lat: float,
    nw_lon: float,
    se_lon: float
) -> bool:
    """
    指定された座標が範囲内に収まっているかチェックする

    Args:
        lat (float): 緯度
        lon (float): 経度
        se_lat (float): 南東の緯度
        nw_lat (float): 北西の緯度
        nw_lon (float): 北西の経度
        se_lon (float): 南東の経度

    Returns:
        bool: 範囲内ならTrue、範囲外ならFalse
    """
    return (se_lat <= lat <= nw_lat) and (nw_lon <= lon <= se_lon)


def decode_arcs(
    arcs_data: List[List[List[int]]],
    arc_indices: List[int],
    transform_params: Optional[Dict[str, Any]] = None
) -> List[List[float]]:
    """
    TopoJSONのarcsをデコードして座標リストに変換する

    TopoJSONではarcsがデルタエンコーディング（差分形式）で保存されており、
    前の点からの相対位置で表現されている。
    また、負のインデックスは逆順を意味する。

    Args:
        arcs_data (List[List[List[int]]]): TopoJSONの全arc配列
        arc_indices (List[int]): 使用するarcのインデックスリスト
        transform_params (Optional[Dict[str, Any]]): スケールと平行移動のパラメータ

    Returns:
        List[List[float]]: [[経度, 緯度], ...] の座標リスト
    """
    coordinates: List[List[float]] = []

    # 各arcインデックスに対して処理
    for arc_index in arc_indices:
        if arc_index < 0:
            # 負のインデックス: ビット反転して逆順で取得
            # ~arc_index は (-arc_index - 1) と同じ
            arc: List[List[int]] = list(reversed(arcs_data[~arc_index]))
        else:
            # 正のインデックス: そのまま取得
            arc = arcs_data[arc_index]

        # デルタエンコーディングをデコード
        # 各点は前の点からの相対位置（差分）で表現されている
        x: int = 0
        y: int = 0  # 累積座標の初期値
        for point in arc:
            # 差分を加算して累積座標を計算
            x += point[0]
            y += point[1]

            # transform パラメータを適用して実際の座標に変換
            if transform_params:
                # スケールと平行移動を適用
                # 実座標 = (累積値 × スケール) + オフセット
                scale: List[float] = transform_params.get('scale', [1.0, 1.0])
                translate: List[float] = transform_params.get('translate', [0.0, 0.0])
                real_x: float = x * scale[0] + translate[0]
                real_y: float = y * scale[1] + translate[1]
                coordinates.append([real_x, real_y])
            else:
                # transformパラメータがない場合は累積値をそのまま使用
                coordinates.append([float(x), float(y)])

    return coordinates


def sample_linestring(linestring: LineString, distance_m: int) -> List[Tuple[float, float]]:
    """
    LineStringをメートル単位の距離で等間隔にサンプリングする

    海岸線に沿って一定間隔で点を抽出する。
    座標系がメートル単位（EPSG:3857など）である必要がある。

    Args:
        linestring (LineString): サンプリング対象のLineStringジオメトリ
        distance_m (int): サンプリング間隔（メートル）

    Returns:
        List[Tuple[float, float]]: [(x, y), ...] のサンプリングされた座標リスト
    """
    points: List[Tuple[float, float]] = []
    length: float = linestring.length  # 線の全長（メートル）

    # 長さが0またはNaN（無効値）の場合は空リストを返す
    if length == 0 or length != length:  # NaNチェック（NaN != NaN は True）
        return points

    # 始点から終点まで、指定間隔でサンプリング
    for current_dist in range(0, int(length), distance_m):
        # 始点からの距離を指定して線上の点を取得
        point = linestring.interpolate(current_dist)
        points.append((point.x, point.y))

    # 最後の点（終点）も必ず追加
    last_point = linestring.interpolate(length)
    # 既に同じ点が追加されていなければ追加
    if points and (last_point.x, last_point.y) != points[-1]:
        points.append((last_point.x, last_point.y))
    elif not points:
        # リストが空の場合（線が短すぎる場合）も終点を追加
        points.append((last_point.x, last_point.y))

    return points


def extract_coast_line(
    topojson_filepath: str,
    object_key: str,
    sampling_distance_m: int,
    se_lat: float,
    nw_lat: float,
    nw_lon: float,
    se_lon: float,
    proj_4326_to_3857: Callable[[float, float], Tuple[float, float]],
    proj_3857_to_4326: Callable[[float, float], Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    TopoJSONファイルから海岸線を抽出し、GPS座標のリストを返す

    Args:
        topojson_filepath (str): TopoJSONファイルのパス
        object_key (str): TopoJSON内のオブジェクトキー
        sampling_distance_m (int): サンプリング間隔（メートル）
        se_lat (float): 南東の緯度
        nw_lat (float): 北西の緯度
        nw_lon (float): 北西の経度
        se_lon (float): 南東の経度
        proj_4326_to_3857 (Callable): WGS84からWeb Mercatorへの変換関数
        proj_3857_to_4326 (Callable): Web MercatorからWGS84への変換関数

    Returns:
        List[Tuple[float, float]]: [(緯度, 経度), ...] の座標リスト

    Raises:
        FileNotFoundError: TopoJSONファイルが見つからない場合
        ValueError: オブジェクトキーが存在しない場合
    """
    # サンプリングされたGPS座標を格納するリスト
    sampled_gps_points: List[Tuple[float, float]] = []

    # TopoJSONファイルを読み込む
    with open(topojson_filepath, 'r', encoding='utf-8') as f:
        topojson_data: Dict[str, Any] = json.load(f)

    # TopoJSONの各要素を取得
    arcs: List[List[List[int]]] = topojson_data.get('arcs', [])  # arc配列（圧縮された座標データ）
    objects: Dict[str, Any] = topojson_data.get('objects', {})  # ジオメトリオブジェクト
    transform_params: Optional[Dict[str, Any]] = topojson_data.get('transform', None)  # 座標変換パラメータ

    # 指定されたオブジェクトキーが存在するか確認
    if object_key not in objects:
        raise ValueError(
            f"オブジェクト '{object_key}' が見つかりません。"
            f"利用可能なオブジェクト: {list(objects.keys())}"
        )

    # 対象オブジェクトとそのジオメトリを取得
    target_object: Dict[str, Any] = objects[object_key]
    geometries: List[Dict[str, Any]] = target_object.get('geometries', [])

    logger.info(f"処理するジオメトリ数: {len(geometries)}")

    # 各ジオメトリ（Polygon）を処理
    for i, geometry in enumerate(geometries):
        geom_type: str = geometry.get('type', '')

        # Polygonタイプのみ処理
        if geom_type == 'Polygon':
            # Polygonの外周のみを取得（最初のring）
            # arcs[0]は外周、arcs[1]以降は穴（内側の境界）
            arc_indices: List[int] = geometry.get('arcs', [[]])[0]

            # arcをデコードして座標リストに変換
            coordinates: List[List[float]] = decode_arcs(arcs, arc_indices, transform_params)

            # 座標が2点未満の場合はスキップ（線を作れない）
            if len(coordinates) < 2:
                continue

            # 座標リストからLineStringジオメトリを作成
            linestring: LineString = LineString(coordinates)

            # ジオメトリの有効性をチェック
            if not linestring.is_valid or linestring.is_empty:
                logger.warning(f"ジオメトリ {i}: 無効なLineStringをスキップ")
                continue

            # 緯度経度（EPSG:4326）からメートル単位の座標系（EPSG:3857）へ変換
            # これにより距離ベースのサンプリングが可能になる
            try:
                projected_geom: LineString = transform(proj_4326_to_3857, linestring)
            except Exception as e:
                logger.warning(f"ジオメトリ {i}: 座標変換エラー - {e}")
                continue

            # メートル単位でサンプリング
            sampled_meter_points: List[Tuple[float, float]] = sample_linestring(
                projected_geom, sampling_distance_m
            )

            # サンプリングされた点を緯度経度に戻し、範囲内のものだけ保存
            for mx, my in sampled_meter_points:
                # EPSG:3857からEPSG:4326へ逆変換
                lon: float
                lat: float
                lon, lat = proj_3857_to_4326(mx, my)

                # 指定範囲内の座標のみ追加
                if is_within_bounds(lat, lon, se_lat, nw_lat, nw_lon, se_lon):
                    sampled_gps_points.append((lat, lon))
        else:
            # Polygon以外のタイプはスキップ
            logger.debug(f"ジオメトリ {i}: サポートされていないタイプ '{geom_type}' をスキップ")

    logger.info(f"サンプリングされたGPS座標の数: {len(sampled_gps_points)}")

    return sampled_gps_points


def save_to_csv(
    gps_points: List[Tuple[float, float]],
    output_filepath: str
) -> None:
    """
    GPS座標リストをCSVファイルとして保存する

    Args:
        gps_points (List[Tuple[float, float]]): [(緯度, 経度), ...] の座標リスト
        output_filepath (str): 出力CSVファイルのパス

    Returns:
        None
    """
    with open(output_filepath, 'w', encoding='utf-8') as csv_f:
        # ヘッダー行を書き込み
        # csv_f.write("latitude,longitude\n")
        # 各座標を1行ずつ書き込み
        for lat, lon in gps_points:
            csv_f.write(f"{lat:.6f},{lon:.6f},0.0,0.0\n")

    logger.info(f"結果を {output_filepath} に保存しました。")


def save_visualization(
    gps_points: List[Tuple[float, float]],
    output_filepath: str,
    nw_lat: float,
    nw_lon: float,
    se_lat: float,
    se_lon: float,
    title: str = '海岸線サンプリング結果'
) -> None:
    """
    GPS座標リストを可視化して画像として保存する

    Args:
        gps_points (List[Tuple[float, float]]): [(緯度, 経度), ...] の座標リスト
        output_filepath (str): 出力画像ファイルのパス
        nw_lat (float): 北西の緯度（境界線表示用）
        nw_lon (float): 北西の経度（境界線表示用）
        se_lat (float): 南東の緯度（境界線表示用）
        se_lon (float): 南東の経度（境界線表示用）
        title (str): グラフのタイトル

    Returns:
        None
    """
    if not gps_points:
        logger.warning("可視化する座標が0件です。")
        return

    # 緯度と経度のリストに分離
    lats: Tuple[float, ...]
    lons: Tuple[float, ...]
    lats, lons = zip(*gps_points)

    # 図を作成
    plt.figure(figsize=(12, 10))

    # 散布図をプロット（青い点で表示）
    plt.scatter(lons, lats, s=1, c='blue', alpha=0.6)

    # 軸ラベルとタイトル
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)

    # グリッド線を表示
    plt.grid(True, alpha=0.3)

    # アスペクト比を等しく（緯度経度の縮尺を揃える）
    plt.axis('equal')

    # 範囲の境界線を赤い破線で表示
    plt.axvline(x=nw_lon, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='boundary')
    plt.axvline(x=se_lon, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axhline(y=nw_lat, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axhline(y=se_lat, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

    # 凡例を表示
    plt.legend()

    # 画像ファイルとして保存（300dpi、余白を最小化）
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    logger.info(f"可視化結果を {output_filepath} に保存しました。")

    # メモリ解放のため図を閉じる
    plt.close()


# ================================================================================
# メイン処理
# ================================================================================

if __name__ == '__main__':

    # ================================================================================
    # 設定パラメータ
    # ================================================================================

    # 抽出する範囲の指定（緯度経度）
    # NW: 北西（左上）の角
    NW_LAT: float = 35.167760706042124
    NW_LON: float = 139.6073867203739

    # SE: 南東（右下）の角
    SE_LAT: float = 35.15693719878452
    SE_LON: float = 139.63062538005556

    # 入力ファイルとオブジェクトキー
    topojson_filename: str = 'aburatsubo.json'  # TopoJSONファイルの名前
    topojson_filepath: str = app_home.joinpath('static', 'data', topojson_filename)
    object_key: str = 'miura'  # TopoJSONのobjects内で海岸線が格納されているキー

    # サンプリング間隔（メートル単位）
    # 海岸線に沿って何メートルごとに点を抽出するか
    sampling_distance_m: int = 10

    # 出力ファイル名
    output_csv_filename: str = 'coastline_points.csv'
    output_csv_path: str = app_home.joinpath('data', output_csv_filename)
    output_image_filename: str = 'coastline_visualization.png'
    output_image_path: str = app_home.joinpath('img', output_image_filename)


    def main() -> None:
        """メイン関数"""
        try:
            # 海岸線座標を抽出
            gps_points: List[Tuple[float, float]] = extract_coast_line(
                topojson_filepath,
                object_key,
                sampling_distance_m,
                SE_LAT,
                NW_LAT,
                NW_LON,
                SE_LON,
                proj_4326_to_3857,
                proj_3857_to_4326
            )

            # 座標が取得できたか確認
            if not gps_points:
                logger.warning("抽出された座標が0件です。")
                return

            # 最初の10点をコンソールに表示
            logger.info("最初の10点:")
            for i in range(min(10, len(gps_points))):
                logger.info(f"  緯度: {gps_points[i][0]:.6f}, 経度: {gps_points[i][1]:.6f}")

            # CSVファイルとして保存
            save_to_csv(gps_points, output_csv_path)

            # 可視化して画像として保存
            save_visualization(
                gps_points,
                output_image_path,
                NW_LAT,
                NW_LON,
                SE_LAT,
                SE_LON,
                title='Aburatsubo Coastline Sampling Results (Within Range)'
            )

        except FileNotFoundError as e:
            logger.error(f"ファイルが見つかりません: {e}")
        except ValueError as e:
            logger.error(f"値エラー: {e}")
        except Exception as e:
            logger.error(f"処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

    main()