#!/usr/bin/env python

# 四分木を実装したスクリプトです。

#
# 標準ライブラリのインポート
#
import logging
import math
import os
import sys

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# WSL1 固有の numpy 警告を抑制
# https://github.com/numpy/numpy/issues/18900
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, module=r"numpy.*", message=r"Signature b")

#
# 外部ライブラリのインポート
#
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # GPS座標から距離を計算
    from geopy.distance import distance

    # データを整形して表示
    from tabulate import tabulate

    # 空間補間・近傍探索
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    # クリギング補間
    from pykrige.ok import OrdinaryKriging

except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    sys.exit(1)

#
# ローカルファイルからインポート
#
try:
    from load_save_csv import load_csv, save_csv
except ImportError as e:
    print(f"ローカルモジュールのインポートに失敗しました: {e}")
    sys.exit(1)


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
file_handler = logging.FileHandler(os.path.join(log_dir, log_file), 'a+')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

#
# ここからスクリプト
#

class QuadtreeNode:
    """ 四分木のノードを表すクラス """

    def __init__(self, bounds: Tuple[float, float, float, float], level: int, parent: Optional['QuadtreeNode'] = None):
        # 領域 boundsは (lat1, lon1, lat2, lon2) の形式
        self.bounds: Tuple[float, float, float, float] = bounds

        # 階層の深さ
        self.level: int = level

        # ノードに含まれるポイントのリスト
        self.points: List[Dict[str, float]] = []

        # 子ノードのリスト
        self.children: List['QuadtreeNode'] = []

        # 親ノードへの参照
        self.parent: Optional['QuadtreeNode'] = parent

        # ノードの中心座標
        self.center: Tuple[float, float] = (
            (bounds[0] + bounds[2]) / 2,
            (bounds[1] + bounds[3]) / 2
        )

        # ノードの緯度・経度の長さ
        self.lat_length: float = abs(bounds[2] - bounds[0])
        self.lon_length: float = abs(bounds[3] - bounds[1])


    def is_leaf(self) -> bool:
        """ ノードが葉ノードかどうかを判定 """
        return len(self.children) == 0


    def subdivide(self) -> None:
        """ 4つの子ノードを作成 """
        lat1, lon1, lat2, lon2 = self.bounds

        # 中央座標
        c_lat, c_lon = self.center

        #
        # NW | NE
        # ---+---
        # SW | SE
        #
        self.children.append(QuadtreeNode((lat1,  lon1,  c_lat, c_lon), self.level + 1, self))  # NW
        self.children.append(QuadtreeNode((lat1,  c_lon, c_lat, lon2 ), self.level + 1, self))  # NE
        self.children.append(QuadtreeNode((c_lat, lon1,  lat2,  c_lon), self.level + 1, self))  # SW
        self.children.append(QuadtreeNode((c_lat, c_lon, lat2,  lon2 ), self.level + 1, self))  # SE


    def insert(self, point: Dict[str, float]) -> bool:
        """ ノードにポイントを挿入 """

        # 渡されたデータが範囲内かどうかを確認して、範囲外ならFalseを返す
        if not self.contains(point):
            return False

        # リーフノードの場合
        if self.is_leaf():
            # ポイント数が上限以下、または分割レベルが上限に達している場合は、points配列に追加して終了
            if len(self.points) <= Quadtree.MAX_POINTS or self.level >= Quadtree.LEVEL_LIMIT:
                self.points.append(point)
                return True
            else:
                # それ以外の場合はノードを分割
                self.subdivide()

        # 自分はリーフではないので、子ノードに順番に格納して、できたらTrueを返す
        for child in self.children:
            if child.insert(point):
                return True

        return False


    def contains(self, point: Dict[str, float]) -> bool:
        """ ノードの範囲内にポイントが含まれるかどうかを判定 """
        lat1, lon1, lat2, lon2 = self.bounds

        # 左辺と上辺は含み、右辺と下辺は含まない
        return lat1 <= point['lat'] < lat2 and lon1 <= point['lon'] < lon2


    def intersects(self, range: Tuple[float, float, float, float]) -> bool:
        """ ノードの範囲が指定された範囲と交差するかどうかを判定 """
        lat1, lon1, lat2, lon2 = self.bounds
        r_lat1, r_lon1, r_lat2, r_lon2 = range
        return not (r_lon1 >= lon2 or r_lon2 <= lon1 or r_lat1 >= lat2 or r_lat2 <= lat1)


    def query(self, range: Tuple[float, float, float, float]) -> List[Dict[str, float]]:
        """ 指定された範囲内のポイントを検索 """
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


    def contains_point_in_range(self, point: Dict[str, float], range: Tuple[float, float, float, float]) -> bool:
        """ 指定された範囲内にポイントが含まれるかどうかを判定 """
        r_lat1, r_lon1, r_lat2, r_lon2 = range
        return r_lat1 <= point['lat'] < r_lat2 and r_lon1 <= point['lon'] < r_lon2


    def get_nodes_at_level(self, level: int) -> List['QuadtreeNode']:
        """ 指定された深さのノードを取得 """
        nodes = []
        if self.level == level:
            nodes.append(self)
        elif self.level < level:
            for child in self.children:
                nodes.extend(child.get_nodes_at_level(level))
        return nodes


    def get_leaf_nodes(self) -> List['QuadtreeNode']:
        """ 葉ノードをすべて取得 """
        if self.is_leaf():
            return [self]
        else:
            leaf_nodes = []
            for child in self.children:
                leaf_nodes.extend(child.get_leaf_nodes())
            return leaf_nodes


    def average(self) -> Dict[str, float]:
        """ ノード内のポイントの平均値を計算 """
        n = len(self.points)
        if n == 0:
            return {}
        avg_lat = sum(p['lat'] for p in self.points) / n
        avg_lon = sum(p['lon'] for p in self.points) / n
        avg_depth = sum(p['depth'] for p in self.points) / n
        return {'lat': avg_lat, 'lon': avg_lon, 'depth': avg_depth}


class Quadtree:

    """ 四分木クラス """

    # 最大分割レベル
    # 四分木の領域の大きさに応じて初期化時に調整される
    LEVEL_LIMIT: int = 1

    # 最小グリッド幅（メートル単位）
    # 四分木の最大分割レベルを決定するために使用
    # 例えば、2メートルに設定すると、最小の葉ノードの一辺の長さが約2メートル以下（1.0～2.0）になる
    MIN_GRID_WIDTH: float = 2.0

    # ノードあたりの最大ポイント数
    # 3点でメッシュを作るので、4分割したときにそれぞれ3点ずつ格納されるといいな、ということで12点に設定
    MAX_POINTS: int = 12

    def __init__(self, bounds: Tuple[float, float, float, float]):

        # ルートノードを作成
        self.root = QuadtreeNode(bounds=bounds, level=0, parent=None)

        # boundsは (lat1, lon1, lat2, lon2) の形式で、NW, SEの対角線で囲まれた矩形領域を想定
        # boundsの各値を取得
        lat1, lon1, lat2, lon2 = bounds

        # 各辺の大きさ（緯度・経度の差）をメートルに変換
        # 南西→北西（緯度方向の距離）
        height_m = distance((lat1, lon1), (lat2, lon1)).meters

        # 南西→南東（経度方向の距離）
        width_m = distance((lat1, lon1), (lat1, lon2)).meters

        # 最大辺を基準に分割階層を決定
        square_length = max(height_m, width_m)

        # 2進法で分割していくので、LEVEL_LIMITは log2(max_length / MIN_GRID_WIDTH) の切り上げ
        if square_length > 0 and self.MIN_GRID_WIDTH > 0:
            Quadtree.LEVEL_LIMIT = int(math.ceil(math.log2(square_length / self.MIN_GRID_WIDTH)))

        logger.info(f"Quadtree initialized with LEVEL_LIMIT={Quadtree.LEVEL_LIMIT} (square_length={square_length:.2f} m)")


    def insert(self, point: Dict[str, float]) -> bool:
        return self.root.insert(point)


    def query(self, range: Tuple[float, float, float, float]) -> List[Dict[str, float]]:
        return self.root.query(range)


    def get_all_nodes(self) -> List[QuadtreeNode]:
        all_nodes = []
        def collect_nodes(node):
            all_nodes.append(node)
            for child in node.children:
                collect_nodes(child)
        collect_nodes(self.root)
        return all_nodes


    def get_nodes_at_level(self, level: int) -> List[QuadtreeNode]:
        return self.root.get_nodes_at_level(level)


    def get_leaf_nodes(self) -> List[QuadtreeNode]:
        return self.root.get_leaf_nodes()


    def get_empty_leaf_nodes(self) -> List[QuadtreeNode]:
        """ 点が一つも含まれていない葉ノードを取得 """
        leaf_nodes = self.get_leaf_nodes()
        return [node for node in leaf_nodes if len(node.points) == 0]


    def get_nonempty_leaf_nodes(self) -> List[QuadtreeNode]:
        """ ポイントデータが一つ以上格納されているリーフノードを取得 """
        return [node for node in self.get_leaf_nodes() if len(node.points) > 0]


    def get_leaf_node_by_point(self, lat: float, lon: float) -> Optional[QuadtreeNode]:
        """
        Quadtreeのルートから辿って、指定座標 (lat, lon) を含む葉ノードを返す
        """
        node = self.root
        point = {'lat': lat, 'lon': lon}
        while not node.is_leaf():
            found = False
            for child in node.children:
                if child.contains(point):
                    node = child
                    found = True
                    break
            if not found:
                return None  # 該当ノードなし
        return node


    def get_deepest_level(self) -> int:
        """
        四分木の最大深さを返す
        """
        leaf_nodes = self.get_leaf_nodes()
        return max(node.level for node in leaf_nodes) if leaf_nodes else 0


    def get_stats(self) -> Dict[str, Any]:
        """
        四分木の統計情報を返す
        - 総ノード数
        - 葉ノードの総数
        - データのない葉ノード数
        - データのある葉ノード数
        - 最大深さ
        - 最大深さのノード数
        - 各葉ノードの点数の最大値
        - 最も深いレベルのノードの一辺の長さ（メートル）
        """
        all_nodes = []
        def collect_nodes(node):
            all_nodes.append(node)
            for child in node.children:
                collect_nodes(child)
        collect_nodes(self.root)

        leaf_nodes = self.get_leaf_nodes()
        leaf_points_counts = [len(node.points) for node in leaf_nodes]
        deepest_level = self.get_deepest_level()

        # 最も深いレベルのノードの一辺の長さをメートルで求める
        deepest_nodes = [node for node in leaf_nodes if node.level == deepest_level]

        if deepest_nodes:
            node = deepest_nodes[0]
            lat1, lon1, lat2, lon2 = node.bounds
            height_m = distance((lat1, lon1), (lat2, lon1)).meters
            width_m = distance((lat1, lon1), (lat1, lon2)).meters
            deepest_node_size = max(height_m, width_m)
        else:
            deepest_node_size = 0.0

        return {
            "total_nodes": len(all_nodes),
            "leaf_nodes": len(leaf_nodes),
            "empty_leaf_nodes": len([node for node in leaf_nodes if len(node.points) == 0]),
            "nonempty_leaf_nodes": len([node for node in leaf_nodes if len(node.points) > 0]),
            "deepest_level": deepest_level,
            "deepest_nodes_count": len(deepest_nodes),
            "max_leaf_points": max(leaf_points_counts) if leaf_points_counts else 0,
            "deepest_node_size_m": deepest_node_size
        }


    def get_stats_text(self) -> str:
        """
        四分木の統計情報を整形して返す
        """
        stats = self.get_stats()
        table = [[k, f"{v:.3f}" if isinstance(v, float) else v] for k, v in stats.items()]
        headers = ["項目", "値"]
        return tabulate(table, headers=headers, numalign='right', tablefmt='github')


def get_latlon_delta(lat: float, lon: float, meters: float = 1.0) -> tuple[float, float]:
    """
    指定した緯度経度における、指定距離（メートル）のグリッドの緯度・経度の差分を返す。

    Args:
        lat (float): 基準となる緯度
        lon (float): 基準となる経度
        meters (float): 距離（メートル）

    Returns:
        (delta_lat, delta_lon): 指定距離だけ移動したときの緯度・経度の差分
    """
    # 緯度方向にmeters移動
    lat1 = distance(meters=meters).destination((lat, lon), bearing=0).latitude

    # 経度方向にmeters移動
    lon1 = distance(meters=meters).destination((lat, lon), bearing=90).longitude

    delta_lat = lat1 - lat
    delta_lon = lon1 - lon
    return delta_lat, delta_lon


def save_quadtree_image(
    quadtree: 'Quadtree',
    filename: str,
    figsize: Tuple[int, int] = (80, 80),
    dpi: int = 200,
    draw_points: bool = True,
    rect_color: str = "red",
    point_color: str = "blue",
    alpha: float = 0.1,  # 薄い色
):
    """ 四分木の可視化画像を保存する """

    leaf_nodes = quadtree.get_leaf_nodes()

    # ノード全体の座標範囲を取得
    lats = []
    lons = []
    for node in leaf_nodes:
        lat1, lon1, lat2, lon2 = node.bounds
        lats.extend([lat1, lat2])
        lons.extend([lon1, lon2])
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ノードごとに矩形を必ず描画
    for node in leaf_nodes:
        lat1, lon1, lat2, lon2 = node.bounds
        width = lon2 - lon1
        height = lat2 - lat1
        if node.points:
            # データがあるリーフノードは薄いグレーで塗りつぶす
            ax.add_patch(
                plt.Rectangle(
                    (lon1, lat1), width, height,
                    fill=True, facecolor=(0.7, 0.7, 0.7, alpha), edgecolor=rect_color, linewidth=0.5
                )
            )
        else:
            # データがないリーフノードは枠線のみ
            ax.add_patch(
                plt.Rectangle(
                    (lon1, lat1), width, height,
                    fill=False, facecolor='none', edgecolor=rect_color, linewidth=0.5
                )
            )

    # 点群の描画（draw_pointsがTrueのときのみ）
    if draw_points:
        all_points = []
        for node in leaf_nodes:
            for p in node.points:
                all_points.append((p['lon'], p['lat']))
        if all_points:
            xs, ys = zip(*all_points)
            ax.scatter(xs, ys, s=1, color=point_color, alpha=0.7, label="points")

    # 座標範囲を明示的に指定
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Quadtree Node Visualization")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def create_quadtree_from_df(df: pd.DataFrame) -> Quadtree:
    """ PandasのDataFrameから四分木を作成して返す """

    # データフレームから統計情報を取得
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    min_lon, max_lon = df['lon'].min(), df['lon'].max()

    # 中央座標を求める
    mid_lat = (min_lat + max_lat) / 2.0
    mid_lon = (min_lon + max_lon) / 2.0

    # ルートになる四分木の境界を正方形で設定
    square_size = max(max_lat - mid_lat, max_lon - mid_lon)
    bounds = (mid_lat - square_size, mid_lon - square_size, mid_lat + square_size, mid_lon + square_size)

    # 四分木を初期化
    quadtree = Quadtree(bounds)

    # データを四分木に挿入
    for _, row in df.iterrows():
        quadtree.insert({'lat': row['lat'], 'lon': row['lon'], 'depth': row['depth']})

    return quadtree


if __name__ == '__main__':



    def kriging_interpolate_leaf_nodes(quadtree, df, grid_size_m=10.0, points_per_tile=10):
        """
        四分木の葉ノードごとに、さらに小さなグリッドに分割してクリギング補間する
        """

        # 経度1度をメートル座標に変換
        METERS_PER_DEG_LAT = 111000.0

        # 緯度1度をメートル座標に変換
        mean_lat = df['lat'].mean()
        METERS_PER_DEG_LON = 111320.0 * np.cos(np.deg2rad(mean_lat))

        # 経度1度をメートル座標に変換
        METERS_PER_DEG_LAT = 111000.0

        # 緯度1度をメートル座標に変換
        # 緯度によって変わるので、df['lat']の平均値を使って経度1度あたりの距離を計算する
        mean_lat = df['lat'].mean()
        METERS_PER_DEG_LON = 111320.0 * np.cos(np.deg2rad(mean_lat))

        # 座標をメートル単位に変換
        x = df['lon'].values * METERS_PER_DEG_LON
        y = df['lat'].values * METERS_PER_DEG_LAT
        coords = np.column_stack([x, y])

        kdtree = cKDTree(coords)

        # 結果格納用のリスト
        interpolated_results = []

        # 葉ノードを取得
        leaf_nodes = quadtree.get_leaf_nodes()

        for node in leaf_nodes:
            if len(node.points) < 4:
                continue

            # ノードの範囲
            lat1, lon1, lat2, lon2 = node.bounds

            # ノード内のデータ抽出
            mask = (
                (df['lat'] >= lat1) & (df['lat'] < lat2) &
                (df['lon'] >= lon1) & (df['lon'] < lon2)
            )
            tile_df = df[mask]

            # KDTreeで半径10m以内の点を抽出して追加する
            """
            node_center_lat = (lat1 + lat2) / 2.0
            node_center_lon = (lon1 + lon2) / 2.0
            node_center_x = node_center_lon * METERS_PER_DEG_LON
            node_center_y = node_center_lat * METERS_PER_DEG_LAT
            indices = kdtree.query_ball_point([node_center_x, node_center_y], r=10.0)
            additional_points = df.iloc[indices]
            tile_df = pd.concat([tile_df, additional_points]).drop_duplicates().reset_index(drop=True)
            """

            if len(tile_df) < 4:
                continue

            # メートル座標に変換
            tile_x = tile_df['lon'].values * METERS_PER_DEG_LON
            tile_y = tile_df['lat'].values * METERS_PER_DEG_LAT
            tile_depths = tile_df['depth'].values

            # グリッド分割数を計算（ノードのサイズをメートルに変換して分割数を決定）
            width_m = distance((lat1, lon1), (lat1, lon2)).meters
            height_m = distance((lat1, lon1), (lat2, lon1)).meters
            n_x = max(2, int(width_m // grid_size_m))
            n_y = max(2, int(height_m // grid_size_m))

            grid_lon = np.linspace(lon1, lon2, n_x)
            grid_lat = np.linspace(lat1, lat2, n_y)
            grid_x = grid_lon * METERS_PER_DEG_LON
            grid_y = grid_lat * METERS_PER_DEG_LAT

            # クリギング補間
            try:
                OK = OrdinaryKriging(
                    tile_x, tile_y, tile_depths,
                    variogram_model='linear',
                    verbose=False,
                    enable_plotting=False
                )
                z, ss = OK.execute('grid', grid_x, grid_y)
                grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
                for xi, yi, zi in zip(grid_xx.flatten(), grid_yy.flatten(), z.flatten()):
                    interpolated_results.append({
                        'lon': xi / METERS_PER_DEG_LON,
                        'lat': yi / METERS_PER_DEG_LAT,
                        'depth': zi
                    })
            except Exception as e:
                logger.warning(f"Kriging failed for node at bounds {node.bounds}: {e}")
                continue

        # 元のデータフレームと補間結果を結合
        interpolated_df = pd.DataFrame(interpolated_results)
        combined_df = pd.concat([df, interpolated_df], ignore_index=True)
        return combined_df




    def main():

        data_filename = "ALL_depth_map_data_202510_dd_ol.csv"
        data_path = data_dir.joinpath(data_filename)
        if not data_path.exists():
            logger.error("File not found: %s" % data_path)
            return

        # read_csv()を使った方が軽いが、PandasのDataFrameで処理を統一する
        df = load_csv(data_path)
        if df is None:
            logger.error(f"データの読み込みに失敗しました: {data_path}")
            return

        # 四分木を作成
        Quadtree.MIN_GRID_WIDTH = 20.0  # 最小ノードの幅が20メートル以下になるように設定
        quadtree = create_quadtree_from_df(df)

        # 四分木の統計情報を表示
        logger.info(f"Initial Quadtree stats\n{quadtree.get_stats_text()}\n")

        # クリギング補間
        interpolated_df = kriging_interpolate_leaf_nodes(quadtree, df, grid_size_m=10.0)
        output_path = app_home.joinpath("static", "data", "processed_data.csv")
        save_csv(interpolated_df, output_path)
        logger.info(f"補間結果を保存しました: {output_path}")

    #
    # 実行
    #
    main()
