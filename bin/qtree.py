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
    import matplotlib.pyplot as plt

    # GPS座標から距離を計算
    from geopy.distance import distance

    # データを整形して表示
    from tabulate import tabulate
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
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


if __name__ == '__main__':

    def misc():
        from geopy.distance import great_circle, distance

        # 座標を (緯度, 経度) のタプルで定義
        tokyo = (35.681236, 139.767125)
        osaka = (34.702485, 135.495574)

        # great_circle (ハバーサインよりも高精度な測地線距離に近い計算) を使用して距離を計算
        d = great_circle(tokyo, osaka)

        print(f"geopyによる距離: {d.km:.2f} km")

        delta_lat, delta_lon = get_latlon_delta(tokyo[0], tokyo[1], meters=5)
        print(f"5m四方の緯度差: {delta_lat:.8f}, 経度差: {delta_lon:.8f}")


    def main():

        data_filename = "ALL_depth_map_data_202502_dd_ol.csv"

        data_path = data_dir.joinpath(data_filename)
        if not data_path.exists():
            logger.error("File not found: %s" % data_path)
            return

        # read_csv()の方が速いが、PandasのDataFrameを使いたいのでそちらを使用
        # data, data_stats = read_csv(data_path)

        # CSVファイルをPandasのデータフレームとして読み込む
        try:
            df = pd.read_csv(data_path)
            logger.info(f"describe() --- 入力データ\n{df.describe().to_markdown()}\n")
        except Exception as e:
            logger.error(f"CSVファイルの読み込みに失敗しました：{str(e)}")
            return

        # データフレームから統計情報を取得
        data_stats = {
            'lat': {
                'min': df['lat'].min(),
                'max': df['lat'].max()
            },
            'lon': {
                'min': df['lon'].min(),
                'max': df['lon'].max()
            },
            'depth': {
                'min': df['depth'].min(),
                'max': df['depth'].max()
            }
        }

        # 中央座標を求める
        mid_lat = (data_stats['lat']['min'] + data_stats['lat']['max']) / 2.0
        mid_lon = (data_stats['lon']['min'] + data_stats['lon']['max']) / 2.0

        # ルートになる四分木の境界を正方形で設定
        square_size = max(
            mid_lat - data_stats['lat']['min'],
            data_stats['lat']['max'] - mid_lat,
            mid_lon - data_stats['lon']['min'],
            data_stats['lon']['max'] - mid_lon
        )
        bounds = (mid_lat - square_size, mid_lon - square_size, mid_lat + square_size, mid_lon + square_size)

        # 四分木を初期化
        quadtree = Quadtree(bounds)

        """
        # データを四分木に挿入
        for row in data:
            lat, lon, depth = row
            point = {'lat': lat, 'lon': lon, 'depth': depth}
            quadtree.insert(point)
        """

        # データを四分木に挿入
        for _, row in df.iterrows():
            point = {'lat': row['lat'], 'lon': row['lon'], 'depth': row['depth']}
            quadtree.insert(point)

        # 四分木の統計情報を表示する
        logger.info(f"Initial Quadtree stats\n{quadtree.get_stats_text()}\n")

        #
        # データ集約
        #

        # 最も深いノード、および一つ上の階層のノードについては、そのノード内のポイントの平均値に置き換える
        deepest_level = quadtree.get_deepest_level()
        nodes = [node for node in quadtree.get_leaf_nodes() if node.level >= (deepest_level - 1) and len(node.points) > 1]
        for node in nodes:
            avg_point = node.average()
            node.points = [avg_point]

        # 四分木の統計情報を表示する
        logger.info(f"Aggregated Quadtree stats\n{quadtree.get_stats_text()}\n")

        #
        # データを増やす
        # ポイントのない葉ノードについては、隣接ノードに値があればその平均値で埋める
        #
        extended_count = 0
        empty_leaf_nodes = quadtree.get_empty_leaf_nodes()
        for node in empty_leaf_nodes:

            if node.level < deepest_level - 3:
                # 深さが浅いノードは無視
                continue

            # 隣接ノードに格納されているポイントを収集
            neighbor_points = []

            # ノードの中心座標
            mid_lat, mid_lon = node.center

            # 8方向の隣接ノードをチェック
            directions = [
                (mid_lat - node.lat_length, mid_lon - node.lon_length),  # NW
                (mid_lat - node.lat_length, mid_lon),                    # N
                (mid_lat - node.lat_length, mid_lon + node.lon_length),  # NE
                (mid_lat, mid_lon - node.lon_length),                    # W
                (mid_lat, mid_lon + node.lon_length),                    # E
                (mid_lat + node.lat_length, mid_lon - node.lon_length),  # SW
                (mid_lat + node.lat_length, mid_lon),                    # S
                (mid_lat + node.lat_length, mid_lon + node.lon_length)   # SE
            ]

            # 8方向の隣接ノードを取得
            for d_lat, d_lon in directions:
                neighbor_node = quadtree.get_leaf_node_by_point(d_lat, d_lon)
                if neighbor_node and len(neighbor_node.points) > 0:
                    neighbor_points.extend(neighbor_node.points)

            # 平均値で埋める
            if len(neighbor_points) > 3:
                avg_lat = sum(p['lat'] for p in neighbor_points) / len(neighbor_points)
                avg_lon = sum(p['lon'] for p in neighbor_points) / len(neighbor_points)
                avg_depth = sum(p['depth'] for p in neighbor_points) / len(neighbor_points)
                node.points.append({'lat': avg_lat, 'lon': avg_lon, 'depth': avg_depth})
                extended_count += 1

        logger.info(f"Extended {extended_count} empty leaf nodes with neighbor averages")

        # 拡張した四分木の統計情報を表示する
        logger.info(f"Extended Quadtree stats\n{quadtree.get_stats_text()}\n")

        # 点群をファイルに保存する
        points = []
        for node in quadtree.get_leaf_nodes():
            points.extend(node.points)
        logger.info(f"Points count: {len(points)}")

        # CSVファイルはJavaScriptのディレクトリに保存
        output_path = app_home.joinpath("static/data/aggregated_data.csv")
        with open(output_path, 'w') as f:
            f.write("lat,lon,depth\n")
            for p in points:
                f.write(f"{p['lat']},{p['lon']},{p['depth']}\n")
        logger.info(f"Points data saved to: {output_path}")

        # 四分木の可視化画像を保存する
        output_image_filename = f"{data_path.stem}_qtree.png"
        output_image_path = image_dir.joinpath(output_image_filename)
        #save_quadtree_image(quadtree=quadtree, filename=output_image_path, draw_points=False)

    #
    # 実行
    #

    main()
