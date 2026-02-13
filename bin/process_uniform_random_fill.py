#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""process_uniform_random_fill.py

入力CSV (lat, lon, depth[, epoch]) の点群から、データが存在する矩形領域（min/max lat/lon）内に
一様乱数で点を生成し、実データから深さ（およびepochがある場合はepoch）を推定して補完点を作成する。

- 生成点: 矩形領域内で一様ランダム
- 推定: cKDTree による k 近傍 + IDW (Inverse Distance Weighting)

想定するCSVはヘッダー無し。既存のパイプライン（bin/load_save_csv.py）に合わせて保存もヘッダー無し。
"""

# スクリプトを引数無しで実行したときのヘルプに使うデスクリプション
SCRIPT_DESCRIPTION = "Generate uniform random points and interpolate depth using IDW"

import argparse
import logging
import sys
from pathlib import Path

# WSL1 固有の numpy 警告を抑制
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, module=r"numpy.*", message=r"Signature b")

try:
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    sys.exit(1)

try:
    from load_save_csv import load_csv, save_csv
except ImportError as e:
    logging.error(f"module import error: {e}")
    sys.exit(1)


app_path = Path(__file__)
app_name = app_path.stem
app_home = app_path.parent.joinpath("..").resolve()

data_dir = app_home.joinpath("data")

log_dir = app_home.joinpath("log")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)

file_handler = logging.FileHandler(log_dir.joinpath(f"{app_name}.log"), "a+", encoding="utf-8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


def _meters_per_degree_lon(mean_lat: float) -> float:
    return 111320.0 * np.cos(np.deg2rad(mean_lat))


def _build_kdtree(df: pd.DataFrame) -> tuple[cKDTree, np.ndarray, float, float]:
    mean_lat = float(df["lat"].mean())
    meters_per_deg_lat = 111000.0
    meters_per_deg_lon = _meters_per_degree_lon(mean_lat)

    x = df["lon"].to_numpy(dtype=float) * meters_per_deg_lon
    y = df["lat"].to_numpy(dtype=float) * meters_per_deg_lat
    coords = np.column_stack([x, y])
    tree = cKDTree(coords)
    return tree, coords, meters_per_deg_lat, meters_per_deg_lon


def _idw_from_neighbors(
    distances_m: np.ndarray,
    neighbor_indices: np.ndarray,
    values: np.ndarray,
    power: float,
) -> float:
    # distances_m, neighbor_indices: shape (k,)
    # values: shape (n,)
    if distances_m.ndim != 1:
        distances_m = distances_m.ravel()
    if neighbor_indices.ndim != 1:
        neighbor_indices = neighbor_indices.ravel()

    # ぴったり同一点がある場合
    zero_mask = distances_m == 0
    if np.any(zero_mask):
        return float(values[int(neighbor_indices[np.argmax(zero_mask)])])

    with np.errstate(divide="ignore"):
        weights = 1.0 / np.power(distances_m, power)

    wsum = float(np.sum(weights))
    if not np.isfinite(wsum) or wsum <= 0:
        return float("nan")

    v = values[neighbor_indices]
    return float(np.sum(weights * v) / wsum)


def generate_uniform_random_filled_points(
    df: pd.DataFrame,
    n_points: int,
    seed: int | None,
    k: int,
    power: float,
    max_distance_m: float | None,
    dense_radius_m: float | None,
    dense_max_neighbors: int | None,
    max_tries: int,
) -> pd.DataFrame:
    if n_points <= 0:
        return df.iloc[0:0].copy()

    if len(df) < 2:
        raise ValueError("入力データ点が少なすぎます（最低2点必要）")

    has_epoch = "epoch" in df.columns

    # 範囲（矩形領域）
    min_lat, max_lat = float(df["lat"].min()), float(df["lat"].max())
    min_lon, max_lon = float(df["lon"].min()), float(df["lon"].max())

    # 近傍探索用のKDTree
    tree, _, meters_per_deg_lat, meters_per_deg_lon = _build_kdtree(df)

    depth_values = df["depth"].to_numpy(dtype=float)
    epoch_values = df["epoch"].to_numpy(dtype=float) if has_epoch else None

    rng = np.random.default_rng(seed)

    k_eff = max(1, min(int(k), len(df)))

    dense_reject_enabled = (
        dense_radius_m is not None
        and dense_max_neighbors is not None
        and float(dense_radius_m) > 0
        and int(dense_max_neighbors) > 0
    )

    generated: list[dict[str, float]] = []
    tries = 0

    while len(generated) < n_points and tries < max_tries:
        remaining = n_points - len(generated)
        # まとめて候補生成してKDTreeに投げる（高速化）
        batch = int(min(max(remaining * 3, 256), 5000))

        cand_lat = rng.uniform(min_lat, max_lat, size=batch)
        cand_lon = rng.uniform(min_lon, max_lon, size=batch)

        cand_x = cand_lon * meters_per_deg_lon
        cand_y = cand_lat * meters_per_deg_lat
        cand_coords = np.column_stack([cand_x, cand_y])

        dense_counts = None
        if dense_reject_enabled:
            # 各候補点の周囲(dense_radius_m)に存在する実データ点数を数える
            # 候補点はKDTreeに含まれないため、self除外などは不要
            neighbors = tree.query_ball_point(cand_coords, r=float(dense_radius_m))
            dense_counts = np.fromiter((len(n) for n in neighbors), dtype=int)

        distances, indices = tree.query(cand_coords, k=k_eff, workers=-1)
        if k_eff == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        for i in range(batch):
            tries += 1
            if tries >= max_tries:
                break

            d0 = float(distances[i, 0])
            if max_distance_m is not None and d0 > max_distance_m:
                continue

            if dense_reject_enabled and dense_counts is not None:
                if int(dense_counts[i]) >= int(dense_max_neighbors):
                    continue

            depth = _idw_from_neighbors(distances[i], indices[i].astype(int), depth_values, power)
            if not np.isfinite(depth):
                continue

            row: dict[str, float] = {
                "lat": float(cand_lat[i]),
                "lon": float(cand_lon[i]),
                "depth": float(depth),
            }

            if has_epoch and epoch_values is not None:
                epoch = _idw_from_neighbors(distances[i], indices[i].astype(int), epoch_values, power)
                if np.isfinite(epoch):
                    row["epoch"] = float(epoch)
                else:
                    # epoch推定ができない場合は最近傍のepoch
                    row["epoch"] = float(epoch_values[int(indices[i, 0])])

            generated.append(row)
            if len(generated) >= n_points:
                break

    if len(generated) < n_points:
        logger.warning(
            "要求点数に到達しませんでした: generated=%d / requested=%d (max_tries=%d, max_distance_m=%s)",
            len(generated),
            n_points,
            max_tries,
            str(max_distance_m),
        )

    out_df = pd.DataFrame(generated)

    # 列順を入力に合わせる
    if has_epoch:
        out_df = out_df[["lat", "lon", "depth", "epoch"]]
    else:
        out_df = out_df[["lat", "lon", "depth"]]

    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
    parser.add_argument("--input", type=str, required=True, help="dataディレクトリ直下の入力CSVファイル名")
    parser.add_argument("--output", type=str, required=True, help="dataディレクトリ直下の出力CSVファイル名")

    parser.add_argument("--n", type=int, default=5000, help="生成する補完点の数 (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード (default: 42)")

    parser.add_argument("--k", type=int, default=8, help="IDWで使う近傍点数 (default: 8)")
    parser.add_argument("--power", type=float, default=2.0, help="IDWの距離べき乗 (default: 2.0)")

    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=None,
        help="最近傍点がこの距離(m)より遠い候補点は捨てる（未指定なら制限なし）",
    )

    parser.add_argument(
        "--dense-radius-m",
        type=float,
        default=10.0,
        help="半径(m)以内の近傍点数で密集判定する半径 (default: 10.0)",
    )
    parser.add_argument(
        "--dense-max-neighbors",
        type=int,
        default=200,
        help="半径内の近傍点数がこの値以上なら候補点を捨てる (default: 200). 0以下で無効化",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=200000,
        help="候補点生成の最大試行回数 (default: 200000)",
    )

    parser.add_argument(
        "--only-generated",
        action="store_true",
        help="生成点だけを出力する（指定しない場合は元データに追記して出力）",
    )

    args = parser.parse_args()

    input_path = Path(data_dir, args.input)
    if not input_path.exists():
        logger.error(f"入力ファイルが存在しません: {input_path}")
        return 2

    output_path = Path(data_dir, args.output)

    df = load_csv(input_path)
    if df is None or df.empty:
        logger.error(f"CSVファイルの読み込みに失敗しました: {input_path}")
        return 2

    # 数値化
    for col in ["lat", "lon", "depth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    df = df.dropna(subset=["lat", "lon", "depth"]).reset_index(drop=True)

    logger.info("Input points: %d", len(df))
    logger.info(
        "Bounds: lat=[%.6f, %.6f], lon=[%.6f, %.6f]",
        float(df["lat"].min()),
        float(df["lat"].max()),
        float(df["lon"].min()),
        float(df["lon"].max()),
    )

    gen_df = generate_uniform_random_filled_points(
        df=df,
        n_points=int(args.n),
        seed=int(args.seed) if args.seed is not None else None,
        k=int(args.k),
        power=float(args.power),
        max_distance_m=float(args.max_distance_m) if args.max_distance_m is not None else None,
        dense_radius_m=float(args.dense_radius_m) if args.dense_radius_m is not None else None,
        dense_max_neighbors=int(args.dense_max_neighbors) if args.dense_max_neighbors is not None else None,
        max_tries=int(args.max_tries),
    )

    if args.only_generated:
        out_df = gen_df
    else:
        out_df = pd.concat([df, gen_df], ignore_index=True)

    save_csv(out_df, output_path)
    logger.info("Saved: %s (rows=%d)", str(output_path), len(out_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
