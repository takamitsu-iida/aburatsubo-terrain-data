#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""process_density_reduce.py

`make dd`後の点群について、半径nメートル以内に複数の点がある領域を1点に集約し、点数を減らす。

- 入力CSV: header無し (lat, lon, depth[, epoch])
- 出力CSV: header無し
- 集約方法: 半径eps(m)でDBSCANクラスタリング（連結成分）→クラスタ内で平均

注意:
- DBSCANは「距離eps以内の点が鎖状につながる」場合、クラスタ直径がepsを超えることがある。
  ただし「半径内に複数点がある領域をまとめる」という用途では、この挙動が自然なことが多い。
"""

SCRIPT_DESCRIPTION = "Reduce point density by aggregating nearby points within radius"

import argparse
import logging
import sys
from pathlib import Path

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, module=r"numpy.*", message=r"Signature b")

try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN
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


def reduce_density_by_radius(df: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if radius_m <= 0:
        return df

    has_epoch = "epoch" in df.columns

    # 数値化・欠損除去
    for col in ["lat", "lon", "depth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if has_epoch:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    df = df.dropna(subset=["lat", "lon", "depth"]).reset_index(drop=True)
    if df.empty:
        return df

    mean_lat = float(df["lat"].mean())
    meters_per_deg_lat = 111000.0
    meters_per_deg_lon = _meters_per_degree_lon(mean_lat)

    x = df["lon"].to_numpy(dtype=float) * meters_per_deg_lon
    y = df["lat"].to_numpy(dtype=float) * meters_per_deg_lat
    coords = np.column_stack([x, y])

    # DBSCANで半径eps以内を連結成分としてクラスタ化
    # min_samples=1 で全点にラベル付け
    clustering = DBSCAN(eps=float(radius_m), min_samples=1, algorithm="kd_tree")
    labels = clustering.fit_predict(coords)

    df = df.copy()
    df["_cluster"] = labels

    before = len(df)
    n_clusters = int(df["_cluster"].nunique())

    if has_epoch:
        reduced = (
            df.groupby("_cluster", as_index=False)
            .agg({"lat": "mean", "lon": "mean", "depth": "mean", "epoch": "mean"})
            .drop(columns=["_cluster"], errors="ignore")
        )
        reduced = reduced[["lat", "lon", "depth", "epoch"]]
    else:
        reduced = (
            df.groupby("_cluster", as_index=False)
            .agg({"lat": "mean", "lon": "mean", "depth": "mean"})
            .drop(columns=["_cluster"], errors="ignore")
        )
        reduced = reduced[["lat", "lon", "depth"]]

    after = len(reduced)
    logger.info(
        "Density reduced: radius_m=%.3f, before=%d, after=%d, clusters=%d, reduced_by=%d",
        float(radius_m),
        before,
        after,
        n_clusters,
        before - after,
    )

    return reduced


def main() -> int:
    parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
    parser.add_argument("--input", type=str, required=True, help="dataディレクトリ直下の入力CSVファイル名")
    parser.add_argument("--output", type=str, required=True, help="dataディレクトリ直下の出力CSVファイル名")
    parser.add_argument(
        "--radius-m",
        type=float,
        default=1.0,
        help="半径nメートル以内の点群を1点に集約する (default: 1.0)",
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

    logger.info("Input points: %d", len(df))
    reduced = reduce_density_by_radius(df, float(args.radius_m))

    save_csv(reduced, output_path)
    logger.info("Saved: %s (rows=%d)", str(output_path), len(reduced))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
