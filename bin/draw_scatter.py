#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSVファイルを読み込んで、散布図を描画するスクリプト

"""

# スクリプトを引数無しで実行したときのヘルプに使うデスクリプション
SCRIPT_DESCRIPTION = 'draw scatter plot from CSV data'

#
# 標準ライブラリのインポート
#
import argparse
import logging
import sys

from pathlib import Path

# WSL1 固有の numpy 警告を抑制
# https://github.com/numpy/numpy/issues/18900
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, module=r"numpy.*", message=r"Signature b")

# このファイルへのPathオブジェクト
app_path = Path(__file__)

# このファイルがあるディレクトリ
app_dir = app_path.parent

# このファイルの名前から拡張子を除いてプログラム名を得る
app_name = app_path.stem

# アプリケーションのホームディレクトリはこのファイルからみて一つ上
app_home = app_path.parent.joinpath('..').resolve()

# データを格納しているディレクトリ
data_dir = app_home.joinpath("data")

# 画像を格納しているディレクトリ
image_dir = app_home.joinpath("img")
image_dir.mkdir(exist_ok=True)

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
    file_handler = logging.FileHandler(log_dir.joinpath(log_file), 'a+')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

#
# 外部ライブラリのインポート
#
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.exception(e)
    sys.exit(1)

#
# ここからスクリプト
#

# 画像サイズ（単位：インチ）
FIG_SIZE = (9, 6)

def save_scatter(df: pd.DataFrame, title="", output_path="") -> None:
    if not output_path:
        return

    df_plt = df.plot.scatter(x="lon", y="lat", title=title, grid=True, figsize=FIG_SIZE, s=0.5)
    df_plt.set_xlabel("lon")
    df_plt.set_ylabel("lat")
    plt.savefig(output_path)
    plt.clf()


if __name__ == '__main__':

    def main() -> None:
        # 引数処理
        parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
        parser.add_argument('--input', type=str, required=True, help='dataディレクトリ直下のCSVファイル名')
        parser.add_argument('--title', type=str, required=True, help='グラフのタイトル')
        args = parser.parse_args()

        # 引数が何も指定されていない場合はhelpを表示して終了
        if not any(vars(args).values()):
            parser.print_help()
            return

        input_file_path = Path(data_dir, args.input)
        if not input_file_path.exists():
            logger.error(f"入力ファイルが存在しません: {input_file_path}")
            return

        # グラフのタイトル
        graph_title = args.title

        # 拡張子を除いた名前
        input_basename = input_file_path.stem

        # 出力するイメージファイル名
        output_image_name = f"{input_basename}.png"

        # 出力するイメージファイルのフルパス
        output_image_path = image_dir.joinpath(output_image_name)

        # CSVファイルをPandasのデータフレームとして読み込む
        try:
            df = pd.read_csv(input_file_path)
            logger.info(f"describe()\n{df.describe().to_markdown()}")
        except Exception as e:
            logger.error(f"CSVファイルの読み込みに失敗しました：{str(e)}")
            return

        # 散布図を保存
        save_scatter(df, title=graph_title, output_path=output_image_path)

    #
    # 実行
    #
    main()