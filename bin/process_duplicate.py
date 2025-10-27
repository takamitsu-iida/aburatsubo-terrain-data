#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeeperのGPSデータを入力として受け取り、重複する座標データを削除した新しいCSVファイルを作成するスクリプトです。


"""

# スクリプトを引数無しで実行したときのヘルプに使うデスクリプション
SCRIPT_DESCRIPTION = 'drop duplicate coordinates from Deeper GPS data'

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

#
# 外部ライブラリのインポート
#
try:
    import pandas as pd
except ImportError as e:
    logging.error("pandas module is not installed. Please install pandas to use this script.")
    sys.exit(1)


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
# ここからスクリプト
#

def read_file(filename, callback):
    try:
        with open(filename) as f:
            for line in f:
                line = line.rstrip()
                callback(line)
    except IOError as e:
        logger.exception(e)


def line_callback(line):
    print(line)


def load_csv(data_path):
    try:
        return pd.read_csv(data_path)
    except:
        return None


def process_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    (lat, lon)が重複した行については、depthの平均値を取り、(lat, lon, depth)で1行だけ残す。
    """
    before = len(df)
    # groupbyで(lat, lon)ごとにdepthの平均を計算
    df_uniq = df.groupby(['lat', 'lon'], as_index=False)['depth'].mean()
    after = len(df_uniq)
    removed = before - after
    logger.info(f"重複排除前: {before}件, 排除後: {after}件, 排除数: {removed}件")
    return df_uniq


if __name__ == '__main__':

    def main() -> None:
        # 引数処理
        parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
        parser.add_argument('--input', type=str, required=True, help='dataディレクトリ直下の入力CSVファイル名')
        parser.add_argument('--output', type=str, required=True, help='dataディレクトリ直下の出力CSVファイル名')
        args = parser.parse_args()

        # 引数が何も指定されていない場合はhelpを表示して終了
        if not any(vars(args).values()):
            parser.print_help()
            return

        # 保存先のファイル名が指定されていない場合は終了
        if not args.output:
            logger.error("出力ファイル名が指定されていません。")
            return

        # 入力ファイルのパス
        input_file_path = Path(data_dir, args.input)
        if not input_file_path.exists():
            logger.error(f"入力ファイルが存在しません: {input_file_path}")
            return

        # 出力ファイルの名前とパス
        output_filename = args.output
        output_file_path = Path(data_dir, output_filename)

        # 入力CSVファイルをPandasのデータフレームとして読み込む
        try:
            # このCSVファイルには列名がないので、header=Noneを指定して読み込む
            df = pd.read_csv(input_file_path, header=None)

            # データフレームに列名を定義
            df.columns = ["lat", "lon", "depth", "time"]

            # 時刻の列は不要なので削除する
            del df["time"]
        except Exception as e:
            logger.error(f"CSVファイルの読み込みに失敗しました：{str(e)}")
            return

        # 読み込んだデータのサマリを表示する
        # to_markdown()
        # を使うにはtabulate moduleが必要

        # head(3)
        logger.info(f"head(3)\n{df.head(3).to_markdown()}\n")

        # tail(3)
        logger.info(f"tail(3)\n{df.tail(3).to_markdown()}\n")

        # describe()
        logger.info(f"describe() --- 削除前\n{df.describe().to_markdown()}\n")

        # 重複した座標のデータを削除する
        df = process_duplicates(df)

        logger.info(f"describe() --- 削除後\n{df.describe().to_markdown()}\n")

        # 重複削除後のデータをCSVファイルとして保存
        try:
            df.to_csv(output_file_path, index=False)
            logger.info(f"重複削除後のデータを保存しました: {output_filename}")
        except Exception as e:
            logger.error(f"CSVファイルの保存に失敗しました：{str(e)}")

    #
    # 実行
    #
    main()
