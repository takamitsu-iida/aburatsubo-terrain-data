#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeeperのGPSデータ(lat, lon, depth, epoch)のうち、
epochをドロップした新しいCSVファイルを作成するスクリプトです。

"""

# スクリプトを引数無しで実行したときのヘルプに使うデスクリプション
SCRIPT_DESCRIPTION = 'drop epoch column from Deeper GPS data'

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

#
# ローカルファイルからインポート
#
try:
    from load_save_csv import load_csv, save_csv
except ImportError as e:
    logging.error("load_save_csv module is not found. Please make sure load_save_csv.py is in the same directory.")
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
file_handler = logging.FileHandler(log_dir.joinpath(log_file), 'a+', encoding='utf-8')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

#
# ここからスクリプト
#

def drop_epoch(df: pd.DataFrame) -> pd.DataFrame:
    df_dropped = df.drop(columns=['epoch'], errors='ignore')
    return df_dropped



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
        df = load_csv(input_file_path)
        if df is None:
            logger.error(f"CSVファイルの読み込みに失敗しました: {input_file_path}")
            return

        # 読み込んだデータのサマリを表示する
        # to_markdown()
        # を使うには tabulate が必要なのでインストールしておくこと(requirements.txtに記載済み)

        # head(3)
        logger.info(f"head(3)\n{df.head(3).to_markdown()}\n")

        # tail(3)
        logger.info(f"tail(3)\n{df.tail(3).to_markdown()}\n")

        # describe()
        logger.info(f"describe() --- 削除前\n{df.describe().to_markdown()}\n")

        if 'epoch' in df.columns:
            # 最小epoch値の取得と変換
            min_epoch = df['epoch'].min()
            min_datetime = pd.to_datetime(min_epoch, unit='ms').tz_localize('UTC').tz_convert('Asia/Tokyo')

            # 最大epoch値の取得と変換
            max_epoch = df['epoch'].max()
            max_datetime = pd.to_datetime(max_epoch, unit='ms').tz_localize('UTC').tz_convert('Asia/Tokyo')

            # 最も古いepoch値を時刻形式で表示
            logger.info(f"Oldest epoch (JST): {min_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")

            # 最も新しいepoch値を時刻形式で表示
            logger.info(f"Newest epoch (JST): {max_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")

            # epoch列を削除する
            df = drop_epoch(df)

            logger.info(f"describe() --- 削除後\n{df.describe().to_markdown()}\n")

            # epoch列を削除したデータをCSVファイルとして保存
            save_csv(df, output_file_path)
            logger.info(f"epoch列を削除したデータを保存しました: {output_filename}")

    #
    # 実行
    #
    main()
