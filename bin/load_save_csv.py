#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeeperのGPSデータのCSVファイルを入出力するスクリプトです。


"""

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
# ここからスクリプト
#

def read_file_lines(file_path: Path, callback: callable) -> list[str] | None:
    lines = []
    try:
        with file_path.open() as f:
            for line in f:
                line = line.rstrip()
                lines.append(line)
                if callback:
                    callback(line)
    except IOError as e:
        logging.exception(f"ファイルの読み込みに失敗しました: {file_path} - {str(e)}")
        return None
    return lines


def line_callback(line: str) -> None:
    print(line)


def load_csv(input_data_path: Path) -> pd.DataFrame | None:
    # 入力CSVファイルをPandasのデータフレームとして読み込む
    try:
        # CSVファイルに列名がないので、header=Noneを指定して読み込む
        df = pd.read_csv(input_data_path, header=None)

        # データフレームに列名を定義
        if df.shape[1] == 3:
            df.columns = ["lat", "lon", "depth"]
        elif df.shape[1] == 4:
            df.columns = ["lat", "lon", "depth", "time"]
            del df["time"]
        else:
            logger.error(f"CSVファイルの列数が3または4ではありません（{df.shape[1]}列）")
            return
    except Exception as e:
        logger.error(f"CSVファイルの読み込みに失敗しました：{str(e)}")
        return



def save_csv(df: pd.DataFrame, output_filename: str) -> None:
    #
    # データフレームをCSVファイルに保存する
    #
    try:
        df.to_csv(output_file_path, index=False, header=False)
        logger.info(f"外れ値を除去したデータフレームを保存しました: {output_filename}")
    except Exception as e:
        logger.error(f"CSVファイルの保存に失敗しました：{str(e)}")





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
            df.to_csv(output_file_path, index=False, header=False)
            logger.info(f"重複削除後のデータを保存しました: {output_filename}")
        except Exception as e:
            logger.error(f"CSVファイルの保存に失敗しました：{str(e)}")

    #
    # 実行
    #
    main()
