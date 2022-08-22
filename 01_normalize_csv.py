#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""pythonスクリプト
"""

__author__ = "takamitsu-iida"
__version__ = "0.0"
__date__ = "2022/08/19"

#
# 標準ライブラリのインポート
#
import logging
import os
import sys


def here(path=''):
  """相対パスを絶対パスに変換して返却します"""
  return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

# アプリケーションのホームディレクトリは一つ上
app_home = here(".")

# 自身の名前から拡張子を除いてプログラム名を得る
app_name = os.path.splitext(os.path.basename(__file__))[0]

# ディレクトリ
data_dir = os.path.join(app_home, "data")

# libフォルダにおいたpythonスクリプトをインポートできるようにするための処理
# このファイルの位置から一つ
if not here("./lib") in sys.path:
  sys.path.append(here("./lib"))

#
# ログ設定
#

# ログファイルの名前
log_file = app_name + ".log"

# ログファイルを置くディレクトリ
log_dir = os.path.join(app_home, "log")
try:
  if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
except OSError:
  pass

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

# default setting
logging.basicConfig()

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
  file_handler = logging.FileHandler(os.path.join(log_dir, log_file), 'a+')
  file_handler.setFormatter(formatter)
  file_handler.setLevel(logging.INFO)
  logger.addHandler(file_handler)

#
# 外部ライブラリのインポート
#

#
# ここからスクリプト
#
if __name__ == '__main__':

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



    def main():
        """メイン関数

        Returns:
        int -- 正常終了は0、異常時はそれ以外を返却
        """

        filename = os.path.join(data_dir, "bathymetry_data.csv")
        read_file(filename=filename, callback=line_callback)


        return 0

    # 実行
    sys.exit(main())