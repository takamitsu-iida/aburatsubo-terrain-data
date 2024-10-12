#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""rtreeを動かす練習スクリプト
"""

__author__ = "takamitsu-iida"
__version__ = "0.1"
__date__ = "2024/10/10"

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
app_home = here("..")

# 自身の名前から拡張子を除いてプログラム名を得る
app_name = os.path.splitext(os.path.basename(__file__))[0]

# ディレクトリ
data_dir = os.path.join(app_home, "data")
img_dir = os.path.join(app_home, "img")

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
try:
    from rtree import index

except ImportError as e:
    logger.exception(e)
    sys.exit(1)

#
# ここからスクリプト
#


if __name__ == '__main__':

    def main():
        """メイン関数
        Returns:
        int -- 正常終了は0、異常時はそれ以外を返却
        """

        # rtreeのインデックスを作成
        idx = index.Index()

        # bounding boxを追加
        left, bottom, right, top = (0.0, 0.0, 1.0, 1.0)

        # 0番目の要素を追加
        idx.insert(0, (left, bottom, right, top))

        #
        # 交差する要素を取得
        #

        # 期待値はインデックス0が該当するので[0]
        result = list(idx.intersection((0.5, 0.5, 1.5, 1.5)))
        print('list(idx.intersection((0.5, 0.5, 1.5, 1.5)))')
        print(result)

        # 期待値は空リスト
        result = list(idx.intersection((1.0000001, 1.0000001, 2.0, 2.0)))
        print('list(idx.intersection((1.0000001, 1.0000001, 2.0, 2.0))')
        print(result)

        # もう一つ追加してから、
        idx.insert(1, (left, bottom, right, top))

        # 近傍を取得
        result = list(idx.nearest((1.0000001, 1.0000001, 2.0, 2.0), 1))
        print('list(idx.nearest((1.0000001, 1.0000001, 2.0, 2.0), 1)')
        print(result);

        #
        # データベースのように使う
        #

        # obj=42を紐づけて追加
        id = 1
        idx.insert(id=id, coordinates=(left, bottom, right, top), obj=42)

        objs = [n.object for n in idx.intersection((left, bottom, right, top), objects=True)]
        print('[n.object for n in idx.intersection((left, bottom, right, top), objects=True)]')
        print(objs)



        return 0

    # 実行
    sys.exit(main())
