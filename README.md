# aburatsubo-terrain-data

<br>

2025/05/02 depth_map_data.csv (processed)

https://takamitsu-iida.github.io/aburatsubo-terrain-data/data/ALL_depth_map_data_202502_dd_ol_ip_mf.csv

<br>

3D Visualization

https://takamitsu-iida.github.io/aburatsubo-terrain-data/index-processed-data.html



<br><br>

## 環境構築

binディレクトリにスクリプトを配置。

testsディレクトリにテスト用スクリプトを配置。

binディレクトリにあるスクリプトを外部からimportするにはPYTHONPATHの設定が必要。

vscodeは .env ファイルがあればそれを読み取るので、以下のように設定する。

```bash
PYTHONPATH=bin
```

venvで仮想環境を整える。

```bash
python3 -m venv .venv
pip install --upgrade pip
pip install -r requirements.txt
```

<br><br>

## CSVデータ

Deeperの地図アプリからCSV形式でデータをダウンロードする。

https://maps.fishdeeper.com/ja-jp

~~ファイル名は共通で `bathymetry_data.csv` となっている。~~

<br>

> [!NOTE]
>
> 2024年8月追記。
>
> 仕様が変わって、ハンバーガーメニューからダウンロードを選択すると、depth_map_data.csvというファイルが取得できる。
>
> 仕様変更前にダウンロードしたファイルは `./data/bathymetry_data.csv` として残してある。

<br>

> [!NOTE]
>
> 2024年8月追記。
>
> 2017年9月よりも古いデータは参考にならないと判断してクラウドから削除。
>
> なぜか重複して登録されているデータもクラウドから削除。

<br>

> [!NOTE]
>
> 水深データの一括ダウンロードは大変重たい処理のようで、長い時間かかるため失敗することが多い。
> 感覚的には10回に1回くらいしか成功しない。
> 釣行ごとの水深データを落として結合したほうがいいかもしれない。

<br>

## 自分用作業メモ

1. main.pyで `ALL_depth_map_data_202408.csv` を読み込んで `./static/data/depth_map_data.csv` を出力する

main.pyは主に重複データの集約と、明らかな異常値の排除を行う。

TODO: 入力に使うファイルはスクリプトに埋め込んでしまっているので要改善。

TODO: 出力するファイルの場所は '/static/data/' にしないとJavaScript側が読み込めないので、出力先を変更する

2. `./static/data/depth_map_data.csv` をJavaScriptでポイントクラウドとして可視化して、目視で異常値と考えられる点を削除する。

ファイル名は `depth_map_data_edited.csv` としてブラウザからダウンロードされるので、それを手作業で移動する。

次の手順はpythonスクリプトなので `./data/` 配下に移動する。

3. 四分木スクリプトで `depth_map_data_edited.csv` を読み込んで `depth_map_data_final.csv` を出力する

出力先は `./static/data` のは以下。

<br>

## CSVファイルの概要

サイズは約8MBで約22万行。

各行は `lat, lon, depth, unix_time_in_msec` の情報で構成されている。

先頭の3行

```text
35.162872,139.61423,2.747,1659737621643
35.16289,139.61423,2.831,1659737623625
35.162903,139.61421,2.987,1659737624655
```

緯度(lat)の分解能は小数点以下6桁。経度(lon)の分解能は小数点以下5桁。
概ね1m程度の分解能になっている。

1行目のUNIX時間1659737621643は2022年8月6日7時13分41.643秒を表している。
2行目のUNIX時間との差は約2秒。
ということはDeeperは2秒に一度、スキャンしていることになる。

CSVファイルの最後の3行

```text
35.16161,139.60029,32.5,0
35.16162,139.60031,33.05,0
35.16162,139.60031,32.39,0
```

古いデータはUNIX時間が欠落していて0になっている。
概ね2017年よりも古いデータはUNIX時間が欠落している模様。

緯度方向の分解能も低くなっている。これは初代Deeperと現行機種との差かも。

古いデータはdropしてもいいけど、今回はそのまま利用する。


CSVファイルをpandasで読み取った時点での統計値。

|       |             lat |             lon |        depth |
|:------|----------------:|----------------:|-------------:|
| count | 222642          | 222642          | 222642       |
| mean  |     35.1635     |    139.607      |     17.8339  |
| std   |      0.00147178 |      0.00355166 |      9.38083 |
| min   |     35.1572     |    139.554      |      1.426   |
| 25%   |     35.1625     |    139.604      |     10.158   |
| 50%   |     35.1637     |    139.607      |     15.8     |
| 75%   |     35.1645     |    139.609      |     25.71    |
| max   |     35.1797     |    139.622      |     46.984   |

CSVから読み取った状態での散布図。左上に異常値が存在する。

![original data](https://takamitsu-iida.github.io/aburatsubo-terrain-data/img/scatter_01.png)

重複した座標も多くあり、pandasで分析すると75,402件が重複している。
重複した部分はgroupbyで集約して水深の平均値を算出し、重複行を削除したあと、集約したものを加えている。

![drop duplicated](https://takamitsu-iida.github.io/aburatsubo-terrain-data/img/scatter_02.png)

異常値として検出されたデータ。
1,039件と思いのほか多くのデータが異常値として弾かれた。

![outlier data](https://takamitsu-iida.github.io/aburatsubo-terrain-data/img/scatter_03.png)

異常値を除いたデータ。

![drop outlier](https://takamitsu-iida.github.io/aburatsubo-terrain-data/img/scatter_04.png)

重複を削除、異常値を削除した状態でのdescribe()はこの通り。

|       |             lat |             lon |        depth |
|:------|----------------:|----------------:|-------------:|
| count | 107440          | 107440          | 107440       |
| mean  |     35.1641     |    139.608      |     16.4776  |
| std   |      0.00237647 |      0.00435879 |      9.62367 |
| min   |     35.1572     |    139.599      |      1.082   |
| 25%   |     35.1628     |    139.604      |      8.409   |
| 50%   |     35.164      |    139.608      |     15.09    |
| 75%   |     35.1649     |    139.61       |     21.928   |
| max   |     35.1737     |    139.622      |     47.539   |


<br><br>

## 座標系メモ

latとlon、どっちが緯度でどっちが経度かすぐにわからなくなる。

<dl>
<dt>latitude</dt> <dd>緯度</dd>
<dt>longitude</dt> <dd>経度</dd>
<dt>地球の半径</dt> <dd>6378.137 km 測地測量の基準としての半径</dd>
</dl>

<br><br>

# 参考文献（データ処理）

[Python で曲面近似（サーフェスフィッティング）する](https://chuckischarles.hatenablog.com/entry/2020/02/06/081238)

[カーブフィッティング手法 scipy.optimize.curve_fit の使い方を理解する](https://qiita.com/maskot1977/items/e4f5f71200180865986e)

[The Nature of Geographic Information](https://www.e-education.psu.edu/natureofgeoinfo/c7_p9.html)

<br><br>

# 実装メモ

Go言語の方が書きやすいものの、データ加工処理の容易さを考慮してPythonで処理。

<br>

## 重複削除のやり方

groupbyで集約しつつ、水深の平均を計算してそれに置き換える。

元データから重複行をすべて削除し、groupbyで計算したものを加える。

```python
    def process_duplicated(df: pd.DataFrame):
        dfx = df[["lat", "lon", "depth"]]
        dfx_uniq = df.drop_duplicates(subset=["lat", "lon"], keep=False)
        dfx_duplicated = dfx.groupby(["lat", "lon"])["depth"].mean().reset_index()
        df = pd.concat([dfx_uniq, dfx_duplicated]).reset_index(drop=True)
        return df
```

<br>

## 外れ値検出

Local Outlier Factor（局所外れ値因子法）を利用する。

LOFの考え方

- 自分を中心に円を書き、k個の近傍点が入るように円を拡大していく。

- 円に入ったk個の点それぞれにつき、同じように円を書き、k個の近傍点が入るように円を拡大していく

- 自分の円を含め、合計1+k個の円ができる

- その円の大きさが、やたら大きい場合は異常値なのではないか、と推測する

LOFの手順

1. 到達可能性距離（Reachability Distance）の計算

2. 局所到達可能性密度（Local Reachability Density）の計算

3. LOF（Local Outlier Factor）の計算


### 到達可能性距離（Reachability Distance）

` reachability_disktance_k(a,b) = max{k_distance(b), d(a, b)} `

```text

 a  b
    c   d

```

k=2の場合、

- aを取り出す。
- aから一番近い隣接ノードbを取り出す
- bからk番目に近い点を取り出す。この場合はdを取り出す
- a-b間の距離とb-d間の距離の大きい方を取り出したものが `reachability_disktance_k(a,b)` となる
- aから二番目に近い隣接ノードcを取り出す
- cからk番目に近い点を取り出す、以下同様

自分を含めると1+k個の到達可能性距離を求めることになる

<br>

### 局所到達可能性密度（Local Reachability Density）

`lrd_k(a) = 1 ÷ (reachability_disktance_k(a, b) + reachability_disktance_k(a, c)) / 2`


```text

 a  b
    c   d

```

- aを取り出す
- aの近傍点b, cを取り出す
- reachability_disktance_k(a, b)を計算する、つまりbからk番目に遠い点の距離を計算して、大きい方を採用する
- reachability_disktance_k(a, c)を計算する、つまりcからk番目に遠い点の距離を計算して、大きい方を採用する
- reachability_disktance_k(a, b)とreachability_disktance_k(a, c)の平均値を出す
- それを1で割る

### LOF（Local Outlier Factor）

`LOF_k(a) = ( lrd_k(b)/lrd_k(a) + lrd_k(c)/lrd_k(a) ) / k`

これが大きいほど、異常値と考えられる。

<br>

### scikit-learnによるLOF

自分で計算すると、k個の近傍点を取り出すところが難しい。

scikit-learnに実装されているLocalOutlierFactorを使うと簡単に計算できる。



```python
from sklearn.neighbors import LocalOutlierFactor  # process_outlier()
```

kの値は、引数n_neighborsで与える。

適切な値は試行錯誤で決める。10とかが適切かな？

```python
        def local_outlier_factor(n_neighbors=20):
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            X = df[["lat", "lon"]]
            lof.fit(X)
            predicted = lof.fit_predict(X)
            return predicted
```

predictedは外れ値なら-1、正常値なら1が格納されたアレイ。

```python
        # 外れ値を除く処理を施す
        pred = process_outlier(df)

        # 外れ値のみのデータ
        outlier = df.iloc[np.where(pred < 0)]

        # dfを外れ値データを除いたデータに置き換える
        df = df.iloc[np.where(pred > 0)]
```

<br>

## データ補間

ポイントクラウドを可視化してみると補間の必要性が直感的に理解できる。

GPS座標が存在する範囲をグリッド化して、各グリッドでの推定値を計算する。

推定値はinverse distance weighted algorithmを用いる。

![inverse distance weighted algorithm](./assets/inv_dist_interp.gif "inverse distance weighted algorithm")

> 引用元
>
> https://www.e-education.psu.edu/natureofgeoinfo/c7_p9.html

そのためには、グリッドの格子点の近傍に存在する値を取ってくる必要がある。

これにはR-treeを使う。

ということで、当面はR-treeを使った近傍探索のやり方を模索する。


<br>

### Rtree

PythonでのRtreeの実装はいくつか存在する。

libspatialindexをPythonでラッピングしたRtreeを使ってみる。

2024年10月時点ではバージョン1.3がインストールされた。

```bash
(.venv) iida$ pip install rtree
Collecting rtree
  Downloading Rtree-1.3.0-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (543 kB)
     |████████████████████████████████| 543 kB 912 kB/s
Installing collected packages: rtree
Successfully installed rtree-1.3.0
```

> [!NOTE]
>
> libspatialindexのインストールが伴うので、Macだとコンパイル作業が走るかもしれない。
>


<br>

### jupyter notebook

試行錯誤しながらデータを加工するときにはjupyter notebookを使えたほうが便利。

WSLにインストールする方法。

```bash
pip install jupyter
```

venvを使っている場合はその環境下にインストールされる。

WSLでjupyterを走らせるには、設定の追加が必要。

`--generate-config`オプションをつけて起動すると、デフォルトのコンフィグファイルが `~/.jupyter/jupyter_notebook_config.py` に作成される。

```bash
iida@FCCLS0073460:~/git/aburatsubo-terrain-data$ jupyter notebook --generate-config
Writing default config to: /home/iida/.jupyter/jupyter_notebook_config.py
```

場所はホームディレクトリ直下なので、このファイルを環境ごとに変更することはできない。

なので、それをコピーしてjupyterを実行するときに、編集した別のコンフィグファイルを指定する。

まずは、ノートブックを保存したいディレクトリ（ここではipynb）を作って、そこに移動し、デフォルトのコンフィグファイルをコピーする。

```bash
mkdir ipynb
cd ipynb
cp ~/.jupyter/jupyter_notebook_config.py .
```

追加すべき設定は `c.NotebookApp.use_redirect_file = False` という1行だけなので、以下のようにコンフィグファイルの最後に追記する。

```bash
$ echo "c.NotebookApp.use_redirect_file = False" >> jupyter_notebook_config.py
```

実行するときに `--config` オプションで編集したファイルを指定する。

```bash
jupyter notebook --config jupyter_notebook_config.py
```

終了するときは `ctrl-c` を2回連打する。

ターミナルを占有されてしまうのを避けるには、バックグランドで起動する。

```bash
nohup jupyter notebook --config jupyter_notebook_config.py >> jupyter.log 2>&1 &
```

これを停止するならブラウザで `File -> Shut Down` でよい。

```bash
pgrep -f jupyter-notebook
```

でプロセスを探してkillしてもよい。


実行時のオプションが多くて手打ちするのは大変なのでシェルスクリプトにした。

```bash
./start-jupyter.sh
```

これの中身。

```bash
#!/bin/bash

# このファイルのある場所
CURRENT_DIR=$(cd $(dirname $0);pwd)

# Start Jupyter Notebook
# logディレクトリがなければ作成する
if [ ! -d $CURRENT_DIR/log ]; then
    mkdir $CURRENT_DIR/log
fi

# すでに実行中のjupyter notebookがないか確認する
if pgrep -f jupyter-notebook > /dev/null; then
    echo "Jupyter Notebook is already running."
    exit 1
fi

# Jupyter Notebookをバックグラウンドで起動する
# nohup: ログアウト後もプロセスを継続する
nohup jupyter-notebook --config $CURRENT_DIR/jupyter_notebook_config.py --notebook-dir $CURRENT_DIR >> $CURRENT_DIR/log/jupyter.log 2>&1 &
echo "Jupyter Notebook started. Check log/jupyter.log for details."
echo "You can access the notebook at http://localhost:8888"
```
