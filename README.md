# aburatsubo-terrain-data

Deeperの地図アプリからCSV形式でデータをダウンロードする。

https://maps.fishdeeper.com/ja-jp

ファイル名は共通で `bathymetry_data.csv` となっている。

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

pandasで読み取った統計値。

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
11万件まで減った。

|       |             lat |             lon |        depth |
|:------|----------------:|----------------:|-------------:|
| count | 119236          | 119236          | 119236       |
| mean  |     35.1635     |    139.608      |     16.0729  |
| std   |      0.00170909 |      0.00404937 |      9.18661 |
| min   |     35.1572     |    139.599      |      1.426   |
| 25%   |     35.1625     |    139.604      |      8.117   |
| 50%   |     35.1637     |    139.608      |     15.032   |
| 75%   |     35.1644     |    139.61       |     21.33    |
| max   |     35.171      |    139.622      |     45.862   |





<br><br>

## 実行環境メモ

データの加工にPythonを使うので、venvを使ってPython仮想環境をセットアップする。

```bash
python3 -m venv .venv
```

シェル利用時にPython仮想環境を自動切換えするために`direnv`を利用する。

```bash
direnv edit .
```

エディタが開くので、以下の内容を追加して保存する。

```text
source .venv/bin/activate
unset PS1
```

<br><br>

## 座標系メモ

latとlon、どっちが緯度でどっちが経度かすぐにわからなくなる。

<dl>
<dt>latitude</dt> <dd>緯度</dd>
<dt>longitude</dt> <dd>経度</dd>
<dt>地球の半径</dt> <dd>6378.137 km 測地測量の基準としての半径</dd>
</dl>

<br><br>

## 参考文献（データ処理）

[マーチングスクエア](https://urbanspr1nter.github.io/marchingsquares/){:target="_blank"}

[マーチングキューブ](https://tatsy.github.io/programming-for-beginners/cpp/march-cubes/){:target="_blank"}

[Python で曲面近似（サーフェスフィッティング）する](https://chuckischarles.hatenablog.com/entry/2020/02/06/081238){:target="_blank"}

[カーブフィッティング手法 scipy.optimize.curve_fit の使い方を理解する](https://qiita.com/maskot1977/items/e4f5f71200180865986e){:target="_blank"}

<br><br>

## 実装メモ

Go言語の方が書きやすいものの、データ加工処理の容易さを考慮してPythonで処理。

<br><br>

### 重複削除のやり方

groupbyで集約しつつ、水深の平均を計算する。

元データから重複行をすべて削除し、groupbyで計算したものを加える。

```python
    def process_duplicated(df: pd.DataFrame):
        dfx = df[["lat", "lon", "depth"]]
        dfx_uniq = df.drop_duplicates(subset=["lat", "lon"], keep=False)
        dfx_duplicated = dfx.groupby(["lat", "lon"])["depth"].mean().reset_index()
        df = pd.concat([dfx_uniq, dfx_duplicated]).reset_index(drop=True)
        return df
```

### 外れ値検出

Local Outlier Factorを利用する。

```python
from sklearn.neighbors import LocalOutlierFactor  # process_outlier()
```

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



### データ補間の考え方

案１．

- pandasにデータを読ませる
- 10mグリッドを想定する
- 幅優先探索でグリッドを移動させる
- グリッド内にデータが0件ならグリッドを、訪問済みフラグを立ててから次のグリッドに移動する
- グリッド内にデータが1件以上あるなら、1mグリッドでデータを補間する
    - 補間方法はRBFを利用する
- 補間したデータをファイルに書き出す

案２．

データが時系列でソートされていることを想定する。

- 先頭データを取り出す
- 10mグリッド内に収まるデータを取り出して補間する
- そのグリッドから抜けるところまでデータを進める

<br><br>

## Goコードの断片（あとで消す）


```go
func Round(x, unit float64) float64 {
    return math.Round(x/unit) * unit
}
```

```go
package main

 import (
 	"fmt"
 	"math"
 )

 type Coordinates struct {
 	Latitude  float64
 	Longitude float64
 }

 const radius = 6371 // Earth's mean radius in kilometers

 func degrees2radians(degrees float64) float64 {
 	return degrees * math.Pi / 180
 }

 func (origin Coordinates) Distance(destination Coordinates) float64 {
 	degreesLat := degrees2radians(destination.Latitude - origin.Latitude)
 	degreesLong := degrees2radians(destination.Longitude - origin.Longitude)
 	a := (math.Sin(degreesLat/2)*math.Sin(degreesLat/2) +
 		math.Cos(degrees2radians(origin.Latitude))*
 			math.Cos(degrees2radians(destination.Latitude))*math.Sin(degreesLong/2)*
 			math.Sin(degreesLong/2))
 	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
 	d := radius * c

 	return d
 }

 func main() {
 	pointA := Coordinates{2.990353, 101.533913}
 	pointB := Coordinates{2.960148, 101.577888}

 	fmt.Println("Point A : ", pointA)
 	fmt.Println("Point B : ", pointB)

 	distance := pointA.Distance(pointB)
 	fmt.Printf("The distance from point A to point B is %.2f kilometers.\n", distance)

 }
```