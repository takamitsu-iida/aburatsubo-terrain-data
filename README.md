# aburatsubo-terrain-data

DeeperからCSV形式でデータをダウンロードする。

https://maps.fishdeeper.com/ja-jp

ファイル名は共通で `bathymetry_data.csv` となっている。

サイズは約8MBで約22万行。

各行は `lat, lon, depth, unix_time_in_msec` の情報で構成されている。

先頭の3行

```csv
35.162872,139.61423,2.747,1659737621643
35.16289,139.61423,2.831,1659737623625
35.162903,139.61421,2.987,1659737624655
```

1行目のUNIX時間1659737621643は2022年8月6日7時13分41.643秒を表している。

2行目のUNIX時間との差は約2秒。
ということは2秒に一度、スキャンしていることになる。

最後の3行

```csv
35.16161,139.60029,32.5,0
35.16162,139.60031,33.05,0
35.16162,139.60031,32.39,0
```

2017年よりも古いデータにはUNIX時間が欠落していて0になっている。

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

<br><br>

## 座標系メモ

<dl>
<dt>latitude</dt> <dd>緯度</dd>
<dt>longitude</dt> <dd>軽度</dd>
<dt>地球の半径</dt> <dd>6378.137 km 測地測量の基準としての半径</dd>
</dl>


<br><br>

## データ処理

[マーチングスクエア](https://urbanspr1nter.github.io/marchingsquares/){:target="_blank"}

[マーチングキューブ](https://tatsy.github.io/programming-for-beginners/cpp/march-cubes/){:target="_blank"}

[Python で曲面近似（サーフェスフィッティング）する](https://chuckischarles.hatenablog.com/entry/2020/02/06/081238){:target="_blank"}

[カーブフィッティング手法 scipy.optimize.curve_fit の使い方を理解する](https://qiita.com/maskot1977/items/e4f5f71200180865986e){:target="_blank"}

<br><br>

## Goコードの断片


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