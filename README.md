# aburatsubo-terrain-data

<br>

[README in Japanese](/README.ja.md)

<br>

These are water depth data collected while floating on the sea in Aburatsubo in a rowboat.

The data was collected using a [Deeper](https://deepersonar.com/en-all) fish finder.

<br>

> [!NOTE]
>
> Aburatsubo Kanagawa Japan, link to google maps
>
> https://maps.app.goo.gl/bigy5d56XkTJaZuN8

<br><br>

## CSV data

<br>

**2025 Oct depth_map_data.csv**

<br>

Before processing csv [ALL_depth_map_data_202510.csv](https://takamitsu-iida.github.io/aburatsubo-terrain-data/data/ALL_depth_map_data_202510.csv)

After processing csv [ALL_depth_map_data_202510_de_dd_ol_ip_mf.csv](https://takamitsu-iida.github.io/aburatsubo-terrain-data/data/ALL_depth_map_data_202510_de_dd_ol_ip_mf.csv)

<br>

|       |            lat |             lon |        depth |            epoch |
|:------|---------------:|----------------:|-------------:|-----------------:|
| count | 164979         | 164979          | 164979       | 164979           |
| mean  |     35.164     |    139.607      |     17.0116  |      1.59862e+12 |
| std   |      0.0021361 |      0.00418346 |      9.74208 |      8.38241e+10 |
| min   |     35.1572    |    139.554      |      1.082   |      1.50491e+12 |
| 25%   |     35.1628    |    139.604      |      8.899   |      1.51399e+12 |
| 50%   |     35.1639    |    139.607      |     15.333   |      1.5927e+12  |
| 75%   |     35.1649    |    139.609      |     24.071   |      1.68696e+12 |
| max   |     35.1797    |    139.622      |     47.539   |      1.75964e+12 |

<br>

Oldest epoch (JST): 2017-09-09 07:08:11

Newest epoch (JST): 2025-10-05 13:19:12

<br><br>

## 3D Visualization (my own 3d map)

I created a custom 3D visualization tool. It's specifically for Aburatsubo in Kanagawa JAPAN.

<br>

[Live Demo](https://takamitsu-iida.github.io/aburatsubo-terrain-data/index-bathymetric-data.html)

[Live Demo (dev version)](https://takamitsu-iida.github.io/aburatsubo-terrain-data/index-bathymetric-data-dev.html)

<br>

**screen shot** (point cloud)

<img src="/assets/screen_point_cloud.png" width="640"/>

<br>

**screen shot** (wireframe)

<img src="/assets/screen_wireframe.png" width="640"/>

<br>

**screen shot** (mesh)

<img src="/assets/screen_mesh.png" width="640"/>

<br>

**screen shot** (contour)

<img src="/assets/screen_contour.png" width="640"/>

<br><br>

## 2D Visualization (by Deeper)

<br>

<img src="/assets/2d-map-by-deeper.png" width="640"/>
