import * as THREE from "three";
import { OrbitControls } from "three/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from 'three/libs/CSS2DRenderer.js';

/*

Pythonでデータを集約しているので、JavaScript側ではそのまま利用します。

ドロネー三角形でメッシュを生成して可視化します。

*/

// lil-gui
import { GUI } from "three/libs/lil-gui.module.min.js";

// stats.js
import Stats from "three/libs/stats.module.js";

// ドロネー三角形
import Delaunator from "delaunatorjs";

/*
import Delaunator from "delaunatorjs";
を実現するには、ちょっと苦労がある。

https://github.com/mapbox/delaunator
ここからReleasesの最新版（2024年9月時点でv5.0.1）をダウンロードする。
この中にindex.jsがあるので、これを使う。

delaunatorは内部でrobust-predicatesのorient2dを使っているため、
orient2dが見つからないというエラーが発生する。

https://github.com/mourner/robust-predicates
ここからReleasesの最新版（2024年9月時点でv3.0.2）をダウンロードする。
この中のsrcフォルダのjavascriptファイルをコピーして使う。

HTMLではこのようなimportmapを使う。

<!-- three.js -->
<script type="importmap">
  {
    "imports": {
      "three": "./static/build/three.module.js",
      "three/libs/": "./static/libs/",
      "three/controls/": "./static/controls/",
      "robust-predicates": "./static/libs/robust-predicates-3.0.2/orient2d.js",
      "delaunatorjs": "./static/libs/delaunator-5.0.1/index.js"
    }
  }
</script>
*/

export class Main {

  // Three.jsを表示するコンテナのHTML要素
  container;

  // そのコンテナのサイズ
  sizes = {
    width: 0,
    height: 0
  }

  // 水深を表示するHTML要素
  depthContainer;

  // 緯度経度を表示するHTML要素
  coordinatesContainer;

  // 方位磁針を表示するHTML要素
  compassContainer;
  compassElement;

  // Three.jsの各種インスタンス
  scene;
  camera;
  renderer;
  controller;
  statsjs;

  // マウス座標にあるオブジェクトを取得するためのraycaster(renderDepthで利用)
  raycaster = new THREE.Raycaster();

  // マウス座標(renderDepthで利用、mousemoveイベントで値をセット)
  // THREE.Vector2();
  mousePosition;

  // 前フレーム時点でのマウス座標(renderDepthで利用)
  // THREE.Vector2();
  previousMousePosition;

  // カメラの向き(renderCompassで利用)
  cameraDirection;

  // 前フレーム時点でのカメラの向き(renderCompassで利用)
  previousCameraDirection;

  // ラベル表示用CSS2DRenderer
  cssRenderer;

  // レンダリング用のパラメータ
  renderParams = {
    animationId: null,
    clock: new THREE.Clock(),
    delta: 0,
    interval: 1 / 30,  // = 30fps
  }

  params = {

    // 利用可能なCSVファイルのリスト（HTMLで指定したもので上書きされる）
    availableDatasets: {
      'bathymetric_data': './static/data/bathymetric_data.csv',
    },

    // 現在選択されているデータセット名
    currentDataset: 'bathymetric_data',

    // ファイルをローディングしている状態かどうか
    isLoading: false,

    // 海底地形図の(lon, lat)をThree.jsのXZ座標のどの範囲に描画するか
    // 持っているGPSデータに応じて調整する
    xzGridSize: 200,  // 200を指定する場合は -100～100 の範囲に描画する

    // xzGridSizeにあわせるために、どのくらい緯度経度の値を拡大するか（自動で計算する）
    xzScale: 10000,  // これは仮の値で、CSVデータを読み込んだ後に正規化する

    // CSVテキストをパースして作成するデータ配列
    // 画面表示に適した値に正規化するのでCSVの値とは異なることに注意
    // [ {lat: 35.16900046, lon: 139.60695032, depth: -10.0}, {...}, ... ]
    depthMapData: null,

    // 三浦市のtopojsonファイルのURL
    topojsonPath: "./static/data/aburatsubo.json",

    // topojsonデータ
    topojsonData: null,

    // topojsonデータに含まれるobjectName（三浦市のデータなら"miura"）
    topojsonObjectName: "miura",

    // guiをクローズド状態で開始するか？
    guiClosed: false,

    // ポイントクラウドを表示するか？
    showPointCloud: true,

    // ポイントクラウドのパーティクルサイズ
    pointSize: 0.2,

    // CSVに何個のデータがあるか（CSV読み取り時に自動計算）
    totalPointCount: 0,

    // ワイヤーフレーム表示にする？
    wireframe: false,

    // コントローラの設定
    autoRotate: false,
    autoRotateSpeed: 1.0,

    // 緯度経度の最小値、最大値、中央値（CSVから自動で読み取る）
    minLon: 0,
    minLat: 0,
    maxLon: 0,
    maxLat: 0,
    centerLon: 0,
    centerLat: 0,

    // 画面表示用に正規化した緯度経度の最大値、最小値（自動計算）
    normalizedMinLon: 0,
    normalizedMinLat: 0,
    normalizedMaxLon: 0,
    normalizedMaxLat: 0,
    normalizedCenterLon: 0,
    normalizedCenterLat: 0,

    // ランドマークのオブジェクト配列
    // [{ lon: 139.60695032, lat: 35.16200000, depth: -10, name_ja: 'ヤギ瀬', name: 'Yagise' },
    //  { lon: 139.61539000, lat: 35.16160000, depth: 0, name_ja: 'みなとや', name: 'Minatoya' },...]
    landmarks: [],

    // ランドマークを表示する？
    showLandmarks: true,

    // ランドマークラベルの高さ
    labelY: 20,

    // 縮尺を表示する？
    showScale: true,

    // コンパスを表示する？
    showCompass: true,

    // stats.jsを表示する？
    showStats: true,

    // ドロネー三角形のフィルタリングパラメータ
    enableDelaunayFilter: true,
    maxTriangleEdgeLength: 20,  // エッジの最大長さ 画面サイズxzGridSizeが200の場合の初期値
  }

  //
  // データセットを切り替える関数
  //
  switchDataset = async (datasetName) => {
    if (this.params.isLoading) {
      return;
    }

    // ローディング状態を表示
    this.showLoadingIndicator(true);

    try {
      // プリセットファイル
      let dataPath = this.params.availableDatasets[datasetName];

      // CSVデータをロード
      await this.loadCsv(dataPath);

      // データを正規化
      this.normalizeDepthMapData();

      // 現在のデータセット名を更新
      this.params.currentDataset = datasetName;

      // コンテンツを再初期化
      this.initContents();

    } catch (error) {
      console.error(`Error switching dataset: ${error}`);
    } finally {
      // ローディング状態を非表示
      this.showLoadingIndicator(false);
    }
  }

  // ローディングインジケータの表示/非表示
  showLoadingIndicator = (show) => {
    this.params.isLoading = show;

    const loadingContainer = document.getElementById('loadingContainer');

    if (show) {
      // ローディング画面を表示
      loadingContainer.style.display = 'block';
      loadingContainer.classList.remove('fadeout');
      loadingContainer.classList.add('visible');
    } else {
      // ローディング画面を非表示にする
      const interval = setInterval(() => {
        loadingContainer.classList.add('fadeout');
        clearInterval(interval);
      }, 500);

      // アニメーション完了後に完全に非表示にする
      loadingContainer.addEventListener('transitionend', (event) => {
        if (event.target === loadingContainer) {
          // 表示を完全に無効化してマウスイベントを通すようにする
          loadingContainer.style.display = 'none';
          loadingContainer.classList.remove('visible', 'fadeout');
        }
      }, { once: true }); // once: true で一度だけ実行
    }

  }


  // 変数
  // 地形図のポイントクラウド（guiで表示を操作するためにインスタンス変数にする）
  pointMeshList;

  // 変数
  // 地形図のメッシュのリスト（guiで表示を操作するためにインスタンス変数にする）
  terrainMeshList = [];

  // コンストラクタ
  constructor(params = {}) {
    this.params = Object.assign(this.params, params);
    this.init();
  }


  init = async () => {

    // 取得するデータのパスを取得
    const dataPath = this.params.availableDatasets[this.params.currentDataset];

    // データを読み込む
    await Promise.all([
      this.loadCsv(dataPath),
      this.loadTopojson(this.params.topojsonPath)
    ]);

    // 初期状態で表示されているローディング状態を非表示にする(50ms待機してからフェードアウト)
    this.showLoadingIndicator(false);

    if (this.params.depthMapData === null) {
      return;
    }

    if (this.params.topojsonData === null) {
      return;
    }

    // 緯度経度の値を画面表示用に正規化する
    this.normalizeDepthMapData();

    // scene, camera, renderer, controllerを初期化
    this.initThreejs();

    // stats.jsを初期化
    this.initStatsjs();

    // lil-guiを初期化
    this.initGui();

    // 色の凡例を初期化
    this.initLegend();

    // コンテンツを初期化
    this.initContents();
  }


  initContents = () => {
    // アニメーションを停止
    this.stop();

    // シーン上のメッシュを削除する
    this.clearScene();

    // 全てを削除した状態で描画
    this.renderer.render(this.scene, this.camera);

    // CSVデータをドロネー三角形でメッシュ化する
    this.initDelaunayFromCsv();

    // topojsonデータから地図のシェイプを作成
    const shapes = this.createShapesFromTopojson(this.params.topojsonData, this.params.topojsonObjectName);

    // シェイプの配列からメッシュを作成
    this.createMeshFromShapes(shapes);

    // ランドマークを表示
    this.initLandmarks();

    // 縮尺を表示
    this.initScale();

    // 方位磁針を表示
    this.initCompass();

    // フレーム毎の処理
    this.render();
  }


  // pathで指定されたCSVファイルを取得してパースする
  loadCsv = async (path) => {
    try {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(`HTTP status: ${response.status}`);
      }

      // テキストデータを取得
      const text = await response.text();

      // CSVのテキストデータをパース
      this.params.depthMapData = this.parseCsv(text);

    } catch (error) {
      const errorMessage = `Error while loading ${path}: ${error}`;
      console.error(errorMessage);
      let p = document.createElement('p');
      p.textContent = errorMessage;
      p.style.color = 'white';
      loadingContainer.appendChild(p);
    }
  }


  parseCsv = (text) => {
    // 行に分割
    const lines = text.split('\n').filter(line => line.trim().length > 0);

   // 先頭行が数字で始まる場合はヘッダ無しと判定
    const firstLine = lines[0].trim();
    const isHeader = isNaN(Number(firstLine.split(',')[0]));
    const headers = isHeader ? lines[0].split(',').map(h => h.trim()) : ["lat", "lon", "depth", "epoch"];

    // データ開始行のインデックス
    const startIdx = isHeader ? 1 : 0;

    // 行ごとにパースしたデータを格納する配列
    const dataList = [];

    // 緯度経度の最大値、最小値を取得するための変数
    let minLat = 9999;
    let maxLat = -9999;
    let minLon = 9999;
    let maxLon = -9999;

    // 2行目以降をパース
    for (let i = startIdx; i < lines.length; i++) {
      const rows = lines[i].split(',');
      if (rows.length === headers.length) {
        const d = {};
        for (let j = 0; j < headers.length; j++) {
          d[headers[j]] = parseFloat(rows[j].trim());
        }
        // 5列ある場合は、クラスタ番号が入っている
        if (rows.length === 5) {
          d['cluster'] = parseInt(rows[4].trim());
        }
        dataList.push(d);

        // 緯度経度の最大値、最小値を調べる
        minLat = Math.min(minLat, d.lat);
        maxLat = Math.max(maxLat, d.lat);
        minLon = Math.min(minLon, d.lon);
        maxLon = Math.max(maxLon, d.lon);
      }
    }

    // console.log(`minLat: ${minLat}\nmaxLat: ${maxLat}\nminLon: ${minLon}\nmaxLon: ${maxLon}`);

    // 後から参照できるように保存しておく
    this.params.minLon = minLon;
    this.params.maxLon = maxLon;
    this.params.minLat = minLat;
    this.params.maxLat = maxLat;
    this.params.centerLon = (minLon + maxLon) / 2;
    this.params.centerLat = (minLat + maxLat) / 2;

    // 全部で何個のデータがあるか
    this.params.totalPointCount = dataList.length;

    // 緯度の差分、経度の差分で大きい方を取得
    const diffSize = Math.max(maxLat - minLat, maxLon - minLon);

    // このdiffSizeがxzGridSizeになるように係数を計算
    this.params.xzScale = this.params.xzGridSize / diffSize;

    return dataList;
  }


  // normalizeDepthMapData()で行っている正規化をメソッド化
  // Three.jsのZ軸の向きが手前方向なので、緯度方向はマイナスにする必要がある
  // これでTopojsonの座標を正規化した場合、
  // シェイプをXY平面からXZ平面に向きを変えるときに
  //   geometry.rotateX(Math.PI / 2);
  // という向きにしないと、地図が上下逆さまになる
  //
  normalizeCoordinates = ([lon, lat]) => {
    const scale = this.params.xzScale;
    const centerLon = this.params.centerLon;
    const centerLat = this.params.centerLat;
    return [
      (lon - centerLon) * scale,
      -1 * (lat - centerLat) * scale
    ];
  }


  // normalizeDepthMapData()で行っている正規化の逆変換
  inverseNormalizeCoordinates = (x, z) => {
    const scale = this.params.xzScale;
    const centerLon = this.params.centerLon;
    const centerLat = this.params.centerLat;
    return [
      x / scale + centerLon,
      -1 * z / scale + centerLat
    ];
  }


  normalizeDepthMapData = () => {

    // 緯度経度の中央値を取り出す
    const centerLon = this.params.centerLon;
    const centerLat = this.params.centerLat;

    // 拡大率
    const scale = this.params.xzScale;

    // 正規化後の最小緯度、最大緯度、最小経度、最大経度
    let normalizedMinLat = 9999;
    let normalizedMaxLat = -9999;
    let normalizedMinLon = 9999;
    let normalizedMaxLon = -9999;

    // params.depthMapDataを上書きで正規化する
    this.params.depthMapData.forEach((d) => {

      // 経度(lon)はX軸に対応する
      // センターに寄せて、スケールをかける
      const lon = (d.lon - centerLon) * scale;

      // 緯度(lat)はZ軸に対応する
      // Three.jsのZ軸の向きと、地図の南北は逆になるのでマイナスをかける
      const lat = -1 * (d.lat - centerLat) * scale;

      // 深さ(depth)はY軸に対応する
      // 深さなので、マイナスをかける
      const depth = -1 * d.depth;

      // 正規化したデータを上書きで保存
      d.lat = lat;
      d.lon = lon;
      d.depth = depth;

      // 最小緯度、最大緯度、最小経度、最大経度を更新
      normalizedMinLat = Math.min(normalizedMinLat, lat);
      normalizedMaxLat = Math.max(normalizedMaxLat, lat);
      normalizedMinLon = Math.min(normalizedMinLon, lon);
      normalizedMaxLon = Math.max(normalizedMaxLon, lon);
    });

    // 正規化した最小値、最大値、中央値を保存しておく
    this.params.normalizedMinLat = normalizedMinLat;
    this.params.normalizedMaxLat = normalizedMaxLat;
    this.params.normalizedMinLon = normalizedMinLon;
    this.params.normalizedMaxLon = normalizedMaxLon;
    this.params.normalizedCenterLon = (normalizedMinLon + normalizedMaxLon) / 2;
    this.params.normalizedCenterLat = (normalizedMinLat + normalizedMaxLat) / 2;
    // console.log(`normalizedMinLat: ${this.params.normalizedMinLat}\nnormalizedMaxLat: ${this.params.normalizedMaxLat}\nnormalizedMinLon: ${this.params.normalizedMinLon}\nnormalizedMaxLon: ${this.params.normalizedMaxLon}`);
  }

  loadTopojson = async (path) => {
    try {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(`HTTP status: ${response.status}`);
      }

      // topojsonデータを取得
      const topojsonData = await response.json();

      if (topojsonData.hasOwnProperty('transform')) {
        this.params.translate = topojsonData.transform.translate;
      } else {
        new Error('No transform property in jsonData');
      }

      if (!topojsonData.hasOwnProperty('objects')) {
        new Error('No objects property in jsonData');
      }

      if (!topojsonData.objects.hasOwnProperty(this.params.topojsonObjectName)) {
        new Error(`No ${this.params.topojsonObjectName} property in objects`);
      }

      // jsonデータを保存
      this.params.topojsonData = topojsonData;
    } catch (error) {
      const errorMessage = `Error while loading ${path}: ${error}`;
      console.error(errorMessage);
      let p = document.createElement('p');
      p.textContent = errorMessage;
      p.style.color = 'white';
      loadingContainer.appendChild(p);
    }
  }


  initThreejs = () => {

    // Three.jsを表示するHTML要素
    this.container = document.getElementById("threejsContainer");

    // そのコンテナのサイズ
    this.sizes.width = this.container.clientWidth;
    this.sizes.height = this.container.clientHeight;

    // 水深を表示するHTML要素
    this.depthContainer = document.getElementById("depthContainer");

    // 緯度経度を表示するHTML要素
    this.coordinatesContainer = document.getElementById("coordinatesContainer");

    // resizeイベントのハンドラを登録
    window.addEventListener("resize", this.onWindowResize, false);

    // シーン
    this.scene = new THREE.Scene();

    // カメラ
    this.camera = new THREE.PerspectiveCamera(
      60,
      this.sizes.width / this.sizes.height,
      1,
      1000
    );
    this.camera.position.set(0, 100, 100);

    // レイヤを設定
    this.camera.layers.enable(0); // enabled by default
    this.camera.layers.enable(1); // 1: landmark
    this.camera.layers.enable(2); // 2: scale

    // レンダラ
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
    });

    this.renderer.setSize(this.sizes.width, this.sizes.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // 地図は大きいのでクリッピングを有効にして表示領域を制限する
    this.renderer.localClippingEnabled = true;

    // コンテナにレンダラを追加
    this.container.appendChild(this.renderer.domElement);

    // コントローラ
    this.controller = new OrbitControls(this.camera, this.renderer.domElement);
    this.controller.autoRotate = this.params.autoRotate;
    this.controller.autoRotateSpeed = this.params.autoRotateSpeed;

    // 軸を表示
    //
    //   Y(green)
    //    |
    //    +---- X(red)
    //   /
    //  Z(blue)
    //
    const xzGridSize = this.params.xzGridSize;
    const axesHelper = new THREE.AxesHelper(xzGridSize);
    this.scene.add(axesHelper);

    // 環境光
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));

    // ディレクショナルライト
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
    directionalLight.position.set(-this.params.xzGridSize/2, 0, 0);
    this.scene.add(directionalLight);

    // 正規化したマウス座標を保存
    this.mousePosition = new THREE.Vector2();
    this.previousMousePosition = new THREE.Vector2();
    this.renderer.domElement.addEventListener("mousemove", (event) => {
      this.mousePosition.x = (event.clientX / this.sizes.width) * 2 - 1;
      this.mousePosition.y = -(event.clientY / this.sizes.height) * 2 + 1;
    }, false);

    // ラベル表示に利用するCSS2DRendererを初期化
    this.cssRenderer = new CSS2DRenderer();
    this.cssRenderer.setSize(this.sizes.width, this.sizes.height);
    this.cssRenderer.domElement.style.position = 'absolute';
    this.cssRenderer.domElement.style.top = 0;
    this.cssRenderer.domElement.style.pointerEvents = 'none';
    this.container.appendChild(this.cssRenderer.domElement);
  }


  initGui = () => {
    const guiContainer = document.getElementById("guiContainer");

    const gui = new GUI({
      container: guiContainer,
      width: 300,
    });

    // 画面が小さい場合は初期状態で閉じた状態にする
    if (this.params.guiClosed ||
      window.matchMedia('(max-width: 640px)').matches ||
      window.matchMedia('(max-height: 640px)').matches) {
      gui.close();
    }

    // 一度だけ実行するための関数
    const doLater = (job, tmo) => {
      // 処理が登録されているならタイマーをキャンセル
      var tid = doLater.TID[job];
      if (tid) {
        window.clearTimeout(tid);
      }
      // タイムアウト登録する
      doLater.TID[job] = window.setTimeout(() => {
        // 実行前にタイマーIDをクリア
        doLater.TID[job] = null;
        // 登録処理を実行
        job.call();
      }, tmo);
    }

    // 処理からタイマーIDへのハッシュ
    doLater.TID = {};

    // 表示切り替えフォルダ
    const displayFolder = gui.addFolder(navigator.language.startsWith("ja") ? "表示切り替え" : "Display");

    displayFolder
      .add(this.params, "autoRotate")
      .name(navigator.language.startsWith("ja") ? "自動回転" : "rotation")
      .onChange((value) => {
        this.controller.autoRotate = value;
      });

    displayFolder
      .add(this.params, "pointSize")
      .name(navigator.language.startsWith("ja") ? "ポイントサイズ" : "pointSize")
      .min(0.1)
      .max(1.0)
      .step(0.1)
      .onChange((value) => {
        doLater(() => {
          this.pointMeshList.forEach((pointMesh) => {
            pointMesh.material.size = value;
          });
        }, 250);
      });

      const displayParams = {
      'wireframe': () => {
        this.params.wireframe = !this.params.wireframe;
        this.terrainMeshList.forEach((terrainMesh) => {
          terrainMesh.material.wireframe = this.params.wireframe;
        });
      },

      'pointCloud': () => {
        this.params.showPointCloud = !this.params.showPointCloud;
        this.pointMeshList.forEach((pointMesh) => {
          pointMesh.visible = this.params.showPointCloud;
        });
      },

      'landmark': () => {
        this.params.showLandmarks = !this.params.showLandmarks;
        this.camera.layers.toggle(1);
      },

      'scale': () => {
        this.params.showScale = !this.params.showScale;
        this.camera.layers.toggle(2);
      },
    };

    displayFolder
      .add(displayParams, "wireframe")
      .name(navigator.language.startsWith("ja") ? "ワイヤーフレーム表示" : "Wireframe");

    displayFolder
      .add(displayParams, "pointCloud")
      .name(navigator.language.startsWith("ja") ? "ポイントクラウド表示" : "Show point cloud");

    displayFolder
      .add(displayParams, 'landmark')
      .name(navigator.language.startsWith("ja") ? "ランドマーク表示" : "Show landmark");

    displayFolder
      .add(displayParams, 'scale')
      .name(navigator.language.startsWith("ja") ? "縮尺表示" : "Show scale");

    // ドロネー三角形に関するフィルタ処理
    const delaunayFolder = gui.addFolder(navigator.language.startsWith("ja") ? "ドロネー三角形" : "Delaunay Filter");

    delaunayFolder
      .add(this.params, "enableDelaunayFilter")
      .name(navigator.language.startsWith("ja") ? "フィルタ有効" : "Enable Filter")
      .onChange(() => {
        this.initContents(); // メッシュを再構築
      });

    delaunayFolder
      .add(this.params, "maxTriangleEdgeLength")
      .name(navigator.language.startsWith("ja") ? "最大辺長" : "Max Edge Length")
      .min(5)
      .max(50)
      .step(1)
      .onChange(() => {
        doLater(() => {
          this.initContents(); // メッシュを再構築
        }, 250);
      });

    // データセット選択フォルダ
    const dataFolder = gui.addFolder(navigator.language.startsWith("ja") ? "データセット" : "Dataset");

    // プリセットデータセットの選択
    dataFolder
      .add(this.params, "currentDataset", Object.keys(this.params.availableDatasets))
      .name(navigator.language.startsWith("ja") ? "プリセット" : "Preset")
      .onChange((value) => {
        this.switchDataset(value);
      });

  }


  initStatsjs = () => {
    if (this.params.showStats === false) {
      return;
    }

    let container = document.getElementById("statsjsContainer");
    if (!container) {
      container = document.createElement("div");
      container.id = "statsjsContainer";
      this.container.appendChild(container);
    }

    this.statsjs = new Stats();
    this.statsjs.dom.style.position = "relative";
    this.statsjs.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    container.appendChild(this.statsjs.dom);
  }


  render = () => {
    // 再帰処理
    this.renderParams.animationId = requestAnimationFrame(this.render);

    this.renderParams.delta += this.renderParams.clock.getDelta();
    if (this.renderParams.delta < this.renderParams.interval) {
      return;
    }

    {
      // stats.jsを更新
      if (this.statsjs && this.params.showStats) {
        this.statsjs.update();
      }

      // カメラコントローラーを更新
      this.controller.update();

      // シーンをレンダリング
      this.renderer.render(this.scene, this.camera);

      // CSS2DRendererをレンダリング
      this.cssRenderer.render(this.scene, this.camera);

      // 水深を表示
      this.renderDepth();

      // 方位磁針を更新
      this.renderCompass();
    }

    this.renderParams.delta %= this.renderParams.interval;
  }


  stop = () => {
    if (this.renderParams.animationId) {
      cancelAnimationFrame(this.renderParams.animationId);
    }
    this.renderParams.animationId = null;
  }


  clearScene = () => {
    const objectsToRemove = [];

    this.scene.children.forEach((child) => {
      if (child.type === 'AxesHelper' || child.type === 'GridHelper' || String(child.type).indexOf('Light') >= 0) {
        return;
      }
      objectsToRemove.push(child);
    });

    objectsToRemove.forEach((object) => {
      this.scene.remove(object);
      if (object.geometry) {
        object.geometry.dispose();
      }
      if (object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach(material => material.dispose());
        } else {
          object.material.dispose();
        }
      }
    });
  }


  onWindowResize = (event) => {
    this.sizes.width = this.container.clientWidth;
    this.sizes.height = this.container.clientHeight;

    this.camera.aspect = this.sizes.width / this.sizes.height;
    this.camera.updateProjectionMatrix();

    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(this.sizes.width, this.sizes.height);

    // CSS2DRendererにもサイズを反映する
    this.cssRenderer.setSize(this.sizes.width, this.sizes.height);
  };


  splitPointsByCluster = (data) => {
    const clusters = {};
    data.forEach(point => {
      const clusterId = point.cluster;
      if (!clusters[clusterId]) clusters[clusterId] = [];
      clusters[clusterId].push(point);
    });

    // クラスタごとの配列リストを返す
    return Object.values(clusters);
  }


  // 三角形が有効かどうかを判定する関数
  isValidTriangle = (posA, posB, posC) => {
    if (!this.params.enableDelaunayFilter) {
      return true;
    }

    // 各辺の長さを計算
    const edgeAB = Math.sqrt(
      Math.pow(posA.x - posB.x, 2) + Math.pow(posA.z - posB.z, 2)
    );
    const edgeBC = Math.sqrt(
      Math.pow(posB.x - posC.x, 2) + Math.pow(posB.z - posC.z, 2)
    );
    const edgeCA = Math.sqrt(
      Math.pow(posC.x - posA.x, 2) + Math.pow(posC.z - posA.z, 2)
    );

    // エッジ長さのチェック
    if (edgeAB > this.params.maxTriangleEdgeLength ||
        edgeBC > this.params.maxTriangleEdgeLength ||
        edgeCA > this.params.maxTriangleEdgeLength) {
      return false;
    }

    return true;
  }


  // CSVデータを使ってドロネー三角形でメッシュ化する
  initDelaunayFromCsv = () => {

    // 作成するポイントクラウドのリスト
    const pointMeshList = [];

    // 作成するメッシュのリスト
    const terrainMeshList = [];

    // CSVデータをそのまま使う
    const data = this.params.depthMapData;

    // dataをクラスタごとに分解する
    const clusteredData = this.splitPointsByCluster(data);

    // クラスタごとにドロネー三角形でメッシュを作成
    clusteredData.forEach(cluster => {

      // Three.jsのVector3配列を作成して、データを格納する
      const positions = [];
      const colors = [];
      cluster.forEach(point => {
        // lon: X, lat: Z, depth: Y
        positions.push(new THREE.Vector3(point.lon, point.depth, point.lat));
        const color = this.getDepthColor(point.depth);
        colors.push(color.r, color.g, color.b);
      });

      // ポイントクラウドのジオメトリを作成
      const geometry = new THREE.BufferGeometry().setFromPoints(positions);
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

      // マテリアルを作成
      const pointsMaterial = new THREE.PointsMaterial({
        color: 0x99ccff,
        size: this.params.pointSize,
      });

      // 点群を作成
      const pointMesh = new THREE.Points(geometry, pointsMaterial);
      pointMesh.visible = this.params.showPointCloud;
      this.scene.add(pointMesh);
      pointMeshList.push(pointMesh);

      const delaunay = Delaunator.from(
        positions.map(v => [v.x, v.z])
      );

      const meshIndex = [];

      for (let i = 0; i < delaunay.triangles.length; i += 3) {
        const a = delaunay.triangles[i + 0];
        const b = delaunay.triangles[i + 1];
        const c = delaunay.triangles[i + 2];

        // 三角形の各頂点を取得
        const posA = positions[a];
        const posB = positions[b];
        const posC = positions[c];

        // 三角形が有効かどうかを判定して、メッシュインデックスに追加
        if (this.isValidTriangle(posA, posB, posC)) {
          meshIndex.push(a, b, c);
        }
      }

      geometry.setIndex(meshIndex);
      geometry.computeVertexNormals();

      // メッシュマテリアル
      const material = new THREE.MeshLambertMaterial({
        vertexColors: true,
        wireframe: this.params.wireframe,
      });

      // メッシュを生成
      const terrainMesh = new THREE.Mesh(geometry, material);
      this.scene.add(terrainMesh);
      terrainMeshList.push(terrainMesh);
    });

    // 何個のポイントが表示されているかを表示
    document.getElementById('debugContainer').textContent = `${this.params.totalPointCount.toLocaleString()} points displayed`;

    // インスタンス変数に保存
    this.pointMeshList = pointMeshList;
    this.terrainMeshList = terrainMeshList;
  }


  depthSteps = [
    -60, -55, -50, -45, -40, -35, -30, -25, -20, -16, -12, -10, -8, -6, -5, -4, -3, -2, -1,
  ];


  depthColors = {
    '-60': 0x2e146a,
    '-55': 0x451e9f,
    '-50': 0x3b31c3,
    '-45': 0x1f47de,
    '-40': 0x045ef9,
    '-35': 0x0075fd,
    '-30': 0x008ffd,
    '-25': 0x01aafc,
    '-20': 0x01c5fc,
    '-16': 0x45ccb5,
    '-12': 0x90d366,
    '-10': 0xb4df56,
    '-8': 0xd9ed4c,
    '-6': 0xfdfb41,
    '-5': 0xfee437,
    '-4': 0xfecc2c,
    '-3': 0xfeb321,
    '-2': 0xff9b16,
    '-1': 0xff820b,
  }


  getDepthColor = (depth) => {
    const depthSteps = this.depthSteps;
    const depthColors = this.depthColors;
    for (let i = 0; i < depthSteps.length; i++) {
      if (depth <= depthSteps[i]) {
        return new THREE.Color(depthColors[depthSteps[i]]);
      }
    }
    return new THREE.Color(depthColors[depthSteps[depthSteps.length - 1]]);
  }


  initLegend = () => {
    const legendContainer = document.getElementById('legendContainer');

    const depthSteps = this.depthSteps;

    // 上が浅い水深になるように逆順にループ
    for (let i = depthSteps.length - 1; i >= 0; i--) {
      const depth = depthSteps[i];

      // 水深に応じた色を取得
      const color = this.getDepthColor(depth);

      // divを作成
      const legendItem = document.createElement('div');

      // divにCSSクラス legend-item を設定
      legendItem.className = 'legend-item';

      // 水深に応じたidを設定して、あとから取り出せるようにする
      legendItem.id = `legend-${depth}`;

      const colorBox = document.createElement('div');
      colorBox.className = 'legend-color';
      colorBox.style.backgroundColor = `#${color.getHexString()}`;

      const label = document.createElement('span');
      label.textContent = `${depth}m`;

      legendItem.appendChild(colorBox);
      legendItem.appendChild(label);
      legendContainer.appendChild(legendItem);
    }
  }


  updateLegendHighlight = (depth) => {
    this.clearLegendHighlight();

    const depthSteps = this.depthSteps;

    // depthに最も近いdepthStepsの値を見つける
    let closestDepth = depthSteps[0];
    let minDiff = Math.abs(depth - closestDepth);
    for (let i = 1; i < depthSteps.length; i++) {
      const diff = Math.abs(depth - depthSteps[i]);
      if (diff < minDiff) {
        closestDepth = depthSteps[i];
        minDiff = diff;
      }
    }

    const legendItem = document.getElementById(`legend-${closestDepth}`);
    if (legendItem) {
      legendItem.classList.add('highlight');
    }

  }


  clearLegendHighlight = () => {
    const highlightedItems = document.querySelectorAll('.highlight');
    highlightedItems.forEach(item => item.classList.remove('highlight'));
  }


  createShapesFromTopojson = (topojsonData, objectName) => {

    // Shapeを格納する配列
    const shapes = [];

    // GeoJSONに変換
    const geojsonData = topojson.feature(topojsonData, topojsonData.objects[objectName]);

    // FeatureCollectionからFeatureを取り出す
    const features = geojsonData.features;

    // featureを一つずつ取り出す
    features.forEach(feature => {

      // featureのGeometryタイプがLineStringの場合
      if (feature.geometry.type === 'LineString') {
        const shape = new THREE.Shape();

        const coordinates = feature.geometry.coordinates;

        let coord;
        coord = coordinates[0];
        coord = this.normalizeCoordinates(coord);

        // パスを開始
        shape.moveTo(
          coord[0],
          coord[1]
        );

        for (let i = 1; i < coordinates.length; i++) {
          coord = coordinates[i];
          coord = this.normalizeCoordinates(coord);

          // 線分を追加
          shape.lineTo(
            coord[0],
            coord[1]
          );
        }

        shapes.push(shape);
      }

      // featureのGeometryタイプがPolygonの場合
      else if (feature.geometry.type === 'Polygon') {
        const shape = new THREE.Shape();

        const coordinates = feature.geometry.coordinates[0];

        let coord;
        coord = coordinates[0];
        coord = this.normalizeCoordinates(coord);

        shape.moveTo(
          coord[0],
          coord[1]
        );

        for (let i = 1; i < coordinates.length; i++) {
          coord = coordinates[i];
          coord = this.normalizeCoordinates(coord);
          shape.lineTo(
            coord[0],
            coord[1]
          );
        }

        shapes.push(shape);
      }

      // featureのGeometryタイプがMultiPolygonの場合
      else if (feature.geometry.type === 'MultiPolygon') {
        feature.geometry.coordinates.forEach(polygon => {
          const shape = new THREE.Shape();
          const coordinates = polygon[0];

          let coord;
          coord = coordinates[0];
          coord = this.normalizeCoordinates(coord);

          shape.moveTo(
            coord[0],
            coord[1]
          );

          for (let i = 1; i < coordinates.length; i++) {
            coord = coordinates[i];
            coord = this.normalizeCoordinates(coord);
            shape.lineTo(
              coord[0],
              coord[1]
            );
          }

          shapes.push(shape);
        });
      }

    });

    return shapes;
  }


  createMeshFromShapes = (shapes) => {
    // ExtrudeGeometryに渡すdepthパラメータ（厚み）
    const depth = 1.0;

    // ExtrudeGeometryで厚みを持たせる
    const geometry = new THREE.ExtrudeGeometry(shapes, {
      depth: depth,
      bevelEnabled: true,   // エッジを斜めにする
      bevelSize: 0.5,       // 斜めのサイズ
      bevelThickness: 0.5,  // 斜めの厚み
      bevelSegments: 1,     // 斜めのセグメント数
    });

    // XZ平面化
    // 回転の向きに注意！
    // Lat方向（Z軸方向）の座標をマイナスに正規化しているので、奥側に倒すように回転させる
    // つまり、画面には裏面が見えている
    geometry.rotateX(Math.PI / 2);

    // 地図は大きすぎるので海底地形図の倍の大きさになるようにクリッピングする
    const clippingSize = this.params.xzGridSize;

    // マテリアル、ここでは適当にMeshStandardMaterialを使う
    const material = new THREE.MeshStandardMaterial({
      color: 0xf0f0f0,

      // 透明にしない
      // transparent: true,
      // depthWrite: false,
      // opacity: 0.9,

      // クリッピングして表示領域を制限する
      clippingPlanes: [
        new THREE.Plane(new THREE.Vector3(0, 0, 1), clippingSize * 2),   // Z座標がxzGridSize * 2以下を表示
        new THREE.Plane(new THREE.Vector3(0, 0, -1), clippingSize * 2),  // Z座標が-xzGridSize * 2以上を表示
        new THREE.Plane(new THREE.Vector3(-1, 0, 0), clippingSize),      // X座標がxzGridSize以下を表示
        new THREE.Plane(new THREE.Vector3(1, 0, 0), clippingSize),       // X座標が-xzGridSize以上を表示
        new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),                  // Y座標が0以上を表示
      ],
    });

    // メッシュ化
    const mesh = new THREE.Mesh(geometry, material);

    // シーンに追加
    this.scene.add(mesh);
  }


  renderDepth = () => {
    // 前フレーム時点でのマウス位置から変わっていないなら処理をスキップ
    if (this.mousePosition.equals(this.previousMousePosition)) {
      return;
    }

    // 前フレーム時点のマウス位置を更新
    this.previousMousePosition.copy(this.mousePosition);

    // レイキャストを使用してマウスカーソルの位置を取得
    this.raycaster.setFromCamera(this.mousePosition, this.camera);

    // シーン全体を対象にレイキャストを行う
    // const intersects = this.raycaster.intersectObject(this.terrainMesh);
    const intersects = this.raycaster.intersectObject(this.scene, true);

    if (intersects.length > 0) {
      const intersect = intersects[0];

      // 水深データを取得
      const depth = intersect.point.y;

      // 緯度経度を取得
      const x = intersect.point.x;
      const z = intersect.point.z;
      const [lon, lat] = this.inverseNormalizeCoordinates(x, z);

      if (depth < 0) {
        this.depthContainer.textContent = `Depth: ${depth.toFixed(1)}m`;
        this.coordinatesContainer.textContent = `${lat.toFixed(8)}, ${lon.toFixed(8)}`;

        // 凡例をハイライト
        this.updateLegendHighlight(depth);
      } else {
        this.depthContainer.textContent = '';
        this.coordinatesContainer.textContent = '';

        // ハイライトをクリア
        this.clearLegendHighlight();
      }
    } else {
      this.depthContainer.textContent = '';
      this.coordinatesContainer.textContent = '';

      // ハイライトをクリア
      this.clearLegendHighlight();
    }
  }


  initCompass = () => {
    // 方位磁針を表示するコンテナ要素
    this.compassContainer = document.getElementById('compassContainer');
    this.compassContainer.style.display = this.params.showCompass ? 'block' : 'none';

    // 方位磁針を表示する要素
    this.compassElement = document.getElementById('compass');

    // カメラの向きを保存するVector3
    this.cameraDirection = new THREE.Vector3();

    // 前フレーム時点のカメラの向きを保存するVector3
    this.previousCameraDirection = new THREE.Vector3();
  }


  renderCompass = () => {
    if (this.params.showCompass === false || !this.compassElement) {
      return;
    }

    // カメラの向きを取得
    this.camera.getWorldDirection(this.cameraDirection);

    // 前フレーム時点のカメラの向きと変更がなければ処理をスキップ
    if (this.cameraDirection.equals(this.previousCameraDirection)) {
      return;
    }

    // 前回のカメラの向きを更新
    this.previousCameraDirection.copy(this.cameraDirection);

    // カメラの方向を方位磁針の回転角度に変換
    const angle = Math.atan2(this.cameraDirection.x, this.cameraDirection.z);
    const degrees = THREE.MathUtils.radToDeg(angle) + 180;  // 0度が北を向くように調整

    // 方位磁針の回転を更新
    this.compassElement.style.transform = `rotate(${degrees}deg)`;
  }


  initLandmarks = () => {
    const LAYER = 1;  // ランドマークを表示するレイヤー

    this.params.landmarks.forEach((landmark) => {

      // 正規化した緯度経度を取得
      let [lon, lat] = this.normalizeCoordinates([landmark.lon, landmark.lat]);

      // Y座標はその場所の真上になるように設定
      const position = new THREE.Vector3(lon, this.params.labelY, lat);

      // CSS2DRendererを使用してラベルを作成
      const div = document.createElement('div');
      div.className = 'landmark-label';
      div.textContent = (navigator.language.startsWith('ja') && 'name_ja' in landmark) ? landmark.name_ja : landmark.name || '';
      const cssObject = new CSS2DObject(div);

      cssObject.position.copy(position);
      cssObject.layers.set(LAYER);
      // cssObject.center.x = 0;  // 1にすると右に寄る
      cssObject.center.y = 1;     // 1にすると上に寄る
      this.scene.add(cssObject);

      // ラインを作成
      const material = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.5,
        depthWrite: false,
      });
      const points = [];
      points.push(new THREE.Vector3(lon, landmark.depth, lat));
      points.push(new THREE.Vector3(lon, position.y, lat));
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geometry, material);

      // ラインを表示するレイヤーを設定
      line.layers.set(LAYER);

      // 初期状態で表示するかどうか
      if (this.params.showLandmarks) {
        this.camera.layers.enable(LAYER);
      } else {
        this.camera.layers.disable(LAYER);
      }

      this.scene.add(line);
    });
  }


  initScale = () => {
    const LAYER = 2;  // 縮尺を表示するレイヤー

    const EARTH_RADIUS_KM = 6371; // 地球の半径（km）
    const kmToRadians = 1 / (EARTH_RADIUS_KM * Math.PI / 180); // 1kmをラジアンに変換
    const kmToDisplayLength = kmToRadians * this.params.xzScale;

    // ラインのマテリアル
    const material = new THREE.LineBasicMaterial({ color: 0xffffff });

    // 開始点は原点からちょっと浮かせる（地図が厚みを持っているため）
    const start = new THREE.Vector3(0, 1.1, 0);

    // 横線を追加
    {
      const end = new THREE.Vector3(kmToDisplayLength, 1.1, 0);
      const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
      const line = new THREE.Line(geometry, material);
      line.layers.set(LAYER);
      this.scene.add(line);

      // 100mごとの目印を追加
      for (let i = 1; i <= 10; i++) {
        const markerPosition1 = new THREE.Vector3(i * kmToDisplayLength / 10, 1.1, 0);
        const markerPosition2 = new THREE.Vector3(markerPosition1.x, markerPosition1.y + 1, markerPosition1.z)
        const markerGeometry = new THREE.BufferGeometry().setFromPoints([markerPosition1, markerPosition2]);
        const markerLine = new THREE.Line(markerGeometry, material);
        markerLine.layers.set(LAYER);
        this.scene.add(markerLine);
      }

      // ラベルを追加
      const div = document.createElement('div');
      div.className = 'scale-label';
      div.textContent = '1km';
      const cssObject = new CSS2DObject(div);

      cssObject.position.copy(end);
      cssObject.position.y = 4;
      cssObject.layers.set(LAYER);
      this.scene.add(cssObject);
    }


    // 縦線を追加
    {
      const end = new THREE.Vector3(0, 1.1, kmToDisplayLength);
      const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
      const line = new THREE.Line(geometry, material);
      line.layers.set(LAYER);
      this.scene.add(line);

      // 100mごとの目印を追加
      for (let i = 1; i <= 10; i++) {
        const markerPosition1 = new THREE.Vector3(0, 1.1, i * kmToDisplayLength / 10);
        const markerPosition2 = new THREE.Vector3(markerPosition1.x, markerPosition1.y + 1, markerPosition1.z)
        const markerGeometry = new THREE.BufferGeometry().setFromPoints([markerPosition1, markerPosition2]);
        const markerLine = new THREE.Line(markerGeometry, material);
        markerLine.layers.set(LAYER);
        this.scene.add(markerLine);
      }

      // ラベルを追加
      const div = document.createElement('div');
      div.className = 'scale-label';
      div.textContent = '1km';
      const cssObject = new CSS2DObject(div);
      cssObject.position.copy(end);
      cssObject.position.y = 4;
      cssObject.layers.set(LAYER);
      this.scene.add(cssObject);
    }

    if (this.params.showScale) {
      this.camera.layers.enable(LAYER);
    } else {
      this.camera.layers.disable(LAYER);
    }

  }

}
