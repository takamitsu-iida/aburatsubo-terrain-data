import * as THREE from "three";
import { OrbitControls } from "three/controls/OrbitControls.js";

// lil-gui
import { GUI } from "three/libs/lil-gui.module.min.js";

// stats.js
import Stats from "three/libs/stats.module.js";

// デローネ三角形
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

  container;

  sizes = {
    width: 0,
    height: 0
  }

  scene;
  uiScene
  camera;
  uiCamera;
  renderer;
  controller;

  statsjs;

  renderParams = {
    animationId: null,
    clock: new THREE.Clock(),
    delta: 0,
    interval: 1 / 30,  // = 30fps
  }

  params = {
    // path: "./static/data/depth_map_data.csv",
    path: "./static/data/depth_map_data_edited.csv",

    csvData: null,

    // ポイントクラウドを表示するか？
    showPointCloud: true,

    // ポイントクラウドのサイズ
    pointSize: 0.2,

    // ワイヤーフレーム表示にする？
    wireframe: false,

    // コントローラの設定
    autoRotate: true,
    autoRotateSpeed: 1.0,

    // 緯度経度の最大値、最小値（CSVから自動で読み取る）
    minLat: 0,
    maxLat: 0,
    minLon: 0,
    maxLon: 0,

    // 四分木の分割パラメータ
    divideParam: 5,  // 見栄えに変化がないので5のままでよい（guiはdisableにしておく）
    maxPoints: 5,

    // 四分木に分割した領域の配列
    areas: null,
  }

  // 地形図のポイントクラウド（guiで表示を操作するためにインスタンス変数にする）
  pointMesh;

  // デローネ三角形で作成する地形図のメッシュ（guiで表示を操作するためにインスタンス変数にする）
  terrainMesh;


  constructor(params = {}) {
    this.params = Object.assign(this.params, params);
    this.init();
  }


  init = async () => {
    // データを読み込む
    // その間、ローディング画面を表示する
    await this.loadCsv(this.params.path);

    if (this.params.csvData === null) {
      return;
    }

    // console.log(this.params.csvData);
    // {lat: xxx, lon: xxx, depth: xxx}

    // scene, camera, renderer, controllerを初期化
    this.initThreejs();

    // stats.jsを初期化
    this.initStatsjs();

    // lil-guiを初期化
    this.initGui();

    // コンテンツを初期化
    this.initContents();
  }


  initContents = () => {
    // アニメーションを停止
    this.stop();

    // シーン上のメッシュを削除する
    this.clearScene();

    // 削除した状態を描画
    this.renderer.render(this.scene, this.camera);

    // CSVデータを四分木に分割してareas配列を作成する
    this.initQuadTree();

    // 四分木で領域分割したareas配列を元に、デローネ三角形でメッシュを表示する
    this.initDelaunay();

    // フレーム毎の処理
    this.render();
  }


  loadCsv = async (path) => {
    const loadingContainer = document.getElementById('loadingContainer');

    try {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(`HTTP status: ${response.status}`);
      }

      // テキストデータを取得
      const text = await response.text();

      // CSVデータをパース
      this.params.csvData = this.parseCsv(text);

      // 瞬時にfetchできても0.5秒はローディング画面を表示する
      const interval = setInterval(() => {
        loadingContainer.classList.add('fadeout');
        clearInterval(interval);
      }, 500);

      // ローディング画面を非表示にする
      loadingContainer.addEventListener('transitionend', (event) => {
        event.target.remove();
      });

    } catch (error) {
      const errorMessage = `Error while loading CSV: ${error}`;
      console.error(errorMessage);
      let p = document.createElement('p');
      p.textContent = errorMessage;
      p.style.color = 'white';
      loadingContainer.appendChild(p);
    }

  }


  // CSVデータをパースする関数
  parseCsv = (text) => {
    // 行に分割
    const lines = text.split('\n').filter(line => line.trim().length > 0);

    // 先頭行が数字で始まる場合はヘッダ無しと判定
    const firstLine = lines[0].trim();
    const isHeader = isNaN(Number(firstLine.split(',')[0]));
    const headers = isHeader ? lines[0].split(',').map(h => h.trim()) : ["lat", "lon", "depth"];

    // データ開始行のインデックス
    const startIdx = isHeader ? 1 : 0;

    const data = [];

    let minLat = 9999;
    let maxLat = -9999;
    let minLon = 9999;
    let maxLon = -9999;

    for (let i = startIdx; i < lines.length; i++) {
      const row = lines[i].split(',');
      if (row.length === headers.length) {
        const rowData = {};
        for (let j = 0; j < headers.length; j++) {
          rowData[headers[j].trim()] = row[j].trim();
        }
        this.normalizeData(rowData);
        data.push(rowData);

        // 緯度経度の最大値、最小値を取得
        if (rowData.lat < minLat) {
          minLat = rowData.lat;
        }
        if (rowData.lat > maxLat) {
          maxLat = rowData.lat;
        }
        if (rowData.lon < minLon) {
          minLon = rowData.lon;
        }
        if (rowData.lon > maxLon) {
          maxLon = rowData.lon;
        }
      }
    }

    console.log(`minLat: ${minLat}\nmaxLat: ${maxLat}\nminLon: ${minLon}\nmaxLon: ${maxLon}`);

    this.params.minLat = minLat;
    this.params.maxLat = maxLat;
    this.params.minLon = minLon;
    this.params.maxLon = maxLon;

    return data;
  }


  normalizeData = (rowData) => {

    // |       |             lat |             lon |        depth |
    // |:------|----------------:|----------------:|-------------:|
    // | mean  |     35.1641     |    139.608      |     16.4776  |

    // 小数点以下を消すなら、このスケールになるんだけど、とんでもなくでかい数字になるので
    // const scale = 100000000000000;

    // このくらいがちょうど良さそう
    const scale = 10000;

    const latMean = 35.1641;
    const lonMean = 139.608;

    rowData.lat = -1 * (parseFloat(rowData.lat) - latMean) * scale;
    rowData.lon = (parseFloat(rowData.lon) - lonMean) * scale;
    rowData.depth = -1 * parseFloat(rowData.depth);

  }


  initThreejs = () => {
    // コンテナ
    this.container = document.getElementById("threejsContainer");

    // コンテナのサイズ
    this.sizes.width = this.container.clientWidth;
    this.sizes.height = this.container.clientHeight;

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
    this.camera.position.set(0, 0, 100);

    // レンダラ
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
    });
    this.renderer.setSize(this.sizes.width, this.sizes.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // 凡例表示のためにautoClearをfalseにする
    this.renderer.autoClear = false;
    // 背景を黒に設定
    this.renderer.setClearColor(0x000000, 1);
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
    const axesHelper = new THREE.AxesHelper(10000);
    this.scene.add(axesHelper);

    // 環境光
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.5));

    // ディレクショナルライト
    const light = new THREE.DirectionalLight(0xffffff, 0.8);
    light.position.set(-50, 0, 0);
    this.scene.add(light);

  }


  initGui = () => {
    const guiContainer = document.getElementById("guiContainer");
    const gui = new GUI({
      container: guiContainer,
      width: 300,
    });

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

    gui
      .add(this.params, "autoRotate")
      .name(navigator.language.startsWith("ja") ? "自動回転" : "rotation")
      .onChange((value) => {
        this.controller.autoRotate = value;
      });

    gui
      .add(this.params, "autoRotateSpeed")
      .name(navigator.language.startsWith("ja") ? "回転スピード" : "autoRotateSpeed")
      .min(1.0)
      .max(10.0)
      .step(0.1)
      .onChange((value) => {
        this.controller.autoRotateSpeed = value;
      });

    gui
      .add(this.params, "wireframe")
      .name(navigator.language.startsWith("ja") ? "ワイヤーフレーム表示" : "wireframe")
      .onChange(() => {
        this.terrainMesh.material.wireframe = this.params.wireframe;
      });

    gui
    .add(this.params, "showPointCloud")
      .name(navigator.language.startsWith("ja") ? "ポイントクラウド表示" : "showPointCloud")
      .onChange((value) => {
        this.pointMesh.visible = value;
      });

    gui
      .add(this.params, "pointSize")
      .name(navigator.language.startsWith("ja") ? "ポイントサイズ" : "pointSize")
      .min(0.1)
      .max(1.0)
      .step(0.1)
      .onChange((value) => {
        this.pointMesh.material.size = value;
      });

    gui
      .add(this.params, "divideParam")
      .name(navigator.language.startsWith("ja") ? "四分木分割パラメータ" : "divideParam")
      .min(1)
      .max(10)
      .step(1)
      .onFinishChange(() => {
        doLater(this.initContents, 100);
      })
      .disable();  // このパラメータは分割数に影響するが、見栄えには変化がないのでdisableにしておく

    gui
      .add(this.params, "maxPoints")
      .name(navigator.language.startsWith("ja") ? "四分木分割しきい値" : "maxPoints")
      .min(1)
      .max(10)
      .step(1)
      .onFinishChange(() => {
        doLater(this.initContents, 100);
      });

  }


  initStatsjs = () => {
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
      this.statsjs.update();

      // カメラコントローラーの更新
      this.controller.update();

      // 凡例表示のためにautoClearをfalseにしているので、手動でクリア操作を行う
      this.renderer.clear();

      // 再描画
      this.renderer.render(this.scene, this.camera);
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
      if (child.type === 'AxesHelper' || child.type === 'GridHelper' || String(child.type).indexOf('Light') >= 0 ) {
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
  };


  quadtree = (area, maxpoints, maxdivision) => {
    if (area.points.length <= maxpoints || area.depth >= maxdivision) {
      return [area];
    }

    const midX = (area.minLon + area.maxLon) / 2;
    const midZ = (area.minLat + area.maxLat) / 2;

    const areas = [
      new Area(area.minLon, area.minLat, midX, midZ, area.depth + 1),
      new Area(midX, area.minLat, area.maxLon, midZ, area.depth + 1),
      new Area(area.minLon, midZ, midX, area.maxLat, area.depth + 1),
      new Area(midX, midZ, area.maxLon, area.maxLat, area.depth + 1)
    ];

    for (const point of area.points) {
      for (const subArea of areas) {
        if (point.lat >= subArea.minLat && point.lat < subArea.maxLat && point.lon >= subArea.minLon && point.lon < subArea.maxLon) {
          subArea.append(point);
          break;
        }
      }
    }

    let result = [];
    for (const subArea of areas) {
      result = result.concat(this.quadtree(subArea, maxpoints, maxdivision));
    }

    return result;
  }


  initQuadTree = () => {
    // X軸 = Longitude(経度)
    const minLon = this.params.minLon;  // -93.15585150687866
    const maxLon = this.params.maxLon;  // 137.8217825273964

    // Z軸 = Latitude(緯度)
    const minLat = this.params.minLat;  // -96.32579062369473
    const maxLat = this.params.maxLat;  // 68.51506401226004

    // 四分木の分割パラメータを調整する
    const divideParam = this.params.divideParam;
    const maxDivision = Math.ceil(Math.max((maxLon - minLon) / divideParam, (maxLat - minLat) / divideParam));

    console.log(`maxdivision: ${maxDivision}`);

    // 領域内に含まれる最大の点の数、これを超えていたらさらに小さく分割する
    const maxPoints = this.params.maxPoints;

    // 対象とする領域を生成
    const initial = new Area(minLon, minLat, maxLon, maxLat, 0);

    const data = this.params.csvData;
    // dataは配列で、各要素は以下のようなオブジェクト
    // [ {lat: 67.88624331335313, lon: -81.94761236723025, depth: -21.1785}, {...}, ... ]
    // 初期領域にデータを追加する
    for (const d of data) {
      initial.append(d);
    }

    // 四分木に分割する
    const areas = this.quadtree(initial, maxPoints, maxDivision);
    console.log(`initial area is divided into ${areas.length} areas`);

    // 保存しておく
    this.params.areas = areas;
  }


  initDelaunay = () => {

    // 四分木で分割した領域を一つの点とするポイントクラウドを作成する
    const areas = this.params.areas;

    const point3d = [];
    const colors = [];
    areas.forEach((area) => {
      // 点がない場合は無視する
      if (area.points.length < 1) {
        return;
      }
      // エリアの分割レベルが浅い部分は無視する
      if (area.depth < 5) {
        return;
      }

      // 頂点の座標を計算
      const lon = (area.minLon + area.maxLon) / 2;
      const lat = (area.minLat + area.maxLat) / 2;
      const depth = area.points.reduce((acc, cur) => acc + cur.depth, 0) / area.points.length;
      point3d.push(new THREE.Vector3(lon, depth, lat));

      // 深さに応じて頂点に色を付ける
      const color = this.getDepthColor(depth);
      colors.push(color.r, color.g, color.b);
    });

    console.log(`${point3d.length} points are created`);

    // ポイントクラウドのジオメトリを作成
    const geometry = new THREE.BufferGeometry().setFromPoints(point3d);

    // 頂点カラーを設定
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    // マテリアルを作成
    const pointsMaterial = new THREE.PointsMaterial({
      color: 0x99ccff,
      size: this.params.pointSize,
    });

    // 点群を作成
    const pointMesh = new THREE.Points(geometry, pointsMaterial);

    // 表示するかどうか
    pointMesh.visible = this.params.showPointCloud;

    // シーンに追加
    this.scene.add(pointMesh);

    // インスタンス変数に保存
    this.pointMesh = pointMesh;

    // デローネ三角形を形成する
    const delaunay = Delaunator.from(
      point3d.map(v => {
        return [v.x, v.z];
      })
    );

    // デローネ三角形のインデックスをmeshIndexに代入してThree.jsのインデックスに変換
    const meshIndex = [];
    for (let i = 0; i < delaunay.triangles.length; i++) {
      meshIndex.push(delaunay.triangles[i]);
    }

    // 点群のジオメトリにインデックスを追加してポリゴン化
    geometry.setIndex(meshIndex);

    // 法線ベクトルを計算
    geometry.computeVertexNormals();

    // マテリアルを生成
    const material = new THREE.MeshLambertMaterial({
      vertexColors: true, // 頂点カラーを使用
      wireframe: this.params.wireframe,
    });

    // メッシュを生成
    const terrainMesh = new THREE.Mesh(geometry, material);

    // シーンに追加
    this.scene.add(terrainMesh);

    // インスタンス変数に保存
    this.terrainMesh = terrainMesh;
  }


  getDepthColor(depth) {
    let color;
    if (depth < -60.0) {
      color = new THREE.Color(0x2e146a);
    } else if (depth < -55.0) {
      color = new THREE.Color(0x451e9f);
    } else if (depth < -50.0) {
      color = new THREE.Color(0x3b31c3);
    } else if (depth < -45.0) {
      color = new THREE.Color(0x1f47de);
    } else if (depth < -40.0) {
      color = new THREE.Color(0x045ef9);
    } else if (depth < -35.0) {
      color = new THREE.Color(0x0075fd);
    } else if (depth < -30.0) {
      color = new THREE.Color(0x008ffd);
    } else if (depth < -25.0) {
      color = new THREE.Color(0x01aafc);
    } else if (depth < -20.0) {
      color = new THREE.Color(0x01c5fc);
    } else if (depth < -16.0) {
      color = new THREE.Color(0x45ccb5);
    } else if (depth < -12.0) {
      color = new THREE.Color(0x90d366);
    } else if (depth < -10.0) {
      color = new THREE.Color(0xb4df56);
    } else if (depth < -8.0) {
      color = new THREE.Color(0xd9ed4c);
    } else if (depth < -6.0) {
      color = new THREE.Color(0xfdfb41);
    } else if (depth < -5.0) {
      color = new THREE.Color(0xfee437);
    } else if (depth < -4.0) {
      color = new THREE.Color(0xfecc2c);
    } else if (depth < -3.0) {
      color = new THREE.Color(0xfeb321);
    } else if (depth < -2.0) {
      color = new THREE.Color(0xff9b16);
    } else if (depth < -1.0) {
      color = new THREE.Color(0xff820b);
    } else if (depth < 0.0) {
      color = new THREE.Color(0xff7907);
    } else {
      color = new THREE.Color(0xffffff);
    }
    return color;
  }

}



class Area {
  // Lon = X軸
  minLon;
  maxLon;

  // Lat = Z軸
  minLat;
  maxLat;

  // 四分木の深さ
  depth;

  // この領域に含まれる点の配列
  // [{lat: xxx, lon: xxx, depth: xxx}, ...]
  points = [];

  constructor(minLon, minLat, maxLon, maxLat, depth) {
    this.minLon = minLon;
    this.minLat = minLat;
    this.maxLon = maxLon;
    this.maxLat = maxLat;
    this.depth = depth;
  }

  append(point) {
    this.points.push(point);
  }
}
