import * as THREE from "three";
import { OrbitControls } from "three/controls/OrbitControls.js";

// lil-gui
import { GUI } from "three/libs/lil-gui.module.min.js";

// stats.js
import Stats from "three/libs/stats.module.js";

// SelectionBox
import { SelectionBox } from "three/libs/interactive/SelectionBox.js";
import { SelectionHelper } from "three/libs/interactive/SelectionHelper.js";


export class Main {

  container;

  sizes = {
    width: 0,
    height: 0
  }

  scene;
  camera;
  renderer;
  controller;

  statsjs;

  renderParams = {
    clock: new THREE.Clock(),
    delta: 0,
    interval: 1 / 30,  // = 30fps
  }

  params = {
    path: "./static/data/depth_map_data.csv",
    csvData: null,

    autoRotate: false,
    autoRotateSpeed: 1.0,
  }

  pointCloud;

  constructor(params={}) {
    this.params = Object.assign(this.params, params);
    this.init();
  }


  async init() {
    // データを読み込む
    // その間、ローディング画面を表示する
    await this.loadCsv(this.params.path);

    if (this.params.csvData === null) {
      return;
    }

    console.log(this.params.csvData);
    // {lat: xxx, lon: xxx, depth: xxx}

    // scene, camera, renderer, controllerを初期化
    this.initThreejs();

    // lil-guiを初期化
    this.initGui();

    // stats.jsを初期化
    this.initStatsjs();

    // CSVデータを元に点群を初期化
    this.initPointCloud();

    // Selectionを初期化
    this.initSelectionBox();

    // ダウンロードボタンを初期化
    this.initDownloadButton();

    // フレーム毎の処理(requestAnimationFrameで再帰的に呼び出される)
    this.render();
  }


  async loadCsv(path) {
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

      // 瞬時にfetchできても0.5秒間はローディング画面を表示する
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


  // CSV形式のテキストをパース
  parseCsv(text) {
    const data = [];

    const lines = text.split('\n');
    const headers = lines[0].split(',');
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(',');
      if (row.length === headers.length) {
        const rowData = {};
        for (let j = 0; j < headers.length; j++) {
          rowData[headers[j].trim()] = row[j].trim();
        }
        this.normalizeData(rowData);
        data.push(rowData);
      }
    }

    return data;
  }


  normalizeData(rowData) {

    // |       |             lat |             lon |        depth |
    // |:------|----------------:|----------------:|-------------:|
    // | mean  |     35.1641     |    139.608      |     16.4776  |

    // 小数点以下を消すなら、このスケールになるんだけど、とんでもなくでかい数字になるので
    // const scale = 100000000000000;

    // このくらいがちょうど良さそう
    const scale = 10000;

    const latMean = 35.1641;
    const lonMean = 139.608;

    rowData.nLat = -1 * (parseFloat(rowData.lat) - latMean) * scale;
    rowData.nLon = (parseFloat(rowData.lon) - lonMean) * scale;
    rowData.nDepth = -1 * parseFloat(rowData.depth);

  }


  initThreejs() {
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
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.sizes.width, this.sizes.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // コントローラ
    this.controller = new OrbitControls(this.camera, this.renderer.domElement);
    this.controller.autoRotate = this.params.autoRotate;
    this.controller.autoRotateSpeed = this.params.autoRotateSpeed;
    // 一時停止
    // this.controller.enabled = false;

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

    this.scene.add(new THREE.AmbientLight(0xffffff));

  }

  initGui() {
    const guiContainer = document.getElementById("guiContainer");
    const gui = new GUI({
      container: guiContainer,
      width: 300,
    });

    gui
      .add(this.params, "autoRotate")
      .name("rotation")
      .onChange((value) => {
        this.controller.autoRotate = value;
      });
  }


  initStatsjs() {
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
    requestAnimationFrame(this.render);

    this.renderParams.delta += this.renderParams.clock.getDelta();
    if (this.renderParams.delta < this.renderParams.interval) {
      return;
    }

    {
      // stats.jsを更新
      this.statsjs.update();

      // カメラコントローラーの更新
      this.controller.update();

      // 再描画
      this.renderer.render(this.scene, this.camera);
    }

    this.renderParams.delta %= this.renderParams.interval;
  }


  onWindowResize = (event) => {
    this.sizes.width = this.container.clientWidth;
    this.sizes.height = this.container.clientHeight;

    this.camera.aspect = this.sizes.width / this.sizes.height;
    this.camera.updateProjectionMatrix();

    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(this.sizes.width, this.sizes.height);
  };


  initPointCloud() {

    // データを取り出す
    const dataList = this.params.csvData;

    // データの入れ物
    const positionArray = new Float32Array(dataList.length * 3);
    const csvArray = new Float32Array(dataList.length * 3);
    const colorArray = new Float32Array(dataList.length * 3);
    const removedArray = new Float32Array(dataList.length);

    dataList.forEach((obj, index) => {

      // 位置情報
      positionArray[index * 3 + 0] = obj.nLon;    // 経度
      positionArray[index * 3 + 1] = obj.nDepth;  // 水深
      positionArray[index * 3 + 2] = obj.nLat;    // 緯度

      // 元データ
      csvArray[index * 3 + 0] = obj.lon;
      csvArray[index * 3 + 1] = obj.depth;
      csvArray[index * 3 + 2] = obj.lat;

      // 色
      colorArray[index * 3 + 0] = 0.6;
      colorArray[index * 3 + 1] = 0.8;
      colorArray[index * 3 + 2] = 1.0;

      // 削除フラグ
      removedArray[index] = 0.0;
    });

    // THREE.InstancedBufferGeometry を使ってポイントクラウドを描画する
    // ポイントクラウドのように同じ図形を多数描画する場合、
    // GPUに対して同じ図形を描画せよ、と指示することで高速化できる
    const geometry = new THREE.InstancedBufferGeometry();

    // 元になるジオメトリを一つ作成
    // Boxが最も高速だけど、粗く見える
    // Sphereもセグメント数を減らすと高速になる
    // const originGeometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
    const originGeometry = new THREE.SphereGeometry(0.1, 6, 6);

    // シェーダーで必要になるパラメータを追加しておく
    geometry.setAttribute("position", originGeometry.attributes.position.clone());
    geometry.setAttribute("normal", originGeometry.attributes.normal.clone());
    geometry.setAttribute("uv", originGeometry.attributes.uv.clone());
    if (originGeometry.index) {
      geometry.setIndex(originGeometry.index.clone());
    }

    // 個々に設定するアトリビュートを追加
    geometry.setAttribute("instancePosition", new THREE.InstancedBufferAttribute(positionArray, 3));
    geometry.setAttribute("instanceColor", new THREE.InstancedBufferAttribute(colorArray, 3));
    geometry.setAttribute("removed", new THREE.InstancedBufferAttribute(removedArray, 1));
    geometry.setAttribute("csv", new THREE.InstancedBufferAttribute(csvArray, 3));

    const vertex = /* glsl */ `
      attribute vec3 instancePosition;
      attribute vec3 instanceColor;
      attribute float removed;

      varying vec3 vColor;

      void main() {
        // vec4 modelPosition = modelMatrix * vec4(position, 1.0);
        vec4 modelPosition = modelMatrix * vec4(position + instancePosition, 1.0);
        vec4 viewPosition = viewMatrix * modelPosition;
        vec4 projectionPosition = projectionMatrix * viewPosition;

        gl_Position = projectionPosition;

        if (removed > 0.0) {
          vColor = vec3(0.0, 0.0, 0.0);
        } else {
          vColor =  instanceColor;
        }
      }
    `;

    const fragment = /* glsl */ `
      varying vec3 vColor;

      void main() {
        gl_FragColor = vec4(vColor, 1.0);
      }
    `;

    const material = new THREE.ShaderMaterial({
      vertexShader: vertex,
      fragmentShader: fragment,
      uniforms: {},
      vertexColors: false,
      transparent: false,
      depthTest: false,
    });

    this.pointCloud = new THREE.InstancedMesh(geometry, material, dataList.length);

    for (let index = 0; index < dataList.length; index++) {
      const matrix = new THREE.Matrix4();
      matrix.setPosition(
        positionArray[index * 3 + 0],
        positionArray[index * 3 + 1],
        positionArray[index * 3 + 2]
      );
      this.pointCloud.setMatrixAt(index, matrix);
    };

    this.scene.add(this.pointCloud);

    console.log(this.pointCloud);

    // アトリビュートの更新を通知
    // geometry.attributes.instancePosition.needsUpdate = true;

  }


  initSelectionBox() {
    const selectionBox = new SelectionBox(this.camera, this.pointCloud);
    selectionBox.enabled = false;

    // 第三引数のselectBoxはCSSと合わせる
    const selectionHelper = new SelectionHelper(this.renderer, 'selectBox');
    selectionHelper.enabled = false;

    const onPointerDown = (event) => {
      const element = event.currentTarget;

      selectionBox.startPoint.set(
        (event.clientX / element.clientWidth) * 2 - 1,
        -(event.clientY / element.clientHeight) * 2 + 1,
        0.5
      );
    };

    const onPointerMove = (event) => {
      if (selectionHelper.isDown) {
        const element = event.currentTarget;

        selectionBox.endPoint.set(
          (event.clientX / element.clientWidth) * 2 - 1,
          -(event.clientY / element.clientHeight) * 2 + 1,
          0.5
        );
      }
    };

    const onPointerUp = (event) => {
      const element = event.currentTarget;

      selectionBox.endPoint.set(
        (event.clientX / element.clientWidth) * 2 - 1,
        -(event.clientY / element.clientHeight) * 2 + 1,
        0.5
      );

      selectionBox.select();
      {
        const instances = selectionBox.instances;
        const indeces = Object.values(instances)[0];

        if (indeces.length > 0) {
          console.log(indeces);
          for (let i = 0; i < indeces.length; i++) {
            const index = indeces[i];
            this.pointCloud.geometry.attributes.removed.array[index] = 1.0;
            this.pointCloud.geometry.attributes.removed.needsUpdate = true;
          }
        }

      }
      selectionBox.instances = {};

    }

    // Selectionを有効にするボタン
    const enableButton = document.getElementById('enableSelectionBox');
    enableButton.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();

      selectionBox.enabled = true;
      selectionHelper.enabled = true;

      // マウスイベントを追加
      this.renderer.domElement.addEventListener('pointerdown', onPointerDown);
      this.renderer.domElement.addEventListener('pointermove', onPointerMove);
      this.renderer.domElement.addEventListener('pointerup', onPointerUp);

      // OrbitControlsを無効にする
      this.controller.enabled = false;
    });

    // Selectionを無効にするボタンを追加
    const disableButton = document.getElementById('disableSelectionBox');
    disableButton.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();

      selectionBox.enabled = false;
      selectionHelper.enabled = false;

      // マウスイベントを削除
      this.renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      this.renderer.domElement.removeEventListener('pointermove', onPointerMove);
      this.renderer.domElement.removeEventListener('pointerup', onPointerUp);

      // OrbitControlsを有効にする
      this.controller.enabled = true;
    });

  }

  initDownloadButton() {
    const downloadButton = document.getElementById('downloadButton');
    downloadButton.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();

      // ダウンロードするデータを作成
      const header = "lat,lon,depth";
      let text = header + "\n";
      for (let i = 0; i < this.pointCloud.geometry.attributes.removed.array.length; i++) {
        if (this.pointCloud.geometry.attributes.removed.array[i] > 0.0) {
          continue;
        }
        const lon = this.pointCloud.geometry.attributes.csv.array[i * 3 + 0];
        const depth = this.pointCloud.geometry.attributes.csv.array[i * 3 + 1];
        const lat = this.pointCloud.geometry.attributes.csv.array[i * 3 + 2];

        text += `${lat},${lon},${depth}\n`;
      }

      // ダウンロードを実行
      this.textDownload(text, "depth_map_data_edited.csv");
    });
  }

  textDownload = (text, fileName, mimeType = 'text/plain', withBOM = false) => {
    let data = [];

    // BOM付きにしたい場合
    if (withBOM) {
      data.push(new Uint8Array([0xef, 0xbb, 0xbf]));
    }

    data.push(text);

    const blob = new Blob(data, { type: mimeType });

    // a要素の生成
    const anchorElem = document.body.appendChild(document.createElement('a'));
    anchorElem.setAttribute('href', URL.createObjectURL(blob));
    anchorElem.setAttribute('target', '_blank');
    anchorElem.setAttribute('download', fileName);
    anchorElem.style.visibility = 'hidden';

    // イベント実行
    anchorElem.dispatchEvent(new MouseEvent('click', {
      view: window,
      bubbles: false,
      cancelable: false
    }));

    document.body.removeChild(anchorElem);
  };

}
