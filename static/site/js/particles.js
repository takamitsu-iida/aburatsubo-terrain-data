// from local
// import * as THREE from "../../build/three.module.js";
// import { OrbitControls } from "./controls/OrbitControls.js";

// from CDN
import * as THREE from 'three';
import { OrbitControls } from 'https://unpkg.com/three@0.142.0/examples/jsm/controls/OrbitControls.js';
// import GUI from "https://cdn.jsdelivr.net/npm/lil-gui@0.15/+esm";

// サイズ
const sizes = {
  width: window.innerWidth,
  height: window.innerHeight,
};

// DOMエレメント
const canvas = document.querySelector(".webgl");

// シーン
const scene = new THREE.Scene();

// テクスチャ設定
const textureLoader = new THREE.TextureLoader();
const particlesTexture = textureLoader.load("./site/img/particle.png")

// カメラ
const camera = new THREE.PerspectiveCamera(
  100,
  sizes.width / sizes.height,
  0.1,
  100
);
camera.position.set(1, 1, 2);
scene.add(camera);

// マウス操作のためのOrbitControls
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

// レンダラー
const renderer = new THREE.WebGLRenderer({
  "canvas": canvas,
});
renderer.setSize(sizes.width, sizes.height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

//
// グリッドヘルパー
//
const gridHelper = new THREE.GridHelper(100, 100);
scene.add(gridHelper);

//
// パーティクル
//

// ジオメトリを作成
const particlesGeometry = new THREE.BufferGeometry();
const count = 10000;

// (x, y, z)の値を持つのでcount *3で配列を作る
const positionArray = new Float32Array(count * 3);
const colorArray = new Float32Array(count * 3);

// 座標をランダムに設定
for (let i = 0; i < count * 3; i++) {
  positionArray[i] = (Math.random() - 0.5) * 10;
  colorArray[i] = Math.random();
}

// ジオメトリの"position"アトリビュートに位置座標の配列をセットする
// (x, y, z)の3軸なので3をセット
particlesGeometry.setAttribute(
  "position", new THREE.BufferAttribute(positionArray, 3)
);

// ジオメトリの"color"アトリビュートに色配列をセットする
particlesGeometry.setAttribute(
  "color", new THREE.BufferAttribute(colorArray, 3)
);

// マテリアル
const pointMaterial = new THREE.PointsMaterial({
  size: 0.15,
  alphaMap: particlesTexture,
  transparent: true,
  depthWrite: false,
  vertexColors: true, // 頂点に色を付ける、パーティクルの場合はこれ
  blending: THREE.AdditiveBlending, // 重なったところを光らせる
});

// 単色を付けるならこれでよい
// pointMaterial.color.set("green");

// メッシュ化（ジオメトリ＋マテリアル）
// Pointsはパーティクル専用のメッシュ
const particles = new THREE.Points(particlesGeometry, pointMaterial)
scene.add(particles)

//
// アニメーション
//

const clock = new THREE.Clock();

function animate() {
  const elapsedTime = clock.getElapsedTime();

  controls.update();

  // レンダリング
  renderer.render(scene, camera);

  // 再帰
  requestAnimationFrame(animate);
}

//
// ブラウザのリサイズ
//
function onWindowResize() {
  sizes.width = window.innerWidth;
  sizes.height = window.innerHeight;

  camera.aspect = sizes.width / sizes.height;
  camera.updateProjectionMatrix();  // アスペクト比を変えたときには必ず呼ぶ

  renderer.setSize(sizes.width, sizes.height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
}
window.addEventListener("resize", onWindowResize);

//
// 初期化
//
function init() {
  animate();
}
window.addEventListener("load", init);
