/**
 * GeoJSONベースの地図表示アプリケーション
 * ズームレベルに応じてマーカーとポリゴンを切り替えて表示
 *
 * @class Main
 * @example
 * const app = new Main({
 *   geojsonUrls: ['./data/area1.geojson', './data/area2.geojson'],
 *   zoomThreshold: 12
 * });
 * await app.initialize();
 */
export class Main {

  /**
   * デフォルト設定
   * @static
   * @type {Object}
   */
  static DEFAULT_CONFIG = {
    defaultLat: 35.6812,      // 東京駅の緯度
    defaultLon: 139.7671,     // 東京駅の経度
    defaultZoom: 13,
    zoomThreshold: 11,        // マーカー/ポリゴン切り替え閾値
    geojsonUrls: ['./data/ALL_depth_map_data_202510.geojson']
  };

  /**
   * スタイル設定
   * @static
   * @type {Object}
   */
  static STYLES = {
    boundary: {
      color: '#3388ff',
      weight: 2,
      opacity: 0.8,
      fillOpacity: 0.0
    },
    boundaryHover: {
      weight: 3,
      opacity: 1.0
    },
    contour: {
      color: '#00008B',
      weight: 1,
      opacity: 0.6,
      interactive: false
    },
    depthPolygon: {
      weight: 0.5,
      opacity: 0.4,
      fillOpacity: 0.6
    }
  };

  /**
   * 水深カラーマップ
   * 水深（メートル）に応じた色の定義
   * @static
   * @type {Array<{max: number, color: string}>}
   */
  static DEPTH_COLORS = [
    { max: 5, color: '#e6f3ff' },
    { max: 10, color: '#99d6ff' },
    { max: 20, color: '#4db8ff' },
    { max: 30, color: '#0099ff' },
    { max: 50, color: '#0066cc' },
    { max: 100, color: '#004d99' },
    { max: Infinity, color: '#003366' }
  ];

  /**
   * コンストラクタ
   * @param {Object} params - 設定パラメータ
   * @param {number} [params.defaultLat] - デフォルト緯度
   * @param {number} [params.defaultLon] - デフォルト経度
   * @param {number} [params.defaultZoom] - デフォルトズームレベル
   * @param {number} [params.zoomThreshold] - マーカー/ポリゴン切り替え閾値
   * @param {Array<string>} [params.geojsonUrls] - GeoJSONファイルのURLリスト
   */
  constructor(params = {}) {
    this.params = { ...Main.DEFAULT_CONFIG, ...params };

    this.isJapanese = navigator.language.startsWith('ja');

    // 地図とデータ
    this.map = null;
    this.geojsonDataList = [];

    // レイヤー管理
    this.layers = {
      boundaries: [],      // 境界ポリゴン
      depthPolygons: [],   // 水深ポリゴン
      contours: [],        // 等高線
      markers: []          // マーカー
    };
  }


  /**
   * 水深に基づいて色を返す
   *
   * @param {number} depth - 水深（メートル、負の値も許容）
   * @returns {string} カラーコード（例: '#0099ff'）
   *
   * @example
   * getDepthColor(-25) // returns '#4db8ff'
   */
  getDepthColor(depth) {
    const absDepth = Math.abs(depth);

    for (const { max, color } of Main.DEPTH_COLORS) {
      if (absDepth < max) {
        return color;
      }
    }

    return Main.DEPTH_COLORS[Main.DEPTH_COLORS.length - 1].color;
  }


  /**
   * マーカー表示用のレイヤーを作成
   *
   * ズームアウト時に表示される、エリア名とピンアイコンのマーカーを生成します。
   *
   * @param {Object} data - GeoJSON FeatureCollection
   * @returns {L.LayerGroup} マーカーのレイヤーグループ
   * @private
   */
  createMarkerLayer(data) {
    const markers = [];

    data.features.forEach(feature => {
      // 境界Featureのみ処理（水深ポリゴンや等高線は除外）
      if (feature.properties?.type !== 'boundary') {
        return;
      }

      const { center_lat, center_lon, name, link } = feature.properties;

      if (!center_lat || !center_lon) {
        return;
      }

      const displayName = name || (this.isJapanese ? '名前なし' : 'No name');

      // カスタムアイコンでマーカーを作成
      const icon = L.divIcon({
        className: 'area-label',
        html: `
          <div style="text-align: center;">
            <div class="area-label-pin">📍</div>
            <div class="area-label-text">${displayName}</div>
          </div>
        `,
        iconSize: [200, 80],
        iconAnchor: [100, 40]
      });

      const marker = L.marker([center_lat, center_lon], { icon });

      // クリックイベント
      marker.on('click', () => {
        window.location.href = link || './index-bathymetric-data-dev.html';
      });

      // ツールチップ
      const tooltipContent = this.isJapanese ?
        `<strong>${displayName}</strong><br>クリックで3次元可視化ページを開く` :
        `<strong>${displayName}</strong><br>Click to open 3D visualization page`;

      marker.bindTooltip(tooltipContent, {
        direction: 'top',
        offset: [0, -40],
        className: 'custom-tooltip'
      });

      markers.push(marker);
    });

    return L.layerGroup(markers);
  }


  /**
   * 境界ポリゴン表示用のレイヤーを作成
   *
   * データエリアの外周を示す境界線を表示します。
   * クリック/ホバーでインタラクション可能です。
   *
   * @param {Object} data - GeoJSON FeatureCollection
   * @returns {L.GeoJSON|null} 境界レイヤー（境界がない場合はnull）
   * @private
   */
  createBoundaryLayer(data) {
    const boundaryFeatures = data.features.filter(
      f => f.properties?.type === 'boundary'
    );

    if (boundaryFeatures.length === 0) {
      return null;
    }

    return L.geoJSON({ type: 'FeatureCollection', features: boundaryFeatures }, {
      style: Main.STYLES.boundary,
      onEachFeature: (feature, layer) => {
        const { name, description, link } = feature.properties;

        layer.on('click', () => {
          window.location.href = link || './index-bathymetric-data-dev.html';
        });

        const tooltipContent = this.isJapanese ?
          `<strong>${name || '名前なし'}</strong><br>
           ${description || ''}<br>
           <em>クリックで3次元可視化ページを開きます</em>` :
          `<strong>${name || 'No name'}</strong><br>
           ${description || ''}<br>
           <em>Click to open 3D visualization page</em>`;

        layer.bindTooltip(tooltipContent, {
          sticky: true,
          opacity: 0.95,
          className: 'custom-tooltip'
        });

        layer.on('mouseover', () => {
          layer.setStyle(Main.STYLES.boundaryHover);
        });

        layer.on('mouseout', () => {
          layer.setStyle(Main.STYLES.boundary);
        });
      }
    });
  }


  /**
   * 等高線表示用のレイヤーを作成
   *
   * 水深を示す等高線（LineString）を表示します。
   * マウスイベントは受け付けません（interactive: false）。
   *
   * @param {Object} data - GeoJSON FeatureCollection
   * @returns {L.GeoJSON|null} 等高線レイヤー（等高線がない場合はnull）
   * @private
   */

  createContourLayer(data) {
    const contourFeatures = data.features.filter(
      f => f.properties?.type === 'contour'
    );

    if (contourFeatures.length === 0) {
      return null;
    }

    return L.geoJSON(
      { type: 'FeatureCollection', features: contourFeatures },
      { style: Main.STYLES.contour }
    );
  }


  /**
   * 水深ポリゴン表示用のレイヤーを作成
   *
   * 水深範囲ごとに色分けされたポリゴンを表示します。
   * ツールチップで詳細情報を表示し、クリックで詳細ページに遷移します。
   *
   * @param {Object} data - GeoJSON FeatureCollection
   * @returns {L.GeoJSON|null} 水深ポリゴンレイヤー（ポリゴンがない場合はnull）
   * @private
   */
  createDepthPolygonLayer(data) {
    const depthPolygonFeatures = data.features.filter(
      f => f.properties?.type === 'depth_polygon'
    );

    if (depthPolygonFeatures.length === 0) {
      return null;
    }

    const boundaryFeature = data.features.find(f => f.properties?.type === 'boundary');
    const boundaryLink = boundaryFeature?.properties?.link || './index-bathymetric-data-dev.html';

    return L.geoJSON({ type: 'FeatureCollection', features: depthPolygonFeatures }, {
      style: (feature) => {
        const color = this.getDepthColor(feature.properties.depth);
        return {
          ...Main.STYLES.depthPolygon,
          fillColor: color,
          color: color
        };
      },
      onEachFeature: (feature, layer) => {
        layer.on('click', () => {
          window.location.href = boundaryLink;
        });

        const { depth, depth_min, depth_max } = feature.properties;
        const tooltipContent = this.isJapanese ?
          `<strong>水深範囲</strong><br>
           ${depth_min.toFixed(1)}m ~ ${depth_max.toFixed(1)}m<br>
           平均: ${depth.toFixed(1)}m<br>
           <em>クリックで3次元可視化ページを開きます</em>` :
          `<strong>Depth Range</strong><br>
           ${depth_min.toFixed(1)}m ~ ${depth_max.toFixed(1)}m<br>
           Average: ${depth.toFixed(1)}m<br>
           <em>Click to open 3D visualization page</em>`;

        layer.bindTooltip(tooltipContent, {
          sticky: true,
          opacity: 0.95,
          className: 'custom-tooltip'
        });
      }
    });
  }


  /**
   * ズームレベルに応じて表示を切り替え
   *
   * ズームアウト時: マーカーのみ表示
   * ズームイン時: 水深ポリゴン → 境界 → 等高線の順に表示
   *
   * @private
   */
  updateLayerByZoom() {
    const currentZoom = this.map.getZoom();
    const isZoomedOut = currentZoom <= this.params.zoomThreshold;

    // すべてのレイヤーを一旦削除
    Object.values(this.layers).flat().forEach(layer => {
      if (layer && this.map.hasLayer(layer)) {
        this.map.removeLayer(layer);
      }
    });

    if (isZoomedOut) {
      // ズームアウト時: マーカーのみ表示
      this.layers.markers.forEach(layer => this.map.addLayer(layer));
    } else {
      // ズームイン時: 水深ポリゴン → 境界 → 等高線の順に表示
      this.layers.depthPolygons.forEach(layer => layer && this.map.addLayer(layer));
      this.layers.boundaries.forEach(layer => layer && this.map.addLayer(layer));
      this.layers.contours.forEach(layer => layer && this.map.addLayer(layer));
    }
  }

  /**
   * エラー時の地図表示（東京駅）
   *
   * GeoJSONの読み込みに失敗した場合、デフォルト位置（東京駅）の
   * 地図を表示してエラーメッセージを表示します。
   *
   * @private
   */
  showErrorMap() {
    this.map = L.map('map').setView(
      [this.params.defaultLat, this.params.defaultLon],
      this.params.defaultZoom
    );

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19
    }).addTo(this.map);

    const errorMessage = this.isJapanese ?
      `<strong>東京駅</strong><br>GeoJSONファイルの読み込みに失敗しました。` :
      `<strong>Tokyo Station</strong><br>Failed to load GeoJSON file.`;

    L.marker([this.params.defaultLat, this.params.defaultLon])
      .addTo(this.map)
      .bindPopup(errorMessage)
      .openPopup();
  }


  /**
   * 地図の初期化と表示
   *
   * 以下の処理を順次実行します：
   * 1. GeoJSONファイルの並行読み込み
   * 2. データ検証
   * 3. Leafletの初期化
   * 4. レイヤーの作成と追加
   * 5. イベントリスナーの設定
   * 6. 地図範囲の調整
   *
   * @async
   * @throws {Error} GeoJSONの読み込みや検証に失敗した場合
   *
   * @example
   * const app = new Main();
   * await app.initialize();
   */
  async initialize() {
    try {
      // GeoJSONファイルを並行読み込み
      const fetchPromises = this.params.geojsonUrls.map(url =>
        fetch(url).then(response => {
          if (!response.ok) {
            throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
          }
          return response.json();
        }).catch(error => {
          console.error(`Error loading ${url}:`, error);
          throw error;
        })
      );

      this.geojsonDataList = await Promise.all(fetchPromises);

      // データ検証
      if (this.geojsonDataList.length === 0) {
        throw new Error('No GeoJSON data loaded');
      }

      // 各データの検証
      this.geojsonDataList.forEach((data, index) => {
        if (!data.features || !Array.isArray(data.features)) {
          throw new Error(`Invalid GeoJSON format in file ${index + 1}`);
        }
      });

      // 最初のGeoJSONから中心座標を取得
      const firstBoundary = this.geojsonDataList[0].features.find(
        f => f.properties?.type === 'boundary'
      );

      if (!firstBoundary) {
        console.warn('No boundary feature found, using default coordinates');
      }

      const centerLat = firstBoundary?.properties?.center_lat || this.params.defaultLat;
      const centerLon = firstBoundary?.properties?.center_lon || this.params.defaultLon;

      // 地図の初期化
      this.map = L.map('map').setView([centerLat, centerLon], 11);

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
      }).addTo(this.map);

      // 各GeoJSONデータに対してレイヤーを作成
      this.geojsonDataList.forEach(data => {
        this.layers.boundaries.push(this.createBoundaryLayer(data));
        this.layers.depthPolygons.push(this.createDepthPolygonLayer(data));
        this.layers.contours.push(this.createContourLayer(data));
        this.layers.markers.push(this.createMarkerLayer(data));
      });

      // 初期表示
      this.updateLayerByZoom();

      // ズームイベントリスナーを追加
      this.map.on('zoomend', () => this.updateLayerByZoom());

      // すべての境界を含む範囲に地図をフィット
      const validBoundaries = this.layers.boundaries.filter(layer => layer);

      if (validBoundaries.length > 0) {
        const allBounds = validBoundaries.reduce(
          (bounds, layer) => bounds.extend(layer.getBounds()),
          L.latLngBounds([])
        );

        this.map.fitBounds(allBounds, {
          padding: [100, 100],
          maxZoom: 13
        });
      }

    } catch (error) {
      console.error('Initialization error:', error);
      this.showErrorMap();
    }
  }


}