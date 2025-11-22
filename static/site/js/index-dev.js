/**
 * GeoJSONベースの地図表示アプリケーション
 * ズームレベルに応じてマーカーとポリゴンを切り替えて表示
 */
export class Main {

  params = {
    defaultLat: 35.6812,   // デフォルト座標（東京駅の緯度）
    defaultLon: 139.7671,  // デフォルト座標（東京駅の経度）
    defaultZoom: 13,       // デフォルトズームレベル
    zoomThreshold: 11,     // マーカー/ポリゴン切り替えの閾値

    geojsonUrls: [         // GeoJSONファイルのURLリスト
      './data/ALL_depth_map_data_202510.geojson'
    ],
  };

  constructor(params = {}) {
     // paramsを受け取って上記のparamsを上書きする
    this.params = Object.assign(this.params, params);

    this.isJapanese = navigator.language.startsWith('ja');

    this.map = null;
    this.geojsonLayers = [];  // 複数のGeoJSONレイヤーを管理
    this.markerLayers = [];   // 複数のマーカーレイヤーを管理
    this.geojsonDataList = [];
  }


  /**
   * マーカー表示用のレイヤーを作成
   */
  createMarkerLayer(data) {
    const markers = [];

    data.features.forEach(feature => {
      if (feature.properties && feature.properties.center_lat && feature.properties.center_lon) {
        const lat = feature.properties.center_lat;
        const lon = feature.properties.center_lon;
        const name = feature.properties.name || (this.isJapanese ? '名前なし' : 'No name');

        // カスタムアイコンでマーカーを作成
        const icon = L.divIcon({
          className: 'area-label',
          html: `
            <div style="text-align: center;">
              <div class="area-label-pin">📍</div>
              <div class="area-label-text">${name}</div>
            </div>
          `,
          iconSize: [200, 80],
          iconAnchor: [100, 40]
        });

        const marker = L.marker([lat, lon], { icon: icon });

        // クリックイベント
        marker.on('click', () => {
          const linkUrl = feature.properties.link || './index-bathymetric-data-dev.html';
          window.location.href = linkUrl;
        });

        // ツールチップ
        const tooltipContent = this.isJapanese ?
          `<strong>${name}</strong><br>クリックで詳細ページを開きます` :
          `<strong>${name}</strong><br>Click to open details page`;

        marker.bindTooltip(tooltipContent, {
          direction: 'top',
          offset: [0, -40],
          className: 'custom-tooltip'
        });

        markers.push(marker);
      }
    });

    return L.layerGroup(markers);
  }

  /**
   * GeoJSONポリゴン表示用のレイヤーを作成
   */
  createGeoJSONLayer(data) {
    return L.geoJSON(data, {
      style: {
        color: '#3388ff',
        weight: 2,
        opacity: 0.8,
        fillOpacity: 0.3
      },
      onEachFeature: (feature, layer) => {
        // クリックイベントを追加
        layer.on('click', (e) => {
          const linkUrl = feature.properties.link || './index-bathymetric-data-dev.html';
          window.location.href = linkUrl;
        });

        // マウスオーバーでツールチップを表示
        if (feature.properties) {
          const props = feature.properties;
          const popupContent = this.isJapanese ?
            `<strong>${props.name || '名前なし'}</strong><br>
             ${props.description || ''}<br>
             <em>クリックで詳細ページを開きます</em>` :
            `<strong>${props.name || 'No name'}</strong><br>
             ${props.description || ''}<br>
             <em>Click to open details page</em>`;

          layer.bindTooltip(popupContent, {
            sticky: true,
            opacity: 0.95,
            className: 'custom-tooltip'
          });

          // マウスオーバーでハイライト
          layer.on('mouseover', (e) => {
            layer.setStyle({
              fillOpacity: 0.5,
              weight: 3
            });
          });

          // マウスアウトで元に戻す
          layer.on('mouseout', (e) => {
            layer.setStyle({
              fillOpacity: 0.3,
              weight: 2
            });
          });
        }
      }
    });
  }

  /**
   * ズームレベルに応じて表示を切り替え
   */
  updateLayerByZoom() {
    const currentZoom = this.map.getZoom();

    if (currentZoom <= this.params.zoomThreshold) {
      // ズームアウト時: マーカー表示
      this.geojsonLayers.forEach(layer => {
        if (this.map.hasLayer(layer)) {
          this.map.removeLayer(layer);
        }
      });
      this.markerLayers.forEach(layer => {
        if (!this.map.hasLayer(layer)) {
          this.map.addLayer(layer);
        }
      });
    } else {
      // ズームイン時: ポリゴン表示
      this.markerLayers.forEach(layer => {
        if (this.map.hasLayer(layer)) {
          this.map.removeLayer(layer);
        }
      });
      this.geojsonLayers.forEach(layer => {
        if (!this.map.hasLayer(layer)) {
          this.map.addLayer(layer);
        }
      });
    }
  }


  /**
   * エラー時の地図表示（東京駅）
   */
  showErrorMap() {
    this.map = L.map('map').setView([this.params.defaultLat, this.params.defaultLon], this.params.defaultZoom);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19
    }).addTo(this.map);

    const error_message = this.isJapanese ?
      `<strong>東京駅</strong><br>GeoJSONファイルの読み込みに失敗しました。東京駅を表示しています。` :
      `<strong>Tokyo Station</strong><br>Failed to load GeoJSON file, displaying Tokyo Station.`;

    L.marker([this.params.defaultLat, this.params.defaultLon])
      .addTo(this.map)
      .bindPopup(error_message)
      .openPopup();
  }


  /**
   * 地図の初期化と表示
   */
  async initialize() {
    try {
      // 複数のGeoJSONファイルを並行読み込み
      const fetchPromises = this.params.geojsonUrls.map(url =>
        fetch(url).then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} for ${url}`);
          }
          return response.json();
        })
      );

      this.geojsonDataList = await Promise.all(fetchPromises);

      // 最初のGeoJSONから中心座標を取得
      const firstFeature = this.geojsonDataList[0].features[0];
      const centerLat = firstFeature.properties.center_lat || this.params.defaultLat;
      const centerLon = firstFeature.properties.center_lon || this.params.defaultLon;

      // 地図の初期化（最初のGeoJSONの中心座標を使用）
      this.map = L.map('map').setView([centerLat, centerLon], 11);

      // OpenStreetMapタイルレイヤーを追加
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
      }).addTo(this.map);

      // 各GeoJSONデータに対してレイヤーを作成
      this.geojsonDataList.forEach(data => {
        const geojsonLayer = this.createGeoJSONLayer(data);
        const markerLayer = this.createMarkerLayer(data);

        this.geojsonLayers.push(geojsonLayer);
        this.markerLayers.push(markerLayer);
      });

      // 初期表示
      this.updateLayerByZoom();

      // ズームイベントリスナーを追加
      this.map.on('zoomend', () => this.updateLayerByZoom());

      // すべてのGeoJSONを含む範囲に地図をフィット
      const allBounds = this.geojsonLayers.reduce((bounds, layer) => {
        return bounds.extend(layer.getBounds());
      }, L.latLngBounds([]));

      this.map.fitBounds(allBounds, {
        padding: [100, 100],
        maxZoom: 13
      });

    } catch (error) {
      console.error('GeoJSONの読み込みエラー:', error);
      this.showErrorMap();
    }
  }
}