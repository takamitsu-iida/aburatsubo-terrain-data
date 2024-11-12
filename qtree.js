

class QuadtreeNode {

  // 領域を表すオブジェクト
  // X軸 = Longitude(経度)、Z軸 = Latitude(緯度)
  // { lon1, lat1, lon2, lat2 }
  bounds;

  // 階層の深さ
  depth;

  // データ配列
  points;

  // 子ノードの配列
  children;

  // 親ノードへの参照
  parent;

  constructor(bounds, depth = 0, parent = null) {
    this.bounds = bounds;
    this.depth = depth;
    this.points = [];
    this.children = [];
    this.parent = parent;
  }

  isLeaf() {
    return this.children.length === 0;
  }

  subdivide() {
    const { lon1, lat1, lon2, lat2 } = this.bounds;
    const midLon = (lon1 + lon2) / 2;
    const midLat = (lat1 + lat2) / 2;

    // このノードを四分木で分割する
    // children配列には以下の順に追加する
    //  +---+---+
    //  | 0 | 1 |
    //  +---+---+
    //  | 2 | 3 |
    //  +---+---+

    this.children.push(new QuadtreeNode({ lon1: lon1, lat1: lat1, lon2: midLon, lat2: midLat }, this.depth + 1, this));
    this.children.push(new QuadtreeNode({ lon1: midLon, lat1: lat1, lon2: lon2, lat2: midLat }, this.depth + 1, this));
    this.children.push(new QuadtreeNode({ lon1: lon1, lat1: midLat, lon2: midLon, lat2: lat2 }, this.depth + 1, this));
    this.children.push(new QuadtreeNode({ lon1: midLon, lat1: midLat, lon2: lon2, lat2: lat2 }, this.depth + 1, this));
  }

  insert(point) {
    if (!this.contains(point)) {
      return false;
    }

    if (this.isLeaf()) {
      if (this.points.length < Quadtree.MAX_POINTS || this.depth >= Quadtree.MAX_DEPTH) {
        point.quadtreeNode = this;  // ポイントから四分木ノードを辿れるように参照を設定
        this.points.push(point);
        return true;
      } else {
        this.subdivide();
        this.points.forEach(p => this.insertIntoChildren(p));
        this.points = [];
      }
    }

    return this.insertIntoChildren(point);
  }

  insertIntoChildren(point) {
    for (const child of this.children) {
      if (child.insert(point)) {
        return true;
      }
    }
    return false;
  }

  contains(point) {
    const { lon1, lat1, lon2, lat2 } = this.bounds;
    return point.lon >= lon1 && point.lon < lon2 && point.lat >= lat1 && point.lat < lat2;
  }

  query(range, found = []) {
    if (!this.intersects(range)) {
      return found;
    }

    for (const point of this.points) {
      if (range.contains(point)) {
        found.push(point);
      }
    }

    if (!this.isLeaf()) {
      for (const child of this.children) {
        child.query(range, found);
      }
    }

    return found;
  }

  intersects(range) {
    const { lon1, lat1, lon2, lat2 } = this.bounds;
    return !(range.lon1 > lon2 || range.lon2 < lon1 || range.lat1 > lat2 || range.lat2 < lat1);
  }

  getNodesAtDepth(targetDepth, nodes = []) {
    if (this.depth === targetDepth) {
      nodes.push(this);
    } else if (this.depth < targetDepth && !this.isLeaf()) {
      for (const child of this.children) {
        child.getNodesAtDepth(targetDepth, nodes);
      }
    }
    return nodes;
  }

  getLeafNodes(nodes = []) {
    if (this.isLeaf()) {
      nodes.push(this);
    } else {
      for (const child of this.children) {
        child.getLeafNodes(nodes);
      }
    }
    return nodes;
  }

}


class Quadtree {
  /*
    使い方
    const bounds = { lon1: 0, lat1: 0, lon2: 100, lat2: 100 };
    const quadtree = new Quadtree(bounds);

    const points = [
      { lon: 10, lat: 10 },
      { lon: 20, lat: 20 },
      { lon: 30, lat: 30 },
      { lon: 40, lat: 40 },
      { lon: 50, lat: 50 },
    ];

    points.forEach(point => quadtree.insert(point));

    const nodesAtDepth2 = quadtree.getNodesAtDepth(2);
    console.log(nodesAtDepth2);
  */

  static MAX_DIVISION = 10;
  static MAX_POINTS = 5;

  constructor(bounds) {
    this.root = new QuadtreeNode(bounds);
  }

  insert(point) {
    this.root.insert(point);
  }

  query(range) {
    return this.root.query(range);
  }

  getNodesAtDepth(depth) {
    return this.root.getNodesAtDepth(depth);
  }

  getLeafNodes() {
    return this.root.getLeafNodes();
  }

}
