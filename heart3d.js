// 3D heart viewer: Human Reference Atlas heart + great vessels (HuBMAP, CC BY 4.0),
// merged and decimated by tools/build_heart.py. Coordinates in mm: +x patient left, +y superior, +z anterior.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const COLORS = { LV: "#B8322A", LA: "#E0735A", RV: "#2C5F9E", RA: "#6E9BD1", AV: "#B7791F", MV: "#DDAA3C", TV: "#2F8062", PV: "#76B99A",
  AoR: "#5E3A96", Arch: "#9D82C6", PA: "#2A8A9A", Peri: "#7C7A72" };
const CONTEXT = { IVS: "#b7a197", PAP: "#a88b80", SVC: "#8f969e", IVC: "#8f969e", PVn: "#a3a9b0" };
const NAMES = { LV: "Left ventricle", LA: "Left atrium", RV: "Right ventricle", RA: "Right atrium", AV: "Aortic valve", MV: "Mitral valve",
  TV: "Tricuspid valve", PV: "Pulmonic valve", AoR: "Aortic root / ascending aorta", Arch: "Aortic arch", PA: "Pulmonary artery", Peri: "Pericardium (approximate envelope)",
  IVS: "Interventricular septum", PAP: "Papillary muscles", SVC: "Superior vena cava", IVC: "Inferior vena cava", PVn: "Pulmonary veins" };
const GREY = new THREE.Color("#c9c7c2");

let gltfPromise = null;
const loadHeart = () => gltfPromise || (gltfPromise = new GLTFLoader().loadAsync("assets/heart/heart.glb"));

class Heart3D {
  constructor(el, opts = {}) {
    this.el = el; this.opts = opts; this.parts = {}; this.weights = null; this.hl = null; this.view = null; this.cut = false;
    el.innerHTML = '<div class="loading">Loading 3D heart…</div>';
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, devicePixelRatio));
    this.renderer.localClippingEnabled = true;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 1, 3000);
    this.camera.position.set(110, 70, 330);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.dampingFactor = .08;
    this.controls.minDistance = 120; this.controls.maxDistance = 700;
    this.controls.target.set(0, 12, 0);
    this.controls.autoRotate = !!opts.autoRotate; this.controls.autoRotateSpeed = 1.2;
    this.controls.addEventListener("start", () => { this.controls.autoRotate = false; });
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.6));
    const d1 = new THREE.DirectionalLight(0xffffff, 1.6); d1.position.set(150, 200, 300); this.scene.add(d1);
    const d2 = new THREE.DirectionalLight(0xffffff, .6); d2.position.set(-200, -100, -150); this.scene.add(d2);
    this.clip = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    this.sector = new THREE.Group(); this.scene.add(this.sector);
    this.ray = new THREE.Raycaster(); this.mouse = new THREE.Vector2();
    this.label = document.createElement("div"); this.label.className = "lbl3d"; this.label.style.display = "none";
    loadHeart().then(g => this._init(g)).catch(e => { el.innerHTML = `<div class="loading">3D model failed to load (${e.message})</div>`; });
  }
  _init(gltf) {
    this.el.innerHTML = ""; this.el.appendChild(this.renderer.domElement); this.el.appendChild(this.label);
    const root = gltf.scene.clone(true);
    root.traverse(o => {
      if (!o.isMesh) return;
      let code = o.name; let p = o;
      while (p && !(code in COLORS) && !(code in CONTEXT)) { p = p.parent; code = p ? p.name : code; }
      if (!(code in COLORS) && !(code in CONTEXT)) return;
      const peri = code === "Peri";
      if (o.geometry.attributes.color) o.geometry.deleteAttribute("color");
      o.geometry.computeVertexNormals();
      o.material = new THREE.MeshStandardMaterial({
        color: COLORS[code] || CONTEXT[code], roughness: .55, metalness: .05, side: THREE.DoubleSide,
        transparent: peri, opacity: peri ? .1 : 1, depthWrite: !peri, clippingPlanes: [], clipShadows: true
      });
      o.renderOrder = peri ? 10 : 0;
      o.userData.code = code;
      (this.parts[code] = this.parts[code] || []).push(o);
    });
    this.scene.add(root);
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(this.el);
    const dom = this.renderer.domElement;
    dom.addEventListener("pointermove", e => this._hover(e));
    dom.addEventListener("pointerleave", () => { this.label.style.display = "none"; if (this.opts.onHover) this.opts.onHover(null); });
    this.resize(); this._apply();
    if (this.view) this.setView(this.view, true);
    const loop = () => { this._raf = requestAnimationFrame(loop); if (!this._visible()) return; this.controls.update(); this.renderer.render(this.scene, this.camera); };
    loop();
  }
  _visible() { const r = this.el.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.width > 0; }
  resize() { const w = this.el.clientWidth, h = this.el.clientHeight || w; if (!w) return; this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); }
  _hover(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(this.mouse, this.camera);
    const objs = Object.entries(this.parts).filter(([k]) => k !== "Peri").flatMap(([, v]) => v);
    let hits = this.ray.intersectObjects(objs, false);
    if (this.cut) hits = hits.filter(h => this.clip.distanceToPoint(h.point) >= 0);
    const hit = hits[0];
    if (!hit) { this.label.style.display = "none"; if (this.opts.onHover) this.opts.onHover(null); return; }
    const code = hit.object.userData.code;
    const w = this.weights && code in COLORS ? ` · w = ${this.weights[code] ?? 0}` : "";
    this.label.textContent = NAMES[code] + w; this.label.style.display = "block";
    this.label.style.left = (e.clientX - r.left) + "px"; this.label.style.top = (e.clientY - r.top) + "px";
    if (this.opts.onHover) this.opts.onHover(code);
  }
  _apply() {
    for (const [code, meshes] of Object.entries(this.parts)) {
      for (const m of meshes) {
        const mat = m.material, base = new THREE.Color(COLORS[code] || CONTEXT[code]);
        let col = base.clone(), op = 1, emis = 0;
        if (code in COLORS && this.weights) {
          const w = this.weights[code] ?? 0;
          col = w > 0 ? GREY.clone().lerp(base, .35 + .65 * w) : GREY.clone();
          op = w > 0 ? 1 : .22;
        } else if (code in CONTEXT && this.weights) { op = .35; }
        if (this.hl) { if (code === this.hl) emis = .35; else op = Math.min(op, .14); }
        if (code === "Peri") { op = this.hl === "Peri" ? .35 : (this.weights && !(this.weights.Peri > 0) ? .06 : .12); }
        mat.color.copy(col); mat.emissive = col.clone().multiplyScalar(emis);
        mat.transparent = op < 1 || code === "Peri"; mat.opacity = op; mat.depthWrite = op >= 1;
        mat.clippingPlanes = this.cut ? [this.clip] : [];
        mat.needsUpdate = true;
      }
    }
  }
  setWeights(w) { this.weights = w; if (Object.keys(this.parts).length) this._apply(); }
  highlight(code) { this.hl = code; if (Object.keys(this.parts).length) this._apply(); }
  setCut(on) { this.cut = on; if (this.view) this._placeClip(); this._apply(); if (on && this.view) this._faceView(); }
  _placeClip() {
    const V = window.VIEWS3D[this.view]; const n = new THREE.Vector3(...V.n), T = new THREE.Vector3(...V.T);
    // keep the half away from the camera side (+n removed)
    this.clip.setFromNormalAndCoplanarPoint(n.clone().negate(), T);
  }
  _faceView() {
    const V = window.VIEWS3D[this.view]; const n = new THREE.Vector3(...V.n), T = new THREE.Vector3(...V.T), u = new THREE.Vector3(...V.u);
    const c = T.clone().add(u.clone().multiplyScalar(V.depth * .5));
    const up = u.clone().negate();
    this._tween(c.clone().add(n.clone().multiplyScalar(330)), c, up);
  }
  _tween(pos, target, up) {
    const p0 = this.camera.position.clone(), t0 = this.controls.target.clone(), u0 = this.camera.up.clone();
    const start = performance.now(), D = 700;
    const step = now => {
      const k = Math.min(1, (now - start) / D), e = 1 - Math.pow(1 - k, 3);
      this.camera.position.lerpVectors(p0, pos, e); this.controls.target.lerpVectors(t0, target, e);
      if (up) this.camera.up.lerpVectors(u0, up, e).normalize();
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  resetCamera() { this._tween(new THREE.Vector3(110, 70, 330), new THREE.Vector3(0, 12, 0), new THREE.Vector3(0, 1, 0)); }
  setView(name, instant) {
    this.view = name;
    this.sector.clear();
    if (!name || !window.VIEWS3D[name]) return;
    const V = window.VIEWS3D[name];
    const T = new THREE.Vector3(...V.T), u = new THREE.Vector3(...V.u), v = new THREE.Vector3(...V.v);
    const pts = [T.clone()], N = 40, a = V.ang * Math.PI / 180;
    for (let i = 0; i <= N; i++) { const t = -a + 2 * a * i / N; pts.push(T.clone().add(u.clone().multiplyScalar(Math.cos(t) * V.depth)).add(v.clone().multiplyScalar(Math.sin(t) * V.depth))); }
    const pos = []; for (let i = 1; i < pts.length - 1; i++) pos.push(...pts[0].toArray(), ...pts[i].toArray(), ...pts[i + 1].toArray());
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    const fan = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xef6b5b, transparent: true, opacity: .16, side: THREE.DoubleSide, depthWrite: false }));
    fan.renderOrder = 20;
    const edge = new THREE.Line(new THREE.BufferGeometry().setFromPoints([...pts, pts[0]]), new THREE.LineBasicMaterial({ color: 0xc0392b, transparent: true, opacity: .9 }));
    const probe = new THREE.Mesh(new THREE.SphereGeometry(5, 20, 14), new THREE.MeshStandardMaterial({ color: 0x222222, roughness: .4 }));
    probe.position.copy(T);
    this.sector.add(fan, edge, probe);
    if (this.cut) { this._placeClip(); this._apply(); this._faceView(); }
  }
  dispose() { cancelAnimationFrame(this._raf); this.ro && this.ro.disconnect(); this.renderer.dispose(); }
}
window.Heart3D = Heart3D;
window.HEART_NAMES = NAMES;
window.dispatchEvent(new Event("heart3d-ready"));
