import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

const $ = (id) => document.getElementById(id);
const manifest = await fetch("scenes/index.json").then((r) => r.json());
let info = null, ASPECT = 1.5, entry = null;

// ---------------------------------------------------------------- renderer
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
$("stage").appendChild(renderer.domElement);
const canvas = renderer.domElement;

const scene = new THREE.Scene();
// Sorting is driven by hand so two viewpoints can share one sort (split view).
// LOD driving stalls free-view frames on these single-image splats, so it is off.
const spark = new SparkRenderer({ renderer, autoUpdate: false, enableLod: false });
scene.add(spark);
let splat = null;
const hemi = new THREE.HemisphereLight(0xcfe3ff, 0xb59a7a, 1.4);
const sun = new THREE.DirectionalLight(0xfff1dc, 2.4);
scene.add(hemi, sun);

// ---------------------------------------------------------------- cameras
const paintCam = new THREE.PerspectiveCamera(45, ASPECT, 0.05, 500);
const freeCam = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
const controls = new OrbitControls(freeCam, canvas);
controls.enableDamping = true;
controls.enabled = false;

// ---------------------------------------------------------------- walkable heightfield
let hf = null, H = null, clear = null, groundMesh = null, walkMinZ = 0;
const idx = (ix, iz) => iz * hf.nx + ix;
const inGrid = (ix, iz) => ix >= 0 && iz >= 0 && ix < hf.nx && iz < hf.nz;
const walkable = (ix, iz) => inGrid(ix, iz) && Number.isFinite(H[idx(ix, iz)]);
const toCell = (x, z) => [Math.round((x - hf.x0) / hf.cell), Math.round((z - hf.z0) / hf.cell)];
const cellPos = (ix, iz) => new THREE.Vector3(ix * hf.cell + hf.x0, H[idx(ix, iz)], iz * hf.cell + hf.z0);
const isClear = (ix, iz) => inGrid(ix, iz) && clear[idx(ix, iz)] === 1;

function setupWalkable() {
  hf = info.heightfield;
  H = Float32Array.from(hf.h, (v) => (v === null ? NaN : v));
  // a standing NPC occupies its spot
  for (const n of (info.hotspots || []).filter((h) => h.action === "npc" && h.pose !== "sit")) {
    const [cx, cz] = toCell(n.at[0], n.at[2]);
    for (let dz = -4; dz <= 4; dz++) for (let dx = -4; dx <= 4; dx++)
      if (dx * dx + dz * dz <= 12 && inGrid(cx + dx, cz + dz)) H[idx(cx + dx, cz + dz)] = NaN;
  }
  // cells at least ~0.2 m from any edge, so the body does not clip walls
  clear = new Uint8Array(hf.nx * hf.nz);
  walkMinZ = Infinity;
  for (let iz = 0; iz < hf.nz; iz++) for (let ix = 0; ix < hf.nx; ix++) {
    let ok = walkable(ix, iz);
    if (ok) walkMinZ = Math.min(walkMinZ, iz * hf.cell + hf.z0);
    for (let dz = -2; ok && dz <= 2; dz++) for (let dx = -2; ok && dx <= 2; dx++) ok = walkable(ix + dx, iz + dz);
    clear[idx(ix, iz)] = ok ? 1 : 0;
  }
  if (groundMesh) { scene.remove(groundMesh); groundMesh.geometry.dispose(); }
  groundMesh = buildGroundMesh();
}

function heightAt(x, z, fallback = 0) {
  const fx = (x - hf.x0) / hf.cell, fz = (z - hf.z0) / hf.cell;
  const ix = Math.floor(fx), iz = Math.floor(fz);
  let sum = 0, wsum = 0;
  for (const [dx, dz] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
    if (!walkable(ix + dx, iz + dz)) continue;
    const w = (1 - Math.abs(fx - ix - dx)) * (1 - Math.abs(fz - iz - dz));
    sum += w * H[idx(ix + dx, iz + dz)];
    wsum += w;
  }
  return wsum > 1e-6 ? sum / wsum : fallback;
}

function nearestCell(ix, iz, test = isClear) {
  if (test(ix, iz)) return [ix, iz];
  for (let r = 1; r < 80; r++) {
    let best = null, bd = Infinity;
    for (let dz = -r; dz <= r; dz++) for (let dx = -r; dx <= r; dx++) {
      if (Math.max(Math.abs(dx), Math.abs(dz)) !== r || !test(ix + dx, iz + dz)) continue;
      const d = dx * dx + dz * dz;
      if (d < bd) { bd = d; best = [ix + dx, iz + dz]; }
    }
    if (best) return best;
  }
  return null;
}

// invisible collision surface built from the heightfield (stands in for a photogrammetry mesh)
function buildGroundMesh() {
  const pos = [], index = [], vid = new Int32Array(hf.nx * hf.nz).fill(-1);
  for (let iz = 0; iz < hf.nz; iz++) for (let ix = 0; ix < hf.nx; ix++) {
    if (!walkable(ix, iz)) continue;
    vid[idx(ix, iz)] = pos.length / 3;
    const p = cellPos(ix, iz);
    pos.push(p.x, p.y, p.z);
  }
  for (let iz = 0; iz < hf.nz - 1; iz++) for (let ix = 0; ix < hf.nx - 1; ix++) {
    const a = vid[idx(ix, iz)], b = vid[idx(ix + 1, iz)], c = vid[idx(ix, iz + 1)], d = vid[idx(ix + 1, iz + 1)];
    if (a >= 0 && b >= 0 && c >= 0 && d >= 0) index.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(index);
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  m.visible = false;
  scene.add(m);
  return m;
}

// ---------------------------------------------------------------- A* on the grid
function findPath(from, to) {
  const s = nearestCell(...toCell(from.x, from.z), walkable);
  const g = nearestCell(...toCell(to.x, to.z), walkable);
  if (!s || !g) return [];
  const N = hf.nx * hf.nz, start = idx(...s), goal = idx(...g);
  const cost = new Float32Array(N).fill(Infinity), prev = new Int32Array(N).fill(-1), closed = new Uint8Array(N);
  const open = [[0, start]];
  cost[start] = 0;
  const hd = (i) => Math.hypot((i % hf.nx) - g[0], Math.floor(i / hf.nx) - g[1]);
  while (open.length) {
    let bi = 0;
    for (let i = 1; i < open.length; i++) if (open[i][0] < open[bi][0]) bi = i;
    const [, cur] = open.splice(bi, 1)[0];
    if (cur === goal) break;
    if (closed[cur]) continue;
    closed[cur] = 1;
    const cx = cur % hf.nx, cz = Math.floor(cur / hf.nx);
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dz) continue;
      const nx = cx + dx, nz = cz + dz;
      const ni = idx(nx, nz);
      if (!walkable(nx, nz) || closed[ni]) continue;
      // cells close to walls are allowed (narrow passages) but strongly avoided
      const c = cost[cur] + (dx && dz ? 1.414 : 1) * (isClear(nx, nz) ? 1 : 6);
      if (c < cost[ni]) { cost[ni] = c; prev[ni] = cur; open.push([c + hd(ni), ni]); }
    }
  }
  if (prev[goal] < 0 && goal !== start) return [];
  const cells = [];
  for (let i = goal; i >= 0; i = prev[i]) { cells.push(i); if (i === start) break; }
  cells.reverse();
  // string pulling: keep only cells that break line of sight
  const pts = cells.map((i) => cellPos(i % hf.nx, Math.floor(i / hf.nx)));
  const out = [pts[0]];
  let anchor = 0;
  for (let i = 2; i < pts.length; i++) if (!lineClear(pts[anchor], pts[i])) { anchor = i - 1; out.push(pts[anchor]); }
  out.push(pts[pts.length - 1]);
  out[out.length - 1] = to.clone().setY(heightAt(to.x, to.z, pts[pts.length - 1].y));
  return out.slice(1);
}
function lineClear(a, b) {
  const strict = isClear(...toCell(a.x, a.z)) && isClear(...toCell(b.x, b.z));
  const n = Math.ceil(a.distanceTo(b) / (hf.cell * 0.5));
  for (let i = 0; i <= n; i++) {
    const t = i / n, [ix, iz] = toCell(a.x + (b.x - a.x) * t, a.z + (b.z - a.z) * t);
    if (strict ? !isClear(ix, iz) : !walkable(ix, iz)) return false;
  }
  return true;
}

// ---------------------------------------------------------------- hotspots (sit / touch / look)
const ACTION_VERB = { sit: "앉기", touch: "만지기", look: "바라보기", pray: "참배하기" };
let hotspots = [];
const hotspotGroup = new THREE.Group();
scene.add(hotspotGroup);
function setupHotspots() {
  hotspotGroup.clear();
  hotspots = (info.hotspots || []).filter((h) => h.action !== "npc").map((h) => {
    const o = { ...h, target: new THREE.Vector3().fromArray(h.target) };
    if (h.seat) o.seat = new THREE.Vector3().fromArray(h.seat);
    if (h.stand) o.stand = new THREE.Vector3().fromArray(h.stand);
    if (h.facing) o.facing = new THREE.Vector3().fromArray(h.facing).normalize();
    const pick = new THREE.Mesh(new THREE.SphereGeometry(Math.max(h.radius, h.action === "sit" ? 0.45 : 0.3), 12, 8),
      new THREE.MeshBasicMaterial());
    pick.position.copy(o.target);
    pick.visible = false;
    pick.userData.hotspot = o;
    hotspotGroup.add(pick);
    return o;
  });
  // older scene.json files only have a single bench
  if (!hotspots.length && info.bench) {
    const b = info.bench;
    info.hotspots = [{ id: "bench", label: "벤치", action: "sit", radius: 0.45, target: b.seat, ...b }];
    setupHotspots();
  }
}
const seats = () => hotspots.filter((h) => h.action === "sit");

// ---------------------------------------------------------------- character
const loader = new GLTFLoader();
loader.register((p) => new VRMLoaderPlugin(p));
const actor = new THREE.Group();
scene.add(actor);
const actorPick = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 1.6, 10), new THREE.MeshBasicMaterial());
actorPick.position.y = 0.8;
actorPick.visible = false;
actor.add(actorPick);
const eyeTarget = new THREE.Object3D();
scene.add(eyeTarget);

const shadow = (() => {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(64, 64, 4, 64, 64, 64);
  grd.addColorStop(0, "rgba(20,24,40,0.55)");
  grd.addColorStop(1, "rgba(20,24,40,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.6),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  m.rotation.x = -Math.PI / 2;
  scene.add(m);
  return m;
})();

let vrm = null, rig = null;
async function loadVrmModel(url) {
  const gltf = await loader.loadAsync(url);
  const v = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  VRMUtils.rotateVRM0(v);
  v.scene.traverse((o) => { o.frustumCulled = false; });
  // measure the rest pose at the origin
  v.update(0);
  v.scene.updateMatrixWorld(true);
  const wp = (n) => v.humanoid.getRawBoneNode(n).getWorldPosition(new THREE.Vector3());
  return { vrm: v, rig: { hips: wp("hips").y, rShoulder: wp("rightUpperArm"), head: wp("head"), ankle: wp("leftFoot").y } };
}
async function loadVRM(url) {
  const m = await loadVrmModel(url);
  if (vrm) { actor.remove(vrm.scene); VRMUtils.deepDispose(vrm.scene); }
  vrm = m.vrm;
  rig = m.rig;
  actor.add(vrm.scene);
  if (vrm.lookAt) vrm.lookAt.target = eyeTarget;
}

// ---------------------------------------------------------------- pose
const E = new THREE.Euler();
const Q = new THREE.Quaternion(), QI = new THREE.Quaternion();
const ARM_REST = new THREE.Vector3(-1, 0, 0); // right arm in the VRM T-pose
let PV = null, PV0 = false; // the VRM currently being posed; VRM 0.x rigs face the other way
function bone(name) { return PV.humanoid.getNormalizedBoneNode(name); }
// poses are written for VRM 1.0 (facing +Z); a VRM 0.x normalized rig is turned 180 degrees, so X and Z flip
function eul(x, y, z) { return PV0 ? E.set(-x, y, -z, "XYZ") : E.set(x, y, z, "XYZ"); }
const flip = (v) => (PV0 ? v.set(-v.x, v.y, -v.z) : v);
function setBone(name, x = 0, y = 0, z = 0) {
  const b = bone(name);
  if (b) b.quaternion.setFromEuler(eul(x, y, z));
}
// point a forearm along a body-space direction (VRM 1.0 axes: +X = character's left, +Z = forward).
// An elbow hinge alone cannot bring the hands together once the arm hangs down, so aim it instead.
const AIM_Q = new THREE.Quaternion();
function aimForearm(side, dir, weight) {
  const upper = bone(`${side}UpperArm`), lower = bone(`${side}LowerArm`);
  if (!upper || !lower || weight <= 0.001) return;
  const rest = flip(new THREE.Vector3(side === "left" ? 1 : -1, 0, 0));
  const local = flip(dir.clone().normalize()).applyQuaternion(upper.quaternion.clone().invert());
  AIM_Q.setFromUnitVectors(rest, local);
  lower.quaternion.slerp(AIM_Q, weight);
}
const CLASP_L = new THREE.Vector3(-0.66, -0.08, 0.75), CLASP_R = new THREE.Vector3(0.66, -0.08, 0.75);
const PRAY_L = new THREE.Vector3(-0.62, 0.62, 0.48), PRAY_R = new THREE.Vector3(0.62, 0.62, 0.48);
const LAP_L = new THREE.Vector3(-0.26, -0.3, 0.92), LAP_R = new THREE.Vector3(0.26, -0.3, 0.92);
// x: { stair (-1 down .. 1 up), touch, touchDir, crouch, look, wave, headYaw, headPitch, clasp, read, bow }
function applyPose(v, r, t, ph, walkAmt, s, x) {
  PV = v;
  PV0 = v.meta?.metaVersion === "0";
  const c = x.clasp || 0, rd = x.read || 0, bw = x.bow || 0;
  const w = walkAmt * (1 - s);
  const sp = Math.sin(ph), cp = Math.cos(ph);
  const up = Math.max(0, x.stair), down = Math.max(0, -x.stair), cr = x.crouch;
  // legs: VRM faces +Z, a negative X rotation swings a leg forward; stairs lift the swinging knee higher
  const lift = 0.95 + 0.7 * up + 0.25 * down;
  const L = -0.5 * sp * w - 0.45 * up * Math.max(0, cp) * w;
  const R = 0.5 * sp * w - 0.45 * up * Math.max(0, -cp) * w;
  const kneeL = 0.08 * w + lift * Math.max(0, cp) * w, kneeR = 0.08 * w + lift * Math.max(0, -cp) * w;
  setBone("leftUpperLeg", L - 1.5 * s - 1.1 * cr, 0, 0.04 * s + 0.12 * cr);
  setBone("rightUpperLeg", R - 1.5 * s - 1.1 * cr, 0, -0.04 * s - 0.12 * cr);
  setBone("leftLowerLeg", kneeL + 1.5 * s + 1.9 * cr);
  setBone("rightLowerLeg", kneeR + 1.5 * s + 1.9 * cr);
  setBone("leftFoot", -0.6 * (L + kneeL) * (1 - s) - 0.75 * cr);
  setBone("rightFoot", -0.6 * (R + kneeR) * (1 - s) - 0.75 * cr);
  // torso: lean into a climb, back on the way down, forward when reaching low
  const breath = Math.sin(t * 1.6) * 0.015;
  const lean = 0.04 * w + 0.2 * up * w - 0.1 * down * w + 0.05 * s + 0.45 * cr + 0.12 * x.touch * Math.max(0, -x.touchDir.y)
    + 0.55 * bw + 0.1 * rd;
  setBone("hips", 0, 0.12 * sp * w, 0);
  setBone("spine", lean + breath, -0.08 * sp * w, 0);
  setBone("chest", breath, -0.05 * sp * w, 0);
  setBone("neck", -0.04 * w + 0.4 * x.headPitch + 0.15 * bw, 0.4 * x.headYaw, 0);
  setBone("head", 0.05 * s + 0.6 * x.headPitch - 0.3 * cr + 0.4 * rd, 0.6 * x.headYaw, 0);
  // arms: lowered from the T-pose, swinging opposite the legs; hands on the lap when seated,
  // clasped behind the back while looking at something
  // clasped in front (shopkeeper) or holding a book up (reader)
  const swing = 0.45 * sp * w * (1 - x.look), lk = x.look, keep = Math.max(0, 1 - lk - c);
  // relaxed arms hang ~75 degrees down from the T-pose, close to the body
  setBone("leftUpperArm", swing - 0.35 * s * keep + 0.55 * lk - 0.25 * c - 0.1 * rd, 0, -1.3 - 0.12 * s * keep - 0.02 * lk - 0.04 * c);
  setBone("rightUpperArm", -swing - 0.35 * s * keep + 0.55 * lk - 0.25 * c - 0.1 * rd, 0, 1.3 + 0.12 * s * keep + 0.02 * lk + 0.04 * c);
  setBone("leftLowerArm", 0, (-0.25 - 0.25 * w - 0.55 * s) * keep + 1.0 * lk, 0);
  setBone("rightLowerArm", 0, (0.25 + 0.25 * w + 0.55 * s) * keep - 1.0 * lk, 0);
  // hands clasped in front of the belly (shopkeeper), or holding a book together on the lap (reader)
  aimForearm("left", CLASP_L, c);
  aimForearm("right", CLASP_R, c);
  aimForearm("left", LAP_L, rd * s);
  aimForearm("right", LAP_R, rd * s);
  // palms together in front of the chest
  if ((x.pray || 0) > 0.001) {
    bone("leftUpperArm").quaternion.slerp(Q.setFromEuler(eul(-0.45, 0, -1.25)), x.pray);
    bone("rightUpperArm").quaternion.slerp(Q.setFromEuler(eul(-0.45, 0, 1.25)), x.pray);
    aimForearm("left", PRAY_L, x.pray);
    aimForearm("right", PRAY_R, x.pray);
  }
  // reach toward a touched object with the right arm
  if (x.touch > 0.001) {
    Q.setFromUnitVectors(flip(ARM_REST.clone()), flip(x.touchDir.clone()));
    bone("rightUpperArm").quaternion.slerp(Q, x.touch);
    bone("rightLowerArm").quaternion.slerp(QI, x.touch);
  }
  // wave at the viewer with the right hand
  if (x.wave > 0.001) {
    bone("rightUpperArm").quaternion.slerp(Q.setFromEuler(eul(0, 0.25, -1.05)), x.wave);
    bone("rightLowerArm").quaternion.slerp(Q.setFromEuler(eul(0, 0, -0.55 + 0.4 * Math.sin(t * 10))), x.wave);
  }
  const hips = bone("hips");
  hips.position.y = r.hips + (w ? 0.025 * Math.cos(2 * ph) * w - 0.015 * w : 0) - 0.42 * cr;
}

// ---------------------------------------------------------------- behaviour
const SPEED = 1.15, RUN = 2.8, CYCLE = 1.25; // m/s, metres per two steps
const keys = new Set();
const st = {
  mode: "idle", path: [], phase: 0, walk: 0, sit: 0, heading: 0, onArrive: null, seat: null,
  stair: 0, touch: 0, crouch: 0, look: 0, wave: 0, headYaw: 0, headPitch: 0, actionT: 0, focus: null,
};
const mouse = { inside: false, point: new THREE.Vector3() };

function headingTo(dir) { return Math.atan2(dir.x, dir.z); }
function walkTo(target, onArrive = null) {
  const go = () => {
    st.path = findPath(actor.position, target);
    st.onArrive = onArrive;
    st.mode = st.path.length ? "walk" : "idle";
    if (!st.path.length && onArrive) onArrive();
  };
  if (st.mode === "sit" || st.mode === "sitting") standUp(go); else go();
}
function arrivedNear(p, tol = 0.45) {
  return Math.hypot(actor.position.x - p.x, actor.position.z - p.z) <= tol;
}
function turnThen(heading, then) {
  st.mode = "turn";
  st.turnTo = heading;
  st.afterTurn = then;
}
function sitAt(h, after = null) {
  const approach = h.seat.clone().addScaledVector(h.facing, h.approach);
  walkTo(approach, () => {
    if (!arrivedNear(approach)) { st.mode = "idle"; flash(`${h.label}까지 가는 길을 찾지 못했어요`); return; }
    turnThen(headingTo(h.facing), () => {
      st.mode = "sitting";
      st.seat = h;
      st.sitFrom = actor.position.clone();
      st.sitT = 0;
      st.onSat = () => { if (h.say) say(h.say); after && after(); };
    });
  });
}
function touchAt(h, after = null) {
  walkTo(h.stand, () => {
    if (!arrivedNear(h.stand, 0.6)) { st.mode = "idle"; flash(`${h.label}까지 가는 길을 찾지 못했어요`); return; }
    turnThen(headingTo(h.facing), () => {
      st.mode = "touch"; st.focus = h; st.actionT = 2.6;
      if (h.say) say(h.say);
      st.onAction = after;
    });
  });
}
function lookAt(h, after = null) {
  const start = () => {
    const d = h.target.clone().sub(actor.position).setY(0);
    turnThen(headingTo(d), () => {
      st.mode = "look"; st.focus = h; st.actionT = 3.2;
      if (h.say) say(h.say);
      st.onAction = after;
    });
  };
  if (st.mode === "sit") { // gaze from the seat without getting up
    st.focus = h; st.lookSeated = 3.2; if (h.say) say(h.say);
    after && setTimeout(after, 3200);
  } else start();
}
function talkTo(n, after = null) {
  const spot = n.group.position.clone().addScaledVector(n.facing, n.seated ? 0.95 : 0.85);
  walkTo(spot, () => {
    const d = n.group.position.clone().sub(actor.position).setY(0);
    turnThen(headingTo(d), () => {
      st.mode = "talk"; st.actionT = 3.4;
      st.focus = { target: n.headWorld() };
      n.talkT = 3.4;
      const lines = n.h.lines || [];
      if (lines.length) sayAs(n, lines[n.line++ % lines.length]);
      st.onAction = after;
    });
  });
}
function prayAt(h, after = null) {
  walkTo(h.stand, () => {
    if (!arrivedNear(h.stand, 0.8)) { st.mode = "idle"; flash(`${h.label}까지 가는 길을 찾지 못했어요`); return; }
    turnThen(headingTo(h.facing), () => {
      st.mode = "pray"; st.focus = h; st.actionT = 3.6;
      if (h.say) setTimeout(() => say(h.say), 1400);
      st.onAction = after;
    });
  });
}
function doHotspot(h, after = null) {
  if (h.action === "pray") return prayAt(h, after);
  if (h.action === "sit") sitAt(h, after);
  else if (h.action === "touch") touchAt(h, after);
  else lookAt(h, after);
}
function nearestSeat(maxDist = Infinity) {
  let best = null, bd = maxDist;
  for (const h of seats()) { const d = actor.position.distanceTo(h.seat); if (d < bd) { bd = d; best = h; } }
  return best;
}
function wave() {
  if (st.mode === "sit" || st.mode === "sitting" || st.mode === "standing") { st.waveT = 2.2; say("안녕!"); return; }
  st.path = []; st.onArrive = null;
  const toCam = activeCam().position.clone().sub(actor.position).setY(0);
  turnThen(headingTo(toCam), () => { st.mode = "wave"; st.actionT = 2.2; say("안녕!"); });
}
function standUp(done) {
  st.mode = "standing";
  st.sitT = 0;
  st.onStood = done;
}
function flash(msg) {
  const h = $("hint"), old = h.dataset.text || h.textContent;
  h.dataset.text = old;
  h.textContent = msg;
  clearTimeout(flash.t);
  flash.t = setTimeout(() => { h.textContent = old; }, 2500);
}
let bubbleT = 0;
function say(text, secs = 2.8) {
  $("bubble").textContent = text;
  $("bubble").classList.add("show");
  bubbleT = secs;
}
function sayAs(n, text, secs = 3.2) {
  n.bubble.textContent = text;
  n.bubble.classList.add("show");
  n.bubbleT = secs;
}

function update(dt, t) {
  if (!vrm) return;
  const drive = keyDirection();
  const running = keys.has("ShiftLeft") || keys.has("ShiftRight");
  if (drive) {
    scriptId++;
    if (st.mode === "sit") standUp(null);
    else if (st.mode !== "sitting" && st.mode !== "standing") { st.mode = "drive"; st.path = []; st.onArrive = null; }
  } else if (st.mode === "drive") st.mode = "idle";

  // slope ahead: positive climbing, negative descending (stairs, ramps)
  const fwd = new THREE.Vector3(Math.sin(st.heading), 0, Math.cos(st.heading));
  const here = heightAt(actor.position.x, actor.position.z, actor.position.y);
  const ahead = heightAt(actor.position.x + fwd.x * 0.35, actor.position.z + fwd.z * 0.35, here);
  const moving = st.mode === "walk" || (st.mode === "drive" && drive);
  st.stair = THREE.MathUtils.damp(st.stair, moving ? THREE.MathUtils.clamp((ahead - here) / 0.12, -1, 1) : 0, 6, dt);

  const speed = (running && (drive || st.mode === "walk") ? RUN : SPEED) * (1 - 0.3 * Math.abs(st.stair));
  st.speed = THREE.MathUtils.damp(st.speed ?? SPEED, speed, 5, dt);
  if (st.mode === "drive" && drive) {
    const step = st.speed * dt * Math.max(st.walk, 0.3);
    tryMove(drive.x * step, drive.z * step);
    turnToward(headingTo(drive), dt, 10);
  }
  if (st.mode === "walk") {
    const target = st.path[0];
    const d = new THREE.Vector3(target.x - actor.position.x, 0, target.z - actor.position.z);
    const dist = d.length();
    const step = st.speed * dt * st.walk;
    if (dist < 0.05 || dist < step) {
      st.path.shift();
      if (!st.path.length) { st.mode = "idle"; const f = st.onArrive; st.onArrive = null; if (f) f(); }
    } else {
      d.normalize();
      actor.position.x += d.x * step;
      actor.position.z += d.z * step;
      turnToward(headingTo(d), dt, 8);
    }
  } else if (st.mode === "turn") {
    if (turnToward(st.turnTo, dt, 6)) { st.mode = "idle"; const f = st.afterTurn; st.afterTurn = null; f && f(); }
  } else if (st.mode === "touch" || st.mode === "look" || st.mode === "wave" || st.mode === "talk" || st.mode === "pray") {
    st.actionT -= dt;
    if (st.mode === "wave") turnToward(headingTo(activeCam().position.clone().sub(actor.position)), dt, 6);
    if (st.actionT <= 0) { st.mode = "idle"; st.focus = null; const f = st.onAction; st.onAction = null; f && f(); }
  }
  // blends
  st.walk = THREE.MathUtils.damp(st.walk, moving ? 1 : 0, 6, dt);
  st.run = THREE.MathUtils.damp(st.run ?? 0, moving && st.speed > 1.8 ? 1 : 0, 5, dt);
  st.phase += (st.speed * st.walk * dt / (CYCLE * (1 + 0.45 * st.run) * (1 - 0.35 * Math.abs(st.stair)))) * Math.PI * 2;
  st.touch = THREE.MathUtils.damp(st.touch, st.mode === "touch" && st.actionT < 2.3 ? 1 : 0, 7, dt);
  st.look = THREE.MathUtils.damp(st.look, st.mode === "look" ? 1 : 0, 5, dt);
  st.waveT = Math.max(0, (st.waveT ?? 0) - dt);
  st.wave = THREE.MathUtils.damp(st.wave, st.mode === "wave" || st.waveT > 0 ? 1 : 0, 8, dt);
  st.pray = THREE.MathUtils.damp(st.pray ?? 0, st.mode === "pray" ? 1 : 0, 5, dt);
  st.prayBow = THREE.MathUtils.damp(st.prayBow ?? 0, st.mode === "pray" && st.actionT < 2.6 && st.actionT > 1.0 ? 1 : 0, 4, dt);

  // sit / stand transitions move the hips between the floor and the seat
  const seat = st.seat;
  if (seat && (st.mode === "sitting" || st.mode === "standing" || st.mode === "sit")) {
    const seatRoot = seat.seat.clone().addScaledVector(seat.facing, Math.max(0.06, (seat.edge ?? 0.2) - 0.15));
    seatRoot.y = seat.seat.y + 0.09 - rig.hips;
    const standRoot = seat.seat.clone().addScaledVector(seat.facing, seat.approach);
    standRoot.y = heightAt(standRoot.x, standRoot.z, seat.floorY);
    if (st.mode === "sitting" || st.mode === "standing") {
      st.sitT = Math.min(1, st.sitT + dt / 0.9);
      const k = st.sitT * st.sitT * (3 - 2 * st.sitT);
      if (st.mode === "sitting") { st.sit = k; actor.position.lerpVectors(st.sitFrom, seatRoot, k); }
      else { st.sit = 1 - k; actor.position.lerpVectors(seatRoot, standRoot, k); }
      if (st.sitT >= 1) {
        if (st.mode === "sitting") { st.mode = "sit"; const f = st.onSat; st.onSat = null; f && f(); }
        else { st.mode = "idle"; st.sit = 0; st.seat = null; const f = st.onStood; st.onStood = null; f && f(); }
      }
    }
  }
  if (st.mode !== "sit" && st.mode !== "sitting" && st.mode !== "standing") actor.position.y = here;
  actor.rotation.y = st.heading;
  actor.updateMatrixWorld(true);

  // where the head and eyes point: a hotspot being used, else the mouse cursor when idle
  if (st.lookSeated > 0) st.lookSeated -= dt; else if (st.mode === "sit") st.focus = null;
  let gaze = null;
  if (st.focus && (st.mode === "look" || st.mode === "touch" || st.mode === "talk" || st.lookSeated > 0)) gaze = st.focus.target;
  else if (st.mode === "pray") gaze = null;
  else if (st.mode === "wave") gaze = activeCam().position;
  else if (mouse.inside && !moving) gaze = mouse.point;
  const headWorld = actor.localToWorld(rig.head.clone().setY(rig.head.y - 0.42 * st.crouch - (rig.hips - 0.55) * st.sit));
  let yaw = 0, pitch = 0;
  if (gaze) {
    const local = actor.worldToLocal(gaze.clone()).sub(actor.worldToLocal(headWorld.clone()));
    yaw = THREE.MathUtils.clamp(Math.atan2(local.x, local.z), -1.1, 1.1);
    pitch = THREE.MathUtils.clamp(Math.atan2(-local.y, Math.hypot(local.x, local.z)), -0.5, 0.6);
    eyeTarget.position.copy(gaze);
  } else {
    eyeTarget.position.copy(headWorld).addScaledVector(fwd, 3);
  }
  st.headYaw = THREE.MathUtils.damp(st.headYaw, yaw, 5, dt);
  st.headPitch = THREE.MathUtils.damp(st.headPitch, pitch, 5, dt);

  // reaching: low targets make her crouch, and the arm aims from the (lowered) shoulder
  let touchDir = new THREE.Vector3(0, -1, 0);
  let crouchGoal = 0;
  if (st.focus && st.mode === "touch") {
    const shoulder = rig.rShoulder.clone();
    const tl = actor.worldToLocal(st.focus.target.clone());
    crouchGoal = THREE.MathUtils.clamp((0.75 - tl.y) / 0.5, 0, 1);
    shoulder.y -= 0.42 * st.crouch;
    touchDir = tl.sub(shoulder).normalize();
  }
  st.crouch = THREE.MathUtils.damp(st.crouch, crouchGoal * (st.mode === "touch" ? 1 : 0), 6, dt);
  // how stepped the ground is right here (stairs): blends in the foot planting below
  const behind = heightAt(actor.position.x - fwd.x * 0.3, actor.position.z - fwd.z * 0.3, here);
  const rough = THREE.MathUtils.clamp((Math.max(Math.abs(ahead - here), Math.abs(behind - here)) - 0.03) / 0.08, 0, 1);
  st.rough = THREE.MathUtils.damp(st.rough ?? 0, rough, 8, dt);
  applyPose(vrm, rig, t, st.phase, st.walk * (1 + 0.35 * st.run), st.sit,
    { stair: st.stair, touch: st.touch, touchDir, crouch: st.crouch, look: st.look, wave: st.wave,
      headYaw: st.headYaw, headPitch: st.headPitch, pray: st.pray, bow: 0.6 * st.prayBow });
  vrm.update(dt);
  plantFeet(dt, here);

  const floorY = st.sit > 0.01 && seat ? seat.floorY : actor.position.y;
  shadow.position.set(actor.position.x, floorY + 0.012, actor.position.z);
  shadow.material.opacity = 1 - 0.4 * st.sit;

  if (bubbleT > 0) { bubbleT -= dt; if (bubbleT <= 0) $("bubble").classList.remove("show"); }
}
// On steps, the body height comes from the stance foot: whichever foot would sink deepest decides,
// so each footfall lands on a tread and the swing foot lifts over the riser.
const FOOT_L = new THREE.Vector3(), FOOT_R = new THREE.Vector3();
function plantFeet(dt, here) {
  if (st.mode === "sit" || st.mode === "sitting" || st.mode === "standing" || st.rough < 0.02) { st.rootY = actor.position.y; return; }
  const need = (bone, out) => {
    vrm.humanoid.getRawBoneNode(bone).getWorldPosition(out);
    return heightAt(out.x, out.z, here) - (out.y - actor.position.y - rig.ankle);
  };
  const planted = Math.max(need("leftFoot", FOOT_L), need("rightFoot", FOOT_R));
  st.rootY = THREE.MathUtils.damp(st.rootY ?? planted, planted, 16, dt);
  actor.position.y = THREE.MathUtils.lerp(here, st.rootY, st.rough);
  actor.updateMatrixWorld(true);
}
function keyDirection() {
  const f = (keys.has("KeyW") || keys.has("ArrowUp") ? 1 : 0) - (keys.has("KeyS") || keys.has("ArrowDown") ? 1 : 0);
  const r = (keys.has("KeyD") || keys.has("ArrowRight") ? 1 : 0) - (keys.has("KeyA") || keys.has("ArrowLeft") ? 1 : 0);
  if (!f && !r) return null;
  const fwd = activeCam().getWorldDirection(new THREE.Vector3()).setY(0).normalize();
  const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0));
  return fwd.multiplyScalar(f).addScaledVector(right, r).normalize();
}
function tryMove(dx, dz) {
  // move; if blocked, veer 30/60 degrees around small things, then slide along a wall on one axis
  const p = actor.position;
  const ok = (x, z) => walkable(...toCell(x, z)) && walkable(...toCell(x + dx * 2, z + dz * 2));
  for (const a of [0, 0.52, -0.52, 1.05, -1.05]) {
    const c = Math.cos(a), s = Math.sin(a), mx = dx * c - dz * s, mz = dx * s + dz * c;
    if (ok(p.x + mx, p.z + mz)) { p.x += mx; p.z += mz; return; }
  }
  if (walkable(...toCell(p.x + dx, p.z))) p.x += dx;
  else if (walkable(...toCell(p.x, p.z + dz))) p.z += dz;
}
function turnToward(target, dt, rate) {
  let diff = ((target - st.heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
  st.heading += diff * Math.min(1, rate * dt);
  return Math.abs(diff) < 0.03;
}

// ---------------------------------------------------------------- scripted stroll (처음부터)
function snap(x, z) {
  const c = nearestCell(...toCell(x, z));
  return cellPos(...c);
}
let scriptId = 0;
function runScript() {
  const my = ++scriptId;
  const spawn = snap(info.spawn[0], info.spawn[2]);
  Object.assign(st, { mode: "idle", sit: 0, path: [], seat: null, focus: null });
  actor.position.copy(spawn);
  st.heading = Math.PI; // walk away from the viewer, into the picture
  const depth = Math.max(walkMinZ, spawn.z - 20) - spawn.z; // negative: into the picture
  // wander in, go deep (up the stairs where there are some), then try every hotspot on the way back, sitting last
  const order = [...hotspots].sort((a, b) => (a.action === "sit") - (b.action === "sit"));
  const legs = [
    () => walkTo(snap(spawn.x - 0.5, spawn.z + depth * 0.45), next),
    () => walkTo(snap(spawn.x + 0.4, spawn.z + depth * 0.9), next),
    ...npcs.map((n) => () => talkTo(n, () => setTimeout(next, 400))),
    ...order.map((h) => () => doHotspot(h, () => setTimeout(next, h.action === "sit" ? 3000 : 400))),
    () => walkTo(spawn, next),
    () => { if ($("chkLoop").checked) runScript(); },
  ];
  let i = 0;
  function next() { if (my !== scriptId) return; const f = legs[i++]; f && f(); }
  next();
}

// ---------------------------------------------------------------- view modes & layout
let view = "paint";
const activeCam = () => (view === "paint" ? paintCam : freeCam);
const followPrev = new THREE.Vector3();
function setView(v) {
  view = v;
  document.body.classList.toggle("split", v === "split");
  for (const [id, m] of [["btnPaint", "paint"], ["btnFree", "free"], ["btnSplit", "split"]]) $(id).classList.toggle("on", v === m);
  controls.enabled = v !== "paint";
  followPrev.set(0, 0, 0);
  if (v !== "paint" && vrm) {
    // start near the painting viewpoint (single-image splats break down far from it), turned toward the character
    const c = actor.position.clone().add(new THREE.Vector3(0, 0.9 - 0.3 * st.sit, 0));
    controls.target.copy(c);
    const fromPaint = paintCam.position.clone().sub(c);
    const dist = Math.min(fromPaint.length(), 3.2);
    fromPaint.normalize().applyAxisAngle(new THREE.Vector3(0, 1, 0), v === "split" ? 0.45 : 0.25);
    freeCam.position.copy(c).addScaledVector(fromPaint, dist);
    freeCam.position.y = Math.max(freeCam.position.y, c.y + 0.3);
    freeCam.lookAt(c);
  }
  layout();
}
const rtTop = new THREE.WebGLRenderTarget(1, 1, { samples: 0 });
const rtBottom = new THREE.WebGLRenderTarget(1, 1, { samples: 0 });
const overlay = new THREE.Scene();
const orthoCam = new THREE.OrthographicCamera(0, 1, 1, 0, -1, 1);
const quadTop = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: rtTop.texture, side: THREE.DoubleSide }));
const quadBottom = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: rtBottom.texture, side: THREE.DoubleSide }));
overlay.add(quadTop, quadBottom);
let rects = {};
function layout() {
  const W = innerWidth, Hh = innerHeight;
  const pr = renderer.getPixelRatio();
  if (view === "paint") {
    let w = W, h = W / ASPECT;
    if (h > Hh) { h = Hh; w = Hh * ASPECT; }
    renderer.setSize(Math.round(w), Math.round(h));
    rects = { paint: [0, 0, w, h] };
  } else if (view === "free") {
    renderer.setSize(W, Hh);
    freeCam.aspect = W / Hh;
    freeCam.updateProjectionMatrix();
    rects = { free: [0, 0, W, Hh] };
  } else {
    renderer.setSize(W, Hh);
    const half = Hh / 2;
    let w = W, h = W / ASPECT;
    if (h > half) { h = half; w = half * ASPECT; }
    const top = [(W - w) / 2, 0, w, h];
    const bottom = [0, half, W, half];
    rects = { paint: top, free: bottom };
    rtTop.setSize(Math.round(w * pr), Math.round(h * pr));
    rtBottom.setSize(Math.round(W * pr), Math.round(half * pr));
    freeCam.aspect = W / half;
    freeCam.updateProjectionMatrix();
    orthoCam.left = 0; orthoCam.right = W; orthoCam.top = 0; orthoCam.bottom = Hh;
    orthoCam.updateProjectionMatrix();
    const place = (q, [x, y, w2, h2]) => { q.position.set(x + w2 / 2, y + h2 / 2, 0); q.scale.set(w2, -h2, 1); };
    place(quadTop, top);
    place(quadBottom, bottom);
    $("labTop").style.top = `${top[1] + top[3] - 26}px`;
    $("labBottom").style.top = `${Hh - 26}px`;
  }
  paintCam.aspect = ASPECT;
  paintCam.updateProjectionMatrix();
}
addEventListener("resize", layout);

// speech bubbles follow heads in whichever view shows the painting (or the free view)
const tmpV = new THREE.Vector3();
function placeBubbleAt(b, world) {
  if (!b.classList.contains("show")) return;
  const key = rects.paint ? "paint" : "free";
  const cam = key === "paint" ? paintCam : freeCam, rc = rects[key];
  tmpV.copy(world).project(cam);
  const r = canvas.getBoundingClientRect();
  b.style.left = `${r.left + rc[0] + (tmpV.x * 0.5 + 0.5) * rc[2]}px`;
  b.style.top = `${r.top + rc[1] + (-tmpV.y * 0.5 + 0.5) * rc[3]}px`;
  b.style.visibility = tmpV.z < 1 ? "visible" : "hidden";
}
function placeBubble() {
  placeBubbleAt($("bubble"), actor.position.clone().add(new THREE.Vector3(0, 1.75 - 0.5 * st.sit - 0.4 * st.crouch, 0)));
  for (const n of npcs) placeBubbleAt(n.bubble, n.group.position.clone().add(new THREE.Vector3(0, n.seated ? 1.3 : 1.8, 0)));
}

// ---------------------------------------------------------------- NPCs: background characters with a role
const npcs = [];
let npcLoad = 0;
async function setupNpcs() {
  const my = ++npcLoad;
  for (const n of npcs) { scene.remove(n.group); VRMUtils.deepDispose(n.vrm.scene); n.bubble.remove(); }
  npcs.length = 0;
  for (const h of (info.hotspots || []).filter((x) => x.action === "npc")) {
    const m = await loadVrmModel(h.model || "assets/npc_casual.vrm");
    if (my !== npcLoad) { VRMUtils.deepDispose(m.vrm.scene); return; } // scene changed while loading
    if (h.tint) tintVrm(m.vrm, h.tint);
    const g = new THREE.Group();
    g.add(m.vrm.scene);
    const facing = new THREE.Vector3().fromArray(h.facing).normalize();
    const at = new THREE.Vector3().fromArray(h.at);
    const seated = h.pose === "sit";
    if (seated) { g.position.copy(at).addScaledVector(facing, Math.max(0.06, (h.edge ?? 0.2) - 0.15)); g.position.y = at.y + 0.09 - m.rig.hips; }
    else g.position.set(at.x, at.y, at.z);
    g.rotation.y = headingTo(facing);
    const pick = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, seated ? 1.3 : 1.7, 10), new THREE.MeshBasicMaterial());
    pick.position.y = seated ? m.rig.hips : 0.85;
    pick.visible = false;
    g.add(pick);
    scene.add(g);
    const bubble = document.createElement("div");
    bubble.className = "bubble npc";
    document.body.appendChild(bubble);
    const eye = new THREE.Object3D();
    scene.add(eye);
    if (m.vrm.lookAt) m.vrm.lookAt.target = eye;
    const n = { h, vrm: m.vrm, rig: m.rig, group: g, pick, facing, seated, bubble, bubbleT: 0, eye,
      line: 0, greetCd: 0, bowT: 0, bow: 0, talkT: 0, headYaw: 0, headPitch: 0, read: h.idle === "read" ? 1 : 0,
      headWorld() { return g.localToWorld(m.rig.head.clone().setY(seated ? m.rig.head.y - (m.rig.hips - 0.55) : m.rig.head.y)); } };
    pick.userData.npc = n;
    npcs.push(n);
  }
}
// recolour clothes/hair by material name, e.g. { "Tops": "#3d5a73" }; the shade colour follows, a little darker
function tintVrm(v, tint) {
  v.scene.traverse((o) => {
    if (!o.isMesh) return;
    for (const mat of [].concat(o.material)) {
      const key = Object.keys(tint).find((k) => mat.name === k || mat.name.startsWith(`${k} `) || mat.name.startsWith(`${k}(`));
      if (!key || /outline/i.test(mat.name)) continue;
      const c = new THREE.Color(tint[key]);
      if (mat.color) mat.color.copy(c);
      if (mat.shadeColorFactor) mat.shadeColorFactor.copy(c).multiplyScalar(0.72);
    }
  });
}
function updateNpcs(dt, t) {
  const playerHead = actor.localToWorld(rig.head.clone().setY(rig.head.y - (rig.hips - 0.55) * st.sit));
  for (const n of npcs) {
    n.group.updateMatrixWorld(true);
    const dist = Math.hypot(actor.position.x - n.group.position.x, actor.position.z - n.group.position.z);
    // greet once when the player comes close, then wait before greeting again
    n.greetCd -= dt;
    if (dist < 2.4 && n.greetCd <= 0) {
      n.greetCd = 25;
      if (n.h.greet) { sayAs(n, n.h.greet); n.bowT = 1.3; }
    }
    n.bowT -= dt; n.talkT -= dt;
    n.bow = THREE.MathUtils.damp(n.bow, n.bowT > 0 ? 1 : 0, 6, dt);
    const attentive = dist < 3.2 || n.talkT > 0;
    if (n.h.idle === "read") n.read = THREE.MathUtils.damp(n.read, attentive ? 0.35 : 1, 3, dt);
    // gaze at the player when near, otherwise idle glances (or the book)
    let yaw = 0, pitch = 0;
    const head = n.headWorld();
    if (attentive) {
      const local = n.group.worldToLocal(playerHead.clone()).sub(n.group.worldToLocal(head.clone()));
      yaw = THREE.MathUtils.clamp(Math.atan2(local.x, local.z), -1.1, 1.1);
      pitch = THREE.MathUtils.clamp(Math.atan2(-local.y, Math.hypot(local.x, local.z)), -0.5, 0.6);
      n.eye.position.copy(playerHead);
    } else {
      yaw = n.h.idle === "read" ? 0 : 0.35 * Math.sin(t * 0.35 + n.group.position.x);
      n.eye.position.copy(head).add(n.facing.clone().multiplyScalar(3)).add(new THREE.Vector3(0, -0.8 * n.read, 0));
    }
    n.headYaw = THREE.MathUtils.damp(n.headYaw, yaw, 4, dt);
    n.headPitch = THREE.MathUtils.damp(n.headPitch, pitch, 4, dt);
    applyPose(n.vrm, n.rig, t, 0, 0, n.seated ? 1 : 0,
      { stair: 0, touch: 0, touchDir: new THREE.Vector3(0, -1, 0), crouch: 0, look: 0, wave: 0,
        headYaw: n.headYaw, headPitch: n.headPitch, clasp: n.h.idle === "clasp" ? 1 : 0, read: n.read, bow: n.bow });
    n.vrm.update(dt);
    if (n.bubbleT > 0) { n.bubbleT -= dt; if (n.bubbleT <= 0) n.bubble.classList.remove("show"); }
  }
}

// ---------------------------------------------------------------- input
const ray = new THREE.Raycaster();
function pickAt(clientX, clientY) {
  const r = canvas.getBoundingClientRect();
  const px = clientX - r.left, py = clientY - r.top;
  for (const [k, rc] of Object.entries(rects)) {
    if (px >= rc[0] && px <= rc[0] + rc[2] && py >= rc[1] && py <= rc[1] + rc[3]) {
      const cam = k === "paint" ? paintCam : freeCam;
      ray.setFromCamera(new THREE.Vector2(((px - rc[0]) / rc[2]) * 2 - 1, -((py - rc[1]) / rc[3]) * 2 + 1), cam);
      const ground = ray.intersectObject(groundMesh)[0] || null;
      const me = ray.intersectObject(actorPick)[0];
      const npcHit = ray.intersectObjects(npcs.map((n) => n.pick))[0];
      const spots = ray.intersectObjects(hotspotGroup.children);
      const spot = spots.find((s) => !ground || s.distance < ground.distance + 0.6) || null;
      return { ray: ray.ray.clone(), ground, me, npc: npcHit && npcHit.object.userData.npc, spot: spot && spot.object.userData.hotspot };
    }
  }
  return null;
}
let downAt = null;
canvas.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 6) return;
  const p = pickAt(e.clientX, e.clientY);
  if (!p) return;
  scriptId++; // a click takes over from the scripted stroll
  if (p.me) wave();
  else if (p.npc) talkTo(p.npc);
  else if (p.spot) doHotspot(p.spot);
  else if (p.ground) walkTo(p.ground.point);
});
canvas.addEventListener("pointermove", (e) => {
  const p = pickAt(e.clientX, e.clientY);
  mouse.inside = !!p;
  if (p) mouse.point.copy(p.ground ? p.ground.point.clone().setY(p.ground.point.y + 1.2) : p.ray.at(4, new THREE.Vector3()));
  const tip = $("tip");
  const label = p && (p.me ? "캐릭터 · 인사하기" : p.npc ? `${p.npc.h.label} · 말 걸기`
    : p.spot ? `${p.spot.label} · ${ACTION_VERB[p.spot.action]}` : "");
  canvas.style.cursor = label ? "pointer" : "default";
  tip.textContent = label || "";
  tip.style.display = label ? "block" : "none";
  tip.style.left = `${e.clientX + 14}px`;
  tip.style.top = `${e.clientY + 14}px`;
});
canvas.addEventListener("pointerleave", () => { mouse.inside = false; $("tip").style.display = "none"; });

const MOVE_KEYS = ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "ShiftLeft", "ShiftRight"];
addEventListener("keydown", (e) => {
  if (e.target.closest && e.target.closest("input, textarea, select")) return;
  if (MOVE_KEYS.includes(e.code)) { keys.add(e.code); e.preventDefault(); }
  if ((e.code === "Space" || e.code === "KeyE") && !e.repeat) {
    e.preventDefault();
    scriptId++;
    const h = nearestSeat(5);
    if (st.mode === "sit") standUp(null);
    else if (h) sitAt(h);
    else flash(seats().length ? "앉을 곳 가까이(5m 안)에서 Space를 누르면 앉아요" : "이 장면에는 앉을 곳이 없어요");
  }
  if (e.code === "KeyQ" && !e.repeat) { scriptId++; wave(); }
});
addEventListener("keyup", (e) => keys.delete(e.code));
addEventListener("blur", () => keys.clear());

for (const b of document.querySelectorAll(".bar button, .bar input")) b.addEventListener("click", () => b.blur());
$("btnStart").onclick = () => runScript();
$("btnSit").onclick = () => { scriptId++; const h = nearestSeat(); if (h) sitAt(h); };
$("btnPaint").onclick = () => setView("paint");
$("btnFree").onclick = () => setView("free");
$("btnSplit").onclick = () => setView("split");

addEventListener("dragover", (e) => { e.preventDefault(); document.body.classList.add("dragging"); });
addEventListener("dragleave", () => document.body.classList.remove("dragging"));
addEventListener("drop", async (e) => {
  e.preventDefault();
  document.body.classList.remove("dragging");
  const f = [...e.dataTransfer.files].find((x) => x.name.toLowerCase().endsWith(".vrm"));
  if (!f) return;
  const url = URL.createObjectURL(f);
  try { await loadVRM(url); } finally { URL.revokeObjectURL(url); }
});

// ---------------------------------------------------------------- scenes
async function loadScene(id) {
  entry = manifest.find((m) => m.id === id) || manifest[0];
  $("selScene").value = entry.id;
  scriptId++;
  $("loading").textContent = `${entry.name} 불러오는 중…`;
  $("loading").classList.remove("done");
  info = await fetch(`scenes/${entry.id}/scene.json`).then((r) => r.json());
  ASPECT = info.camera.aspect;
  if (splat) { scene.remove(splat); splat.dispose?.(); }
  splat = new SplatMesh({ url: `scenes/${entry.id}/${entry.splat || "splat.sog"}` });
  splat.matrixAutoUpdate = false;
  splat.matrix.fromArray(info.splatToWorld);
  scene.add(splat);
  const mine = splat;
  splat.initialized.then(() => { if (mine === splat) $("loading").classList.add("done"); });
  paintCam.fov = info.camera.fovY;
  paintCam.position.fromArray(info.camera.position);
  paintCam.quaternion.setFromRotationMatrix(
    new THREE.Matrix4().setFromMatrix3(new THREE.Matrix3().fromArray(info.camera.rotation)));
  setupWalkable();
  setupHotspots();
  setupNpcs();
  const L = entry.light || {};
  hemi.color.set(L.sky ?? 0xcfe3ff);
  hemi.groundColor.set(L.ground ?? 0xb59a7a);
  hemi.intensity = L.hemi ?? 1.4;
  sun.color.set(L.sun ?? 0xfff1dc);
  sun.intensity = L.sunIntensity ?? 2.4;
  sun.position.fromArray(L.sunDir ?? [-4, 8, 3]);
  $("btnSit").disabled = !seats().length;
  Object.assign(st, { mode: "idle", path: [], sit: 0, walk: 0, seat: null, focus: null, onArrive: null });
  actor.position.copy(snap(info.spawn[0], info.spawn[2] - 1.5));
  st.heading = 0; // face the viewer at first
  $("bubble").classList.remove("show");
  setView(view);
}
const sel = $("selScene");
for (const m of manifest) sel.add(new Option(m.name, m.id));
sel.onchange = () => { sel.blur(); loadScene(sel.value); };

// ---------------------------------------------------------------- start
await loadVRM("assets/avatar.vrm");
await loadScene(new URLSearchParams(location.search).get("scene") || manifest[0].id);
sel.value = entry.id;
// debug: advance the simulation without rendering (the Browser pane throttles frames when unfocused)
let simT = 0;
function simulate(seconds, dt = 1 / 30) {
  for (let i = 0; i < seconds / dt; i++) { simT += dt; update(dt, simT); updateNpcs(dt, simT); }
}
window.__app = { simulate, spark, renderer, layout, st, actor, setView, walkTo, doHotspot, sitAt, wave, talkTo, npcs, runScript, loadScene,
  freeCam, controls, get info() { return info; }, get hotspots() { return hotspots; }, get vrm() { return vrm; } };

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  update(dt, t);
  updateNpcs(dt, t);
  if (view !== "paint") {
    // follow the character while keeping the user's orbit offset
    const c = actor.position.clone().add(new THREE.Vector3(0, 1.0 - 0.35 * st.sit, 0));
    if (followPrev.lengthSq()) { const d = c.clone().sub(followPrev); freeCam.position.add(d); controls.target.add(d); }
    followPrev.copy(c);
    controls.update();
  }
  spark.updateInternal({ scene, camera: activeCam(), autoUpdate: true });
  if (view === "split") {
    renderer.setRenderTarget(rtTop);
    renderer.render(scene, paintCam);
    renderer.setRenderTarget(rtBottom);
    renderer.render(scene, freeCam);
    renderer.setRenderTarget(null);
    renderer.render(overlay, orthoCam);
  } else {
    renderer.render(scene, activeCam());
  }
  placeBubble();
});
