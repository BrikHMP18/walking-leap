// Walking Leap control panel. Talks to server.py: POST /api/<command>, live state on /api/stream.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
let config = null;
let S = {}; // the latest state from the stream
const FINGERS = [['IF', 0], ['MF', 4], ['RF', 8], ['TH', 12]];

// ------------------------------------------------------------------ commands
async function api(path, body) {
  try {
    const r = await fetch('/api/' + path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}),
    });
    const j = await r.json();
    if (!j.ok) toast(j.msg, true);
    return j;
  } catch (e) {
    toast('the panel server does not answer', true);
    return { ok: false };
  }
}
let toastTimer = null;
function toast(text, bad) {
  const t = $('toast');
  t.textContent = text;
  t.className = 'show' + (bad ? ' bad' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = ''), 2600);
}
function confirmBox(text) {
  return new Promise((done) => {
    $('confirm-text').textContent = text;
    $('confirm').hidden = false;
    const finish = (v) => { $('confirm').hidden = true; done(v); };
    $('confirm-yes').onclick = () => finish(true);
    $('confirm-no').onclick = () => finish(false);
  });
}

// ------------------------------------------------------------------ 3D view
const view = $('three');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x10141c);
const camera = new THREE.PerspectiveCamera(32, 1, 0.005, 10);
camera.up.set(0, 0, 1);
camera.position.set(-0.30, -0.34, 0.24);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(-0.03, -0.02, 0.06);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x1a1d24, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-0.25, -0.15, 0.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -0.3, right: 0.3, top: 0.3, bottom: -0.3, near: 0.05, far: 2 });
scene.add(sun);
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(2, 2),
  new THREE.MeshStandardMaterial({ color: 0x1a202b, roughness: 0.95 }),
);
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(0.6, 60, 0x2c3545, 0x212834);
grid.rotateX(Math.PI / 2);
grid.position.z = 0.0002;
scene.add(grid);

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

let model = null, real = null, ghost = null;
const quat = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]); // MuJoCo w x y z
const vec = (v) => new THREE.Vector3(v[0], v[1], v[2]);

function primitive(g) {
  const s = g.size;
  if (g.type === 6) return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
  if (g.type === 5) return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24).rotateX(Math.PI / 2);
  if (g.type === 3) return new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 16).rotateX(Math.PI / 2);
  if (g.type === 2) return new THREE.SphereGeometry(s[0], 20, 14);
  if (g.type === 4) return new THREE.SphereGeometry(1, 20, 14).scale(s[0], s[1], s[2]);
  return null;
}

function buildHand(bin, look) {
  const root = new THREE.Group();
  const bodies = model.bodies.map(() => {
    const o = new THREE.Object3D();
    o.matrixAutoUpdate = false;
    root.add(o);
    return o;
  });
  const samples = model.bodies.map(() => []);
  const cache = {};
  for (const g of model.geoms) {
    let geo;
    if (g.mesh) {
      const k = g.mesh.vert;
      if (!cache[k]) {
        const v = new Float32Array(bin, g.mesh.vert, g.mesh.nvert * 3);
        const f = new Uint32Array(bin, g.mesh.face, g.mesh.nface * 3);
        const b = new THREE.BufferGeometry();
        b.setAttribute('position', new THREE.BufferAttribute(v, 3));
        b.setIndex(new THREE.BufferAttribute(f, 1));
        b.computeVertexNormals();
        cache[k] = b;
      }
      geo = cache[k];
    } else geo = primitive(g);
    if (!geo) continue;
    const mesh = new THREE.Mesh(geo, look(g));
    mesh.position.copy(vec(g.pos));
    mesh.quaternion.copy(quat(g.quat));
    mesh.castShadow = !look.ghost;
    bodies[g.body].add(mesh);
    // Points to find the lowest one: every 25th vertex, in the body's frame.
    mesh.updateMatrix();
    const p = geo.attributes.position;
    const step = Math.max(1, Math.floor(p.count / 160));
    for (let i = 0; i < p.count; i += step) {
      samples[g.body].push(new THREE.Vector3().fromBufferAttribute(p, i).applyMatrix4(mesh.matrix));
    }
  }
  const local = model.bodies.map((b) => new THREE.Matrix4().compose(vec(b.pos), quat(b.quat), new THREE.Vector3(1, 1, 1)));
  scene.add(root);
  return { root, bodies, samples, local, world: model.bodies.map(() => new THREE.Matrix4()) };
}

const _t = new THREE.Matrix4(), _r = new THREE.Matrix4(), _a = new THREE.Vector3(), _p = new THREE.Vector3();
// Body poses from joint angles (the repo's order) and the palm's pose, as MuJoCo does.
function kinematics(hand, q, rootPos, rootQuat) {
  model.bodies.forEach((b, i) => {
    const m = hand.world[i];
    if (b.parent < 0) m.compose(rootPos, rootQuat, new THREE.Vector3(1, 1, 1));
    else {
      m.multiplyMatrices(hand.world[b.parent], hand.local[i]);
      if (b.joint) {
        const j = b.joint;
        m.multiply(_t.makeTranslation(j.pos[0], j.pos[1], j.pos[2]));
        m.multiply(_r.makeRotationAxis(_a.set(...j.axis).normalize(), q[j.index]));
        m.multiply(_t.makeTranslation(-j.pos[0], -j.pos[1], -j.pos[2]));
      }
    }
    hand.bodies[i].matrix.copy(m);
  });
}
// The palm's orientation that reads "down" = g in its own axes, heading kept straight.
function fromGravity(g) {
  const up = new THREE.Vector3(-g[0], -g[1], -g[2]).normalize();
  const x = new THREE.Vector3(1, 0, 0).addScaledVector(up, -up.x).normalize();
  const y = new THREE.Vector3().crossVectors(up, x);
  const m = new THREE.Matrix4().set(x.x, x.y, x.z, 0, y.x, y.y, y.z, 0, up.x, up.y, up.z, 0, 0, 0, 0, 1);
  return new THREE.Quaternion().setFromRotationMatrix(m);
}
function pose(hand, q, g) {
  kinematics(hand, q, new THREE.Vector3(0, 0, 0), fromGravity(g || [0, 0, -1]));
  let low = Infinity;
  hand.samples.forEach((pts, i) => {
    for (const p of pts) low = Math.min(low, _p.copy(p).applyMatrix4(hand.world[i]).z);
  });
  hand.root.position.z = Number.isFinite(low) ? -low : 0;
}
function selftest() {
  const hand = real;
  let worst = 0;
  for (const c of model.checks) {
    kinematics(hand, c.q, vec(c.root_pos), quat(c.root_quat));
    c.xpos.forEach((x, i) => {
      const p = new THREE.Vector3().setFromMatrixPosition(hand.world[i]);
      worst = Math.max(worst, p.distanceTo(vec(x)));
    });
  }
  const msg = `kinematics vs MuJoCo: worst ${(worst * 1e3).toFixed(4)} mm`;
  console.log(msg);
  toast(msg, worst > 1e-4);
  window.selftestWorst = worst;
}

async function loadModel() {
  const [j, bin] = await Promise.all([
    fetch('hand.json').then((r) => r.json()),
    fetch('hand.bin').then((r) => r.arrayBuffer()),
  ]);
  model = j;
  const solid = (g) => new THREE.MeshStandardMaterial({
    color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]).lerp(new THREE.Color(0x2a2f38), 0.25),
    roughness: 0.55, metalness: 0.15,
  });
  const ghostLook = () => new THREE.MeshStandardMaterial({
    color: 0x5ad1c8, transparent: true, opacity: 0.28, depthWrite: false, roughness: 0.4,
  });
  ghostLook.ghost = true;
  real = buildHand(bin, solid);
  ghost = buildHand(bin, ghostLook);
  ghost.root.visible = false;
  if (location.search.includes('selftest')) selftest();
  pose(real, config.rest_q, [0, 0, -1]);
}

function animate() {
  requestAnimationFrame(animate);
  if (real && S.q && S.state !== 'off') {
    pose(real, S.q, S.g);
    const showGhost = S.clip && S.clip_q;
    ghost.root.visible = !!showGhost;
    if (showGhost) pose(ghost, S.clip_q, S.pred || S.g);
  } else if (real && config) {
    pose(real, config.rest_q, [0, 0, -1]);
    ghost.root.visible = false;
  }
  controls.update();
  renderer.render(scene, camera);
}

// ------------------------------------------------------------------ map
const map = $('map'), ctx = map.getContext('2d');
let trail = [], simTrail = [], epoch = null;
function drawMap() {
  const dpr = devicePixelRatio || 1, w = map.clientWidth, h = map.clientHeight;
  if (map.width !== w * dpr) { map.width = w * dpr; map.height = h * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pts = trail.concat(simTrail);
  const est = S.est || [0, 0, 0];
  let xs = pts.map((p) => p[0]).concat([est[0], 0]), ys = pts.map((p) => p[1]).concat([est[1], 0]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const span = Math.max(160, Math.max(...xs) - Math.min(...xs) + 80, Math.max(...ys) - Math.min(...ys) + 80);
  const k = Math.min(w, h) / span; // px per mm
  // forward (x) is up the screen, left (y) is left
  const P = (x, y) => [w / 2 - (y - cy) * k, h / 2 - (x - cx) * k];
  ctx.lineWidth = 1;
  const g0x = Math.floor((cx - span) / 10) * 10, g0y = Math.floor((cy - span) / 10) * 10;
  for (let i = 0; i <= span * 2 / 10; i++) {
    const strong = ((g0x + i * 10) % 50) === 0;
    ctx.strokeStyle = strong ? '#2a3241' : '#1b212c';
    let [a, b] = P(g0x + i * 10, cy - span), [c, d] = P(g0x + i * 10, cy + span);
    ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
    const s2 = ((g0y + i * 10) % 50) === 0;
    ctx.strokeStyle = s2 ? '#2a3241' : '#1b212c';
    [a, b] = P(cx - span, g0y + i * 10); [c, d] = P(cx + span, g0y + i * 10);
    ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
  }
  const line = (arr, color, dash) => {
    if (arr.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash || []);
    ctx.beginPath();
    arr.forEach((p, i) => { const [a, b] = P(p[0], p[1]); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
    ctx.stroke(); ctx.setLineDash([]);
  };
  line(simTrail, '#7d8798', [4, 4]);
  line(trail, '#4cc2ff');
  const [ox, oy] = P(0, 0);
  ctx.fillStyle = '#56607a'; ctx.beginPath(); ctx.arc(ox, oy, 3, 0, 7); ctx.fill();
  // the hand: a triangle pointing where the fingertips point
  const [hx, hy] = P(est[0], est[1]);
  const yaw = (est[2] * Math.PI) / 180;
  ctx.save(); ctx.translate(hx, hy); ctx.rotate(-yaw);
  ctx.fillStyle = '#4cc2ff'; ctx.beginPath();
  ctx.moveTo(0, -13); ctx.lineTo(9, 9); ctx.lineTo(0, 4); ctx.lineTo(-9, 9); ctx.closePath(); ctx.fill();
  ctx.restore();
  if (S.yaw_gyro !== undefined && S.state === 'on') { // the heading the gyro measured
    const a = (S.yaw_gyro * Math.PI) / 180;
    ctx.strokeStyle = '#ffb020'; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(hx, hy); ctx.lineTo(hx - 24 * Math.sin(a), hy - 24 * Math.cos(a)); ctx.stroke();
  }
  ctx.fillStyle = '#8b94a5'; ctx.font = '11px system-ui';
  ctx.fillText(`${est[0].toFixed(0)} mm fwd · ${est[1].toFixed(0)} mm left · ${est[2].toFixed(0)}°`, 10, h - 10);
  if (S.yaw_gyro !== undefined) ctx.fillText(`gyro heading ${S.yaw_gyro.toFixed(0)}° (orange)`, 10, h - 26);
  if (S.sim) ctx.fillText(`the sim really: ${S.sim[0].toFixed(0)} mm fwd · ${S.sim[1].toFixed(0)} mm left (dashed)`, 10, h - 42);
  const bar = 10 * k; // 1 cm
  ctx.strokeStyle = '#8b94a5'; ctx.beginPath(); ctx.moveTo(w - 14 - bar, h - 14); ctx.lineTo(w - 14, h - 14); ctx.stroke();
  ctx.fillText('1 cm', w - 14 - bar, h - 20);
}

// ------------------------------------------------------------------ the side
function buildBars() {
  $('bars').innerHTML = FINGERS.map(([n, i0]) => `
    <div class="finger"><div class="bars">${[0, 1, 2, 3].map((k) => `<div class="bar"><div id="bar${i0 + k}"></div></div>`).join('')}</div>
    <div class="label">${n}</div></div>`).join('');
}
function covers(name) {
  const p = config.policies[S.policy];
  return !!(p && p.covers.includes(name));
}
function buildSide() {
  const sel = $('policy');
  sel.innerHTML = '<option value="">No policy · the scripts</option>' + Object.entries(config.policies)
    .map(([n, p]) => `<option value="${n}">${n} · ${p.covers.length} clips${p.imu ? '' : ' · no IMU'}</option>`).join('');
  sel.onchange = () => api('policy', { name: sel.value || null, live: false });
  document.querySelectorAll('#seg-live button').forEach((b) => (b.onclick = async () => {
    const live = b.dataset.live === '1';
    if (live && !S.policy) return toast('pick a policy first', true);
    if (live && config.mode === 'real' && !(await confirmBox('The policy will drive the motors. Someone at the power switch?'))) return;
    api('policy', { name: S.policy, live });
  }));
  $('emotes').innerHTML = config.emotes.map((e) => {
    const c = config.clips[e.name];
    return `<button class="emote" data-name="${e.name}" title="${c.note}"><span class="icon">${e.icon}</span>
      <span><div class="name">${c.title}</div><div class="meta">${c.seconds} s</div></span><i></i></button>`;
  }).join('');
  document.querySelectorAll('.emote').forEach((b) => (b.onclick = () => api('emote', { name: b.dataset.name })));
  document.querySelectorAll('.key').forEach((b) => {
    const clip = config.clips[config.moves[b.dataset.key]];
    if (!clip) { b.disabled = true; b.title = 'no clip'; return; }
    const t = clip.travel || [0, 0, 0];
    b.title = `${clip.title}: ${clip.seconds} s, planned ${t[0]} mm forward, ${t[1]} mm left, ${t[2]}°`;
    b.onclick = () => press(b.dataset.key);
  });
}
function press(key) {
  const b = document.querySelector(`.key[data-key="${key}"]`);
  if (!b || b.disabled) return;
  b.classList.add('pressed');
  setTimeout(() => b.classList.remove('pressed'), 150);
  api('move', { key });
}

function render() {
  const on = S.state === 'on', where = S.where;
  const moving = !!S.clip;
  $('dot').className = 'dot' + (moving ? ' moving' : on ? ' on' : '');
  const mode = $('chip-mode');
  mode.textContent = { real: 'real hand', fake: 'fake motors', twin: 'simulated hand' }[S.mode] || S.mode;
  mode.classList.toggle('real', S.mode === 'real');
  $('chip-where').textContent = S.state === 'off' ? 'disconnected' : S.state === 'on' ? where : S.state;
  const cc = $('chip-clip');
  cc.hidden = !moving;
  if (moving) cc.textContent = `${(config.clips[S.clip] || {}).title || S.clip}${S.policy_on ? (S.live ? ' · policy LIVE' : ' · policy shadow') : ''}`;
  $('btn-connect').disabled = S.state !== 'off';
  $('btn-stand').disabled = !on || where !== 'lying' || moving;
  $('btn-lie').disabled = !on || where === 'lying';
  $('btn-end').disabled = !on;
  const standing = on && (where === 'stand' || where === 'moving');
  document.querySelectorAll('.key').forEach((b) => {
    const name = config.moves[b.dataset.key];
    if (!name) return;
    b.disabled = !standing;
    b.querySelector('i').className = covers(name) ? 'policy' : '';
    b.classList.toggle('playing', S.clip === name);
  });
  document.querySelectorAll('.emote').forEach((b) => {
    b.disabled = !standing;
    b.querySelector('i').className = covers(b.dataset.name) ? 'policy' : '';
    b.classList.toggle('playing', S.clip === b.dataset.name);
  });
  const sel = $('policy');
  if (document.activeElement !== sel) sel.value = S.policy || '';
  document.querySelectorAll('#seg-live button').forEach((b) => b.classList.toggle('on', (b.dataset.live === '1') === !!S.live));
  const p = config.policies[S.policy];
  $('policy-hint').innerHTML = !p
    ? 'Every clip plays as written (open loop).'
    : `Drives (green dot): ${p.covers.map((n) => (config.clips[n] || {}).title || n).join(', ')}. The rest play as written.`
      + (S.live ? ' <b style="color:var(--warn)">Live: its targets go to the motors.</b>' : ' Shadow: computed and logged, the clip is sent.');
  $('queue').textContent = S.queue && S.queue.length ? S.queue.map((n) => (config.clips[n] || {}).title || n).join(' → ') : '–';
  const f = (v, unit, d = 1) => (v === null || v === undefined ? '–' : `${Number(v).toFixed(d)}${unit}`);
  $('hud-tilt').textContent = f(S.tilt, '°');
  $('hud-off').textContent = f(S.off, '°');
  $('hud-pol').textContent = f(S.pol, ' rad', 3);
  $('hud-late').textContent = f(S.late, ' ms');
  $('hud-temp').textContent = S.temps && S.temps[1] !== null ? `${S.temps[0]}–${S.temps[1]} °C` : '–';
  $('prog').style.width = S.clip && S.nframes ? `${(100 * Math.max(0, S.frame)) / S.nframes}%` : '0';
  if (S.cur) S.cur.forEach((c, i) => {
    const el = $('bar' + i), a = Math.abs(c);
    if (!el) return;
    el.style.height = `${Math.min(100, (100 * a) / 350)}%`;
    el.style.background = a > 300 ? 'var(--bad)' : a > 200 ? 'var(--warn)' : 'var(--ok)';
  });
  drawMap();
}
function addEvents(list) {
  const log = $('log');
  for (const [, t, text] of list) {
    const d = document.createElement('div');
    const cls = /ABORT|refused|ended/.test(text) ? 'bad' : /stopping|STOP|LIVE/.test(text) ? 'warn' : '';
    d.innerHTML = `<span class="t">${new Date(t * 1000).toLocaleTimeString()}</span><span class="${cls}"></span>`;
    d.lastChild.textContent = text;
    log.prepend(d);
  }
  while (log.childNodes.length > 120) log.lastChild.remove();
}

// ------------------------------------------------------------------ start
async function main() {
  config = await fetch('/api/config').then((r) => r.json());
  buildBars();
  buildSide();
  resize();
  animate();
  loadModel().catch((e) => toast('3D model: ' + e, true));
  $('btn-connect').onclick = async () => {
    if (config.mode === 'real' && !(await confirmBox('Torque on for the real hand. Someone at the power switch?'))) return;
    api('connect');
  };
  $('btn-stand').onclick = () => api('stand_up');
  $('btn-lie').onclick = () => api('lie_down');
  $('btn-end').onclick = () => api('end');
  $('btn-stop').onclick = () => api('stop');
  $('btn-clear').onclick = () => api('clear');
  $('btn-reset-map').onclick = () => api('reset_map');
  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.key === 'Escape') { api('stop'); return; }
    const k = e.key.toLowerCase();
    if (!e.repeat && config.moves[k]) press(k);
  });
  const first = await fetch('/api/state').then((r) => r.json());
  S = first;
  addEvents(first.events || []);
  const es = new EventSource('/api/stream');
  es.onmessage = (m) => {
    S = JSON.parse(m.data);
    if (S.events && S.events.length) addEvents(S.events);
    if (epoch !== S.map_epoch) { epoch = S.map_epoch; trail = []; simTrail = []; }
    if (S.est) {
      const last = trail[trail.length - 1];
      if (!last || Math.hypot(last[0] - S.est[0], last[1] - S.est[1]) > 0.5) trail.push([S.est[0], S.est[1]]);
    }
    if (S.sim) {
      const last = simTrail[simTrail.length - 1];
      if (!last || Math.hypot(last[0] - S.sim[0], last[1] - S.sim[1]) > 0.5) simTrail.push([S.sim[0], S.sim[1]]);
    }
    render();
  };
  es.onerror = () => { $('dot').className = 'dot'; };
  render();
}
main();
