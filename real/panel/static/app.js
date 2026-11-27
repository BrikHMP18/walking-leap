// Walking Leap control panel. Talks to server.py: POST /api/<command>, live state on /api/stream.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
let config = null;
let S = {}; // the latest state from the stream
const FINGERS = [['IF', 0], ['MF', 4], ['RF', 8], ['TH', 12]];

// ------------------------------------------------------------------ commands
async function api(path, body, quiet) {
  try {
    const r = await fetch('/api/' + path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}),
    });
    const j = await r.json();
    if (!j.ok && !quiet) toast(j.msg, true);
    return j;
  } catch (e) {
    if (!quiet) toast('the panel server does not answer', true);
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
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
// What each guard of wlc.Control.guard means, for the stop banner.
const ABORT_HINTS = [
  [/current limit/, 'A motor stayed near its 450 mA cap for 3 s: pushed too hard, or blocked.'],
  [/^ABORT: tilt \d+ deg/, 'The palm tilted past 70°.'],
  [/away from the sim/, 'The palm\'s "down" left the clip\'s by more than 25° for 0.3 s.'],
  [/motor error/, 'A servo reported a hardware error (overload?). If it repeats, power-cycle the hand.'],
  [/IMU is silent/, 'No IMU reading for 60 ms: check its cable.'],
  [/motors are silent/, 'No motor reading for 0.1 s: check the U2D2 and the power.'],
  [/too tilted to lie down/, 'Tilted more than 15°: it held, then let go instead of lying down.'],
];
let dismissedAbort = null;
function showAbort() {
  const a = S.last_abort;
  $('alert').hidden = !a || a.id === dismissedAbort;
  if (!a) return;
  const hint = (ABORT_HINTS.find(([re]) => re.test(a.text)) || [null, ''])[1];
  const what = a.text.replace(/ -> hold, then torque off$/, '');
  $('alert-text').innerHTML = `<b>Stopped</b> at ${new Date(a.t * 1000).toLocaleTimeString()}: ${esc(what)}.`
    + (hint ? ` ${esc(hint)}` : '') + ' Torque is off.';
}
function showCommit() {
  const c = $('chip-commit');
  if (!config.commit) return;
  const moved = config.disk_commit && config.disk_commit !== config.commit;
  c.hidden = false;
  c.textContent = moved ? `code ${config.commit} → ${config.disk_commit}: restart the panel` : config.commit;
  c.title = moved ? 'The repo changed since this server started (git pull?): restart real/panel/run.' : 'The commit this panel runs';
  c.classList.toggle('stale', !!moved);
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
// A velocity policy walks while a key or the joystick is held: the page repeats what is
// held (the server stands the hand still when it stops hearing it) and lets go on keyup,
// blur or a hidden tab. Keys stack: the last one still held wins (W, then Q, then let go
// of Q: forward again). A policy with a command range also takes the joystick (on screen
// or a gamepad's left stick); its keys are then points of the stick.
const NAMES = { forward: 'forward', back: 'back', left: 'left', right: 'right', turn_left: 'turn left',
  turn_right: 'turn right', forward_left: 'curve left', forward_right: 'curve right' };
const STICK_KEYS = { w: [0, 1], s: [0, -1], q: [-1, 1], e: [1, 1], a: [-1, 0], d: [1, 0] };
const STICK_NAMES = { w: 'forward', s: 'back', q: 'curve left', e: 'curve right', a: 'turn left', d: 'turn right' };
function velocity() {
  const p = config.policies[S.policy];
  return !!(p && p.kind === 'velocity');
}
function joystick() {
  const p = config.policies[S.policy];
  return !!(p && p.kind === 'velocity' && p.range);
}
function stickVec(x, y, r) {  // as the server's stick_command, in mm/s and deg/s
  return [y > 0 ? y * r[0][1] : -y * r[0][0], 0, x < 0 ? -x * r[2][1] : x * r[2][0]];
}
function vecText(v, short) {
  if (!v) return '';
  const f = Math.round(v[0] * 10) / 10, l = Math.round(v[1] * 10) / 10, t = Math.round(v[2] * 10) / 10;
  const out = [];
  if (f) out.push(short ? `${Math.abs(f)} mm/s` : `${f > 0 ? 'forward' : 'back'} ${Math.abs(f)} mm/s`);
  if (l) out.push(short ? `${Math.abs(l)} mm/s` : `${l > 0 ? 'left' : 'right'} ${Math.abs(l)} mm/s`);
  if (t) out.push(short ? `${Math.abs(t)}°/s` : `turning ${t > 0 ? 'left' : 'right'} ${Math.abs(t)}°/s`);
  return out.length ? out.join(' · ') : 'standing still';
}
let keysHeld = [], stick = null, walkTimer = null;
function walkHeld() { return stick !== null || keysHeld.length > 0; }
function walkBody() {
  if (stick) return { stick };
  const k = keysHeld[keysHeld.length - 1];
  if (!k) return { key: null };
  return joystick() ? { stick: STICK_KEYS[k] } : { key: k };
}
function walkSend(quiet, start) { return api('walk', { ...walkBody(), start: !!start }, quiet); }
function walkRefresh(start) {
  if (walkHeld()) {
    if (!walkTimer) walkTimer = setInterval(() => walkSend(true), 120);
  } else {
    clearInterval(walkTimer);
    walkTimer = null;
  }
  return walkSend(!start, start);
}
async function hold(key) {
  const p = config.policies[S.policy];
  if (!p || keysHeld.includes(key)) return;
  if (!joystick() && !(p.keys || {})[key]) return toast(`${S.policy} has nothing on ${key.toUpperCase()}`, true);
  keysHeld.push(key);
  const j = await walkRefresh(true);
  if (!j.ok) release(key);
}
function release(key) {
  if (key === undefined) {
    if (!walkHeld()) return;
    keysHeld = [];
    stick = null;
    showKnob(0, 0);
  } else {
    if (!keysHeld.includes(key)) return;
    keysHeld = keysHeld.filter((k) => k !== key);
  }
  walkRefresh(false);
}
function showKnob(x, y) {
  const k = document.querySelector('#stick .knob');
  if (k) k.style.transform = `translate(calc(-50% + ${x * 44}px), calc(-50% + ${-y * 44}px))`;
}
function buildStick() {
  const pad = $('stick');
  let dragging = false;
  const at = (e) => {
    const r = pad.getBoundingClientRect();
    let x = ((e.clientX - r.left) / r.width) * 2 - 1, y = 1 - ((e.clientY - r.top) / r.height) * 2;
    const n = Math.hypot(x, y);
    if (n > 1) { x /= n; y /= n; }
    return [Math.round(x * 100) / 100, Math.round(y * 100) / 100];
  };
  pad.onpointerdown = async (e) => {
    if (!joystick() || pad.classList.contains('off')) return;
    dragging = true;
    try { pad.setPointerCapture(e.pointerId); } catch (err) { /* a synthetic pointer */ }
    stick = at(e);
    showKnob(...stick);
    const j = await walkRefresh(true);
    if (!j.ok) { dragging = false; release(); }
  };
  pad.onpointermove = (e) => { if (dragging) { stick = at(e); showKnob(...stick); } };
  pad.onpointerup = pad.onpointercancel = () => {
    if (!dragging) return;
    dragging = false;
    stick = null;
    showKnob(0, 0);
    walkRefresh(false);
  };
}
let padOn = false;
function pollGamepad() {  // a gamepad's left stick: the joystick, with a dead zone
  const g = (navigator.getGamepads ? [...navigator.getGamepads()] : []).find((x) => x && x.connected);
  $('gamepad').hidden = !g;
  const off = () => { if (padOn) { padOn = false; stick = null; showKnob(0, 0); walkRefresh(false); } };
  if (!g || !joystick() || $('stick').classList.contains('off')) return off();
  let x = g.axes[0] || 0, y = -(g.axes[1] || 0);
  const n = Math.hypot(x, y), dz = 0.15;
  if (n <= dz) return off();
  const s = Math.min(1, (n - dz) / (1 - dz)) / n;
  stick = [Math.round(x * s * 100) / 100, Math.round(y * s * 100) / 100];
  showKnob(...stick);
  if (!padOn) { padOn = true; walkRefresh(true); }
}
function buildSide() {
  const sel = $('policy');
  const all = Object.entries(config.policies);
  const opt = ([n, p]) => `<option value="${esc(n)}" title="${esc(p.source || 'source not recorded')}">${esc(n)} · ${p.kind === 'velocity' ? (p.range ? 'joystick' : 'hold the keys') : p.covers.length + ' clips'}${p.imu ? '' : ' · no IMU'}</option>`;
  const cycles = all.filter(([, p]) => p.kind !== 'velocity'), walking = all.filter(([, p]) => p.kind === 'velocity');
  sel.innerHTML = '<option value="">No policy · the scripts</option>'
    + (cycles.length ? `<optgroup label="Cycles · one clip per key or emote">${cycles.map(opt).join('')}</optgroup>` : '')
    + (walking.length ? `<optgroup label="Walking · keys or joystick">${walking.map(opt).join('')}</optgroup>` : '');
  sel.onchange = async () => {
    await api('policy', { name: sel.value || null, live: false });
    sel.blur();  // the next render() shows the server's policy (this one, or the old one if refused)
  };
  document.querySelectorAll('#seg-live button').forEach((b) => (b.onclick = async () => {
    const live = b.dataset.live === '1';
    if (live && !S.policy) return toast('pick a policy first', true);
    if (live && config.mode === 'real' && !(await confirmBox('The policy will drive the motors. Someone at the power switch?'))) return;
    api('policy', { name: S.policy, live });
  }));
  $('emotes').innerHTML = config.emotes.map((e) => {
    const c = config.clips[e.name];
    return `<button class="emote" data-name="${e.name}" title="${c.note}"><span class="icon">${e.icon}</span>
      <span><div class="name">${c.title}</div><div class="meta">${c.seconds} s${c.start === 'lying' ? ' · lying' : ''}</div></span><i></i></button>`;
  }).join('');
  document.querySelectorAll('.emote').forEach((b) => (b.onclick = () => api('emote', { name: b.dataset.name })));
  document.querySelectorAll('.key').forEach((b) => {
    b.dataset.label = b.querySelector('small').textContent;
    const clip = config.clips[config.moves[b.dataset.key]];
    if (!clip) { b.disabled = true; b.title = 'no clip'; return; }
    const t = clip.travel || [0, 0, 0];
    b.dataset.clipTitle = `${clip.title}: ${clip.seconds} s, planned ${t[0]} mm forward, ${t[1]} mm left, ${t[2]}°`;
    b.onclick = () => { if (!velocity()) press(b.dataset.key); };
    b.onpointerdown = (e) => { if (velocity()) { e.preventDefault(); hold(b.dataset.key); } };
    b.onpointerup = b.onpointerleave = b.onpointercancel = () => release(b.dataset.key);
  });
  $('btn-walk-end').onclick = () => { release(); api('walk', { end: true }); };
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
  cc.classList.toggle('walk', !!S.walking);
  if (S.walking) {
    cc.hidden = false;
    cc.textContent = `Walking · ${vecText(S.walk_vec)} · ${S.live ? 'LIVE' : 'shadow'}`;
  }
  $('btn-connect').disabled = S.state !== 'off';
  $('btn-stand').disabled = !on || where !== 'lying' || moving;
  $('btn-lie').disabled = !on || where === 'lying';
  $('btn-end').disabled = !on;
  const standing = on && (where === 'stand' || where === 'moving');
  const walks = velocity(), vp = config.policies[S.policy];
  const kind = !vp ? 'scripts' : walks ? 'walk' : 'cycles';
  $('policy-badge').textContent = { scripts: 'Scripts', cycles: 'Cycles', walk: 'Walking' }[kind];
  $('policy-badge').className = 'badge ' + kind;
  const stickOn = walks && joystick();
  $('move-badge').textContent = stickOn ? 'joystick or keys' : walks ? 'hold to walk' : '1 cycle per press';
  $('move-badge').className = 'badge ' + (walks ? 'walk' : vp ? 'cycles' : '');
  $('move-card').classList.toggle('walkmode', walks);
  const top = keysHeld[keysHeld.length - 1];
  document.querySelectorAll('.key').forEach((b) => {
    const key = b.dataset.key, name = config.moves[key];
    if (walks) {
      const cmd = stickOn ? null : (vp.keys || {})[key];
      const v = stickOn ? stickVec(...STICK_KEYS[key], vp.range) : vp.commands[cmd];
      const label = stickOn ? STICK_NAMES[key] : NAMES[cmd] || '–';
      b.querySelector('small').innerHTML = v ? `${label}<br>${vecText(v, true)}` : 'not in it';
      b.disabled = !standing || !v || (moving && !S.walking);
      b.title = v ? `hold: ${vecText(v)}${stickOn ? ' (a point of the joystick)' : ' (as trained)'}` : 'not in this policy';
      b.querySelector('i').className = v ? 'policy' : '';
      b.classList.toggle('playing', !!S.walking && top === key && !stick);
      return;
    }
    b.querySelector('small').textContent = b.dataset.label;
    if (!name) { b.disabled = true; return; }
    b.disabled = !standing;
    b.title = b.dataset.clipTitle || '';
    b.querySelector('i').className = covers(name) ? 'policy' : '';
    b.classList.toggle('playing', S.clip === name);
  });
  $('move-sub').textContent = walks ? 'release: stand still' : 'W A S D · Q E';
  $('btn-walk-end').hidden = !S.walking;
  $('stick-wrap').hidden = !stickOn;
  $('stick').classList.toggle('off', !standing || (moving && !S.walking));
  $('stick').classList.toggle('active', !!stick);
  $('stick-read').textContent = S.walking ? vecText(S.walk_vec) : stickOn ? 'drag to walk, let go to stand' : '';
  if (keysHeld.some((k) => { const hb = document.querySelector(`.key[data-key="${k}"]`); return hb && hb.disabled; })) {
    release();  // a held key's button got disabled under the finger: no pointerup will come
  }
  document.querySelectorAll('.emote').forEach((b) => {
    const lyingOne = (config.clips[b.dataset.name] || {}).start === 'lying';
    b.disabled = lyingOne ? !on || where !== 'lying' || moving : !standing;
    b.querySelector('i').className = covers(b.dataset.name) ? 'policy' : '';
    b.classList.toggle('playing', S.clip === b.dataset.name);
  });
  const sel = $('policy');
  if (document.activeElement !== sel) sel.value = S.policy || '';
  document.querySelectorAll('#seg-live button').forEach((b) => b.classList.toggle('on', (b.dataset.live === '1') === !!S.live));
  const p = config.policies[S.policy];
  $('policy-hint').innerHTML = !p
    ? 'Every clip plays as written (open loop).'
    : p.kind === 'velocity'
      ? (p.range
        ? `Walks with the joystick (drag it, or a gamepad's left stick): up and down ${vecText([p.range[0][1], 0, 0], true)} forward to ${vecText([p.range[0][0], 0, 0], true)} back, left and right up to ${vecText([0, 0, p.range[2][1]], true)} of turn. The keys are points of it. Let go: it stands still. Emotes play as written (the walk ends first).`
        : 'Walks while a key is held, at the speeds it trained on (Q and E curve if it learned to); released, it stands still. Emotes play as written (the walk ends first).')
        + (S.live ? ' <b style="color:var(--warn)">Live: it walks.</b>' : ' Shadow: the hand holds its stance, the policy is computed and logged.')
      : `Drives (green dot): ${p.covers.map((n) => (config.clips[n] || {}).title || n).join(', ')}. The rest play as written.`
        + (S.live ? ' <b style="color:var(--warn)">Live: its targets go to the motors.</b>' : ' Shadow: computed and logged, the clip is sent.');
  if (p) $('policy-hint').innerHTML += `<div class="src">From ${esc(p.source || 'an unrecorded run (exported before files kept their source)')}</div>`;
  showAbort();
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
    el.style.height = `${Math.min(100, (100 * a) / 450)}%`;
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
  showCommit();
  $('alert-x').onclick = () => { dismissedAbort = S.last_abort && S.last_abort.id; showAbort(); };
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
    if (e.key === 'Escape') { release(); api('stop'); return; }
    const k = e.key.toLowerCase();
    if (velocity()) {
      if (config.walk.includes(k)) { e.preventDefault(); if (!e.repeat) hold(k); }
      else if (e.key === ' ') { e.preventDefault(); release(); }
      return;
    }
    if (!e.repeat && config.moves[k]) press(k);
  });
  addEventListener('keyup', (e) => { if (keysHeld.length) release(e.key.toLowerCase()); });
  buildStick();
  setInterval(pollGamepad, 50);
  addEventListener('blur', () => release());
  document.addEventListener('visibilitychange', () => { if (document.hidden) release(); });
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
