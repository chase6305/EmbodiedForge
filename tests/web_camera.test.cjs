/* Run with node --test tests/web_camera.test.cjs; no browser/npm dependencies. */
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');

function app() {
  const elements = new Map(), requests = [], timers = new Map();
  let clock = 1000, timerId = 0;
  class Element {
    constructor() {
      this.style = {}; this.dataset = {}; this.options = []; this.listeners = {};
      this.classList = {toggle() {}, add() {}, remove() {}};
      this.captures = new Set(); this.clientHeight = 600;
    }
    addEventListener(name, fn) { this.listeners[name] = fn; }
    setAttribute() {} removeAttribute() {} focus() {}
    add(option) { this.options.push(option); }
    setPointerCapture(id) { this.captures.add(id); }
    hasPointerCapture(id) { return this.captures.has(id); }
    releasePointerCapture(id) {
      assert(this.captures.delete(id), 'only release an owned capture');
      this.onlostpointercapture?.({pointerId: id});
    }
  }
  const element = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const document = new Element();
  document.getElementById = element; document.querySelectorAll = () => [];
  const window = new Element();
  const context = vm.createContext({
    document, window, token: 'test-token', console, AbortController,
    Date: {now: () => clock},
    Option: function(text, value) { this.text = text; this.value = value; },
    setTimeout: (fn, delay) => { timers.set(++timerId, {fn, at: clock + delay}); return timerId; },
    clearTimeout: id => timers.delete(id),
    fetch: (url, options) => new Promise(resolve => {
      // Keep startup polling pending; tests publish explicit rendered states.
      if (url === '/api/control') requests.push({body: JSON.parse(options.body), resolve});
    }),
  });
  const run = code => vm.runInContext(code, context);
  run(readFileSync(join(__dirname, '../src/embodiedforge/viewers/web.js'), 'utf8'));
  const initial = {
    ready: true, camera_control: true, camera: {azimuth: -90, elevation: 45, distance: 3.5},
    default_camera: {azimuth: -90, elevation: 45, distance: 3.5, target: [0, 0, 0]},
    control_id: 0, renderer: 'rtx', renderers: {}, num_envs: 1, env_id: 0,
    render_width: 960, render_height: 540, width: 960, height: 540, target_fps: 30,
    jpeg_bytes: 1000, frame_ms: 2, speed: 1, paused: true, time: 0, step: 0,
    episode: 0, velocity: 0, position: [0, 0, 0], fps: 30, frame_id: 1, error: null,
  };
  function state(patch = {}) {
    run(`connected = true; renderState(${JSON.stringify({...initial, ...patch})});`);
  }
  state();
  const viewport = element('viewport');
  function pointer(type, patch = {}) {
    viewport['onpointer' + type]({pointerId: 1, isPrimary: true, button: 0,
      buttons: type === 'up' ? 0 : 1, clientX: 100, clientY: 100,
      preventDefault() {}, ...patch});
  }
  const snapshot = () => JSON.parse(run('JSON.stringify(camera)'));
  const tick = () => new Promise(resolve => setImmediate(resolve));
  async function accept(index, controlId) {
    requests[index].resolve({ok: true, json: async () => ({accepted: true, control_id: controlId})});
    await tick();
  }
  function advance(ms) {
    clock += ms;
    for (const [id, timer] of [...timers]) {
      if (timer.at <= clock) { timers.delete(id); timer.fn(); }
    }
  }
  return {element, viewport, document, window, requests, state, run, pointer, snapshot, tick, accept, advance};
}

test('only the active pointer moves the camera; release stops rotation outside the viewport', async () => {
  const a = app();
  a.pointer('down');
  a.pointer('move', {clientX: 140});
  const pose = a.snapshot();
  assert.equal(pose.azimuth, -104);
  a.pointer('down', {pointerId: 2, clientX: 800});
  a.pointer('move', {pointerId: 2, clientX: 850});
  a.pointer('up', {pointerId: 2});
  assert.deepEqual(a.snapshot(), pose);
  assert(a.viewport.hasPointerCapture(1));
  a.pointer('up', {clientX: -200});
  a.pointer('move', {clientX: 700, buttons: 0});
  assert.deepEqual(a.snapshot(), pose);
  assert.equal(a.viewport.captures.size, 0);
  await a.tick();
  assert.equal(a.requests.at(-1).body.azimuth, pose.azimuth);
});

test('missing mouse-up, capture loss, pointer cancel, blur and hidden tab end a gesture', () => {
  for (const end of ['buttons', 'capture', 'cancel', 'blur', 'hidden']) {
    const a = app();
    a.pointer('down'); a.pointer('move', {clientX: 120});
    const pose = a.snapshot();
    if (end === 'buttons') a.pointer('move', {clientX: 900, buttons: 0});
    if (end === 'capture') a.viewport.releasePointerCapture(1);
    if (end === 'cancel') a.pointer('cancel');
    if (end === 'blur') a.window.listeners.blur();
    if (end === 'hidden') { a.document.hidden = true; a.document.listeners.visibilitychange(); }
    a.pointer('move', {clientX: 800});
    assert.deepEqual(a.snapshot(), pose, end);
    assert.equal(a.viewport.captures.size, 0, end);
  }
});

test('slow receipts and old state polls cannot rewind the next drag', async () => {
  const a = app();
  a.pointer('down'); a.pointer('move', {clientX: 140}); a.pointer('up');
  await a.tick();
  const pose = a.snapshot();
  a.advance(2000); a.state();
  assert.deepEqual(a.snapshot(), pose, 'protect pose even before HTTP acceptance');
  await a.accept(0, 42);
  a.state({control_id: 41});
  assert.deepEqual(a.snapshot(), pose, 'HTTP acceptance does not mean a rendered frame');
  a.pointer('down'); a.pointer('move', {clientX: 120}); a.pointer('up');
  await a.tick();
  assert.equal(a.snapshot().azimuth, -111);
  a.state({control_id: 42, camera: pose});
  assert.equal(a.snapshot().azimuth, -111, 'earlier receipt cannot replace newer input');
  await a.accept(1, 43);
  a.state({control_id: 43, camera: a.snapshot()});
  a.state({control_id: 44, camera: {...pose, azimuth: 10}});
  assert.equal(a.snapshot().azimuth, 10, 'resume synchronization after the final receipt');
});

test('slow network coalesces camera samples but preserves reset ordering', async () => {
  const a = app();
  a.run('sendCamera()'); await a.tick();
  for (let i = 0; i < 100; i++) a.run(`camera.azimuth = ${i}; sendCamera()`);
  a.element('camera-reset').onclick();
  a.run('camera.azimuth = 12; sendCamera(); camera.azimuth = 15; sendCamera()');
  assert.equal(a.requests.length, 1);
  await a.accept(0, 1);
  assert.equal(a.requests[1].body.azimuth, 99);
  await a.accept(1, 2);
  assert.equal(a.requests[2].body.action, 'camera_reset');
  await a.accept(2, 3);
  assert.equal(a.requests[3].body.azimuth, 15);
  await a.accept(3, 4);
  assert.equal(a.requests.length, 4);
});

test('pending wheel input survives polls and reset cancels its delayed command', async () => {
  const a = app();
  a.viewport.listeners.wheel({deltaY: -3, deltaMode: 1, preventDefault() {}});
  await a.tick(); await a.accept(0, 1);
  const first = a.snapshot();
  a.advance(10);
  a.viewport.listeners.wheel({deltaY: -3, deltaMode: 1, preventDefault() {}});
  const zoom = a.snapshot().distance;
  assert(zoom < 3.5);
  a.state({control_id: 1, camera: first}); assert.equal(a.snapshot().distance, zoom);
  a.element('camera-reset').onclick();
  a.advance(200); await a.tick();
  assert.equal(a.requests.length, 2);
  assert.deepEqual(a.requests[1].body, {action: 'camera_reset'});
  await a.accept(1, 2); a.advance(200); await a.tick();
  assert.equal(a.requests.length, 2, 'reset must cancel the trailing wheel pose');
  assert.equal(a.snapshot().distance, 3.5);
});

test('disabled camera and secondary buttons cannot start a drag; large deltas stay bounded', () => {
  const a = app();
  a.pointer('down', {button: 2}); a.pointer('move', {clientX: 500});
  assert.equal(a.snapshot().azimuth, -90);
  a.state({camera_control: false});
  a.pointer('down'); a.pointer('move', {clientX: 500});
  assert.equal(a.snapshot().azimuth, -90);
  a.state(); a.pointer('down');
  a.pointer('move', {clientX: 15000, clientY: -15000});
  assert(a.snapshot().azimuth >= -180 && a.snapshot().azimuth < 180);
  assert.equal(a.snapshot().elevation, 5);
  a.state({camera_control: false});
  assert.equal(a.viewport.captures.size, 0);
});

test('continuous wheel input updates the renderer before scrolling stops', async () => {
  const a = app();
  for (let i = 0; i < 12; i++) {
    a.viewport.listeners.wheel({deltaY: -2, deltaMode: 0, preventDefault() {}});
    a.advance(16);
    await a.tick();
  }
  assert(a.requests.length > 0, 'continuous scrolling must not starve rendering');
  const latest = a.snapshot();
  a.advance(40); await a.tick();
  await a.accept(0, 1);
  assert.equal(a.requests.at(-1).body.distance, latest.distance);
});

test('the last small drag movement is rendered while the button is still held', async () => {
  const a = app();
  a.pointer('down');
  a.pointer('move', {clientX: 120}); await a.tick();
  await a.accept(0, 1);
  a.advance(10);
  a.pointer('move', {clientX: 130});
  const latest = a.snapshot();
  a.advance(40); await a.tick();
  assert.equal(a.requests.length, 2, 'flush the trailing pose without requiring mouse-up');
  assert.equal(a.requests[1].body.azimuth, latest.azimuth);
  assert(a.viewport.hasPointerCapture(1));
});

test('switching to a renderer without orbit cancels a trailing camera update', async () => {
  const a = app();
  a.pointer('down'); a.pointer('move', {clientX: 120});
  await a.tick(); await a.accept(0, 1);
  a.advance(10); a.pointer('move', {clientX: 130});
  a.state({camera_control: false});
  a.advance(100); await a.tick();
  assert.equal(a.requests.length, 1);
  assert.equal(a.viewport.captures.size, 0);
});

test('interleaved orbit and wheel input flush one final pose with both changes', async () => {
  const a = app();
  a.pointer('down'); a.pointer('move', {clientX: 120});
  await a.tick(); await a.accept(0, 1);
  a.advance(10);
  a.viewport.listeners.wheel({deltaY: -120, deltaMode: 0, preventDefault() {}});
  a.pointer('move', {clientX: 140});
  const latest = a.snapshot();
  a.advance(40); await a.tick();
  assert.equal(a.requests.length, 2);
  assert.equal(a.requests[1].body.azimuth, -104);
  assert.equal(a.requests[1].body.distance, latest.distance);
  assert(latest.distance < 3.5);
});
