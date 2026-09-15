/* Shared point simulation, robot replay and live-policy browser controls. */
const $ = id => document.getElementById(id);
const names = {raster: 'Raster · CPU', mujoco: 'MuJoCo', gl: 'OpenGL · Newton', rtx: 'RTX · OVRTX'};
const velocityIds = ['vx', 'vy', 'wz'];
const presets = {
  'cmd-forward': [.5, 0, 0], 'cmd-back': [-.5, 0, 0],
  'cmd-left': [0, .3, 0], 'cmd-right': [0, -.3, 0],
  'cmd-turn-left': [.3, 0, .5], 'cmd-turn-right': [.3, 0, -.5], 'cmd-zero': [0, 0, 0],
};
let state = null, connected = false, stopped = false, stopping = false;
let connectionEpoch = 0;
let camera = {azimuth: -90, elevation: 45, distance: 3.5};
let drag = null, lastCameraSend = -Infinity, cameraTimer = null, pendingCamera = null, scrubbing = false;
let queuedCamera = null;
let pendingRenderer = null, pendingSelection = null, pendingVelocity = null, pendingReset = null, pendingPause = null;
let rendererRequest = null, pendingSpeed = null, pendingFollow = null;
let pendingResolution = null, pendingFPS = null;
let chain = Promise.resolve(), failures = 0, lastBackendError = null;
const drafts = new Map();
const sameCommand = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 1e-6);
const observed = (pending, s) => pending?.accepted && s.control_id >= pending.control_id;

function error(message) {
  $('error').textContent = message || '';
  $('error').style.display = message ? 'block' : 'none';
}

async function fetchJSON(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(url, {...options, signal: controller.signal});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || '操作失败');
    return result;
  } finally { clearTimeout(timer); }
}

function send(command) {
  // Capture command arguments at the interaction, never from a later poll.
  const body = JSON.stringify(command);
  // Keep the newest unsent pose without moving it across other controls.
  if (command.action === 'camera' && queuedCamera) {
    queuedCamera.body = body;
    return queuedCamera.promise;
  }
  const request = {body, epoch: connectionEpoch};
  queuedCamera = command.action === 'camera' ? request : null;
  chain = chain.catch(() => {}).then(() => {
    if (queuedCamera === request) queuedCamera = null;
    // A recovered connection must not replay actions from before the outage.
    // Requests already sent may still complete; only unsent work is canceled.
    if (request.epoch !== connectionEpoch) throw Error('连接已中断，未发送的操作已取消');
    if (stopped || !connected) throw Error('连接不可用，请等待状态恢复后重新操作');
    return fetchJSON('/api/control', {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-Session-Token': token}, body: request.body,
    });
  });
  request.promise = chain;
  chain.catch(e => {
    // Late failures from the previous connection must not replace new feedback.
    if (request.epoch === connectionEpoch) error(e.name === 'AbortError' ? '请求超时，操作状态未知；请检查当前状态后重试' : e.message);
  });
  return chain;
}

function controls() {
  const disabled = !connected || !state?.ready || stopped || stopping;
  document.querySelectorAll('button,select,input').forEach(el => el.disabled = disabled);
  $('fullscreen').disabled = !document.fullscreenEnabled;
  $('workspace').classList.toggle('disconnected', !connected && !stopped);
  if (disabled) { $('snapshot').removeAttribute('href'); return; }
  $('snapshot').href = '/snapshot.jpg';
  const localBusy = !!(pendingSelection || pendingReset);
  const ended = !!state.live?.blocked_envs.length || !!state.replay?.all_ended;
  $('environment').disabled = localBusy || !!pendingVelocity;
  $('renderer').disabled = !!pendingRenderer || !!pendingResolution;
  $('resolution').disabled = !!pendingRenderer || !!pendingResolution;
  $('target-fps').disabled = !!pendingFPS || !!pendingRenderer || !!pendingResolution;
  $('speed').disabled = !!pendingSpeed;
  $('follow').disabled = !!pendingFollow;
  $('run').disabled = !state.paused || ended || !!pendingPause;
  $('pause').disabled = state.paused || !!pendingPause;
  $('step').disabled = ended || !!pendingPause || localBusy;
  $('reset').disabled = localBusy || !!pendingVelocity;
  $('timeline').disabled = localBusy;
  for (const id of ['camera-reset', 'camera-top', 'zoom-in', 'zoom-out']) $(id).disabled = !state.camera_control;
  for (const id of [...Object.keys(presets), 'cmd-apply']) $(id).disabled = localBusy || !!pendingVelocity || !state.live;
  for (const id of [...velocityIds, 'cmd-revert']) $(id).disabled = localBusy || !state.live;
  $('cmd-revert').disabled ||= !!pendingVelocity || !drafts.has(state.env_id);
  $('reset').textContent = state.replay ? `↻ 记录 ${state.env_id} 回到开头` : `↻ 重置环境 ${state.env_id}`;
  $('cmd-apply').textContent = `应用到环境 ${state.env_id}`;
}

function feedback() {
  $('interaction-status').textContent = pendingSelection ? `正在切换到环境 ${pendingSelection.env_id}…` :
    pendingReset ? `正在重置环境 ${pendingReset.env_id}…` :
    pendingPause ? '等待播放状态更新…' : pendingRenderer ? '正在准备渲染器，仿真暂时等待…' :
    pendingResolution ? '正在调整分辨率，保留当前画面…' : pendingFPS ? '正在调整目标帧率…' : '';
  if (state?.live) {
    const dirty = drafts.has(state.env_id);
    $('velocity-status').dataset.dirty = String(dirty);
    $('velocity-status').textContent = pendingVelocity ? `环境 ${pendingVelocity.env_id}：等待指令生效…` :
      dirty ? `环境 ${state.env_id}：输入尚未应用` : `环境 ${state.env_id}：显示当前已生效指令`;
  }
  controls();
}

function tracked(command, pending, clear) {
  error('');
  feedback();
  send(command).then(result => {
    pending.control_id = result.control_id; pending.accepted = true;
  }).catch(() => { clear(); feedback(); });
}

function pause(paused) {
  if (pendingPause) return;
  const pending = pendingPause = {paused, accepted: false};
  tracked({action: 'pause', paused}, pending, () => { pendingPause = null; });
}
$('run').onclick = () => pause(false);
$('pause').onclick = () => pause(true);
$('step').onclick = () => send({action: 'step'});
$('reset').onclick = () => {
  const pending = pendingReset = {env_id: state.env_id, episode: state.episode, accepted: false};
  tracked({action: 'reset', env_id: pending.env_id}, pending, () => { pendingReset = null; });
};
$('environment').onchange = () => {
  const pending = pendingSelection = {env_id: Number($('environment').value), accepted: false};
  tracked({action: 'select', env_id: pending.env_id}, pending, () => {
    pendingSelection = null; $('environment').value = String(state.env_id);
  });
};

function writeVelocity(value) { velocityIds.forEach((id, i) => $(id).value = value[i]); }
function velocityCommand(value) {
  if (!state?.live || pendingSelection || pendingVelocity || pendingReset || !connected || stopped) return;
  const env_id = state.env_id;
  if (value.some((v, i) => !Number.isFinite(v) || Math.abs(v) > state.live.command_limits[i])) {
    error('速度指令必须是范围内的有限数值'); return;
  }
  writeVelocity(value);
  drafts.set(env_id, value.map(String));
  const pending = pendingVelocity = {env_id, value, draft: value.map(String), accepted: false};
  tracked({action: 'velocity', env_id, value}, pending, () => { pendingVelocity = null; });
}
for (const id of velocityIds) $(id).oninput = () => {
  drafts.set(state.env_id, velocityIds.map(id => $(id).value));
  feedback();
};
for (const [id, value] of Object.entries(presets)) $(id).onclick = () => velocityCommand(value);
$('velocity-form').onsubmit = e => {
  e.preventDefault();
  if ($('cmd-apply').disabled || !$('velocity-form').reportValidity()) return;
  velocityCommand(velocityIds.map(id => $(id).valueAsNumber));
};
$('cmd-revert').onclick = () => { drafts.delete(state.env_id); writeVelocity(state.live.command); feedback(); };
$('speed').onchange = () => {
  const pending = pendingSpeed = {value: Number($('speed').value), accepted: false};
  tracked({action: 'speed', value: pending.value}, pending, () => { pendingSpeed = null; });
};
$('renderer').onchange = () => {
  pendingRenderer = $('renderer').value;
  const pending = rendererRequest = {accepted: false};
  tracked({action: 'renderer', backend: pendingRenderer}, pending, () => { pendingRenderer = null; rendererRequest = null; });
};
$('resolution').onchange = () => {
  const [width, height] = $('resolution').value.split('x').map(Number);
  const pending = pendingResolution = {width, height, accepted: false};
  tracked({action: 'resolution', width, height}, pending, () => { pendingResolution = null; });
};
$('target-fps').onchange = () => {
  const pending = pendingFPS = {value: Number($('target-fps').value), accepted: false};
  tracked({action: 'fps', value: pending.value}, pending, () => { pendingFPS = null; });
};
$('timeline').onpointerdown = () => { scrubbing = true; };
$('timeline').oninput = () => { scrubbing = true; };
$('timeline').onchange = () => {
  scrubbing = false;
  if (!$('timeline').disabled) send({action: 'seek', env_id: state.env_id, frame: Number($('timeline').value)});
};
$('timeline').onpointercancel = () => { scrubbing = false; };
$('follow').onchange = () => {
  const pending = pendingFollow = {enabled: $('follow').checked, accepted: false};
  tracked({action: 'follow', enabled: pending.enabled}, pending, () => { pendingFollow = null; });
};

function cameraEnabled() {
  return connected && !stopped && !stopping && state?.ready && state.camera_control;
}
function clearCameraTimer() {
  clearTimeout(cameraTimer); cameraTimer = null;
}
function scheduleCamera() {
  // Send during continuous input and flush its trailing pose, at most 30 Hz.
  const delay = Math.max(0, 1000 / 30 - (Date.now() - lastCameraSend));
  if (!delay) sendCamera();
  else if (cameraTimer === null) cameraTimer = setTimeout(() => sendCamera(), delay);
}
function sendCamera(action = 'camera') {
  clearCameraTimer();
  if (!cameraEnabled()) return;
  lastCameraSend = Date.now();
  const pending = pendingCamera = {accepted: false};
  return send(action === 'camera_reset' ? {action} : {action, ...camera}).then(result => {
    pending.control_id = result.control_id; pending.accepted = true;
  }).catch(() => { if (pendingCamera === pending) pendingCamera = null; });
}
$('camera-reset').onclick = () => {
  finishDrag(false);
  camera = {...state.default_camera};
  delete camera.target;
  sendCamera('camera_reset');
};
$('camera-top').onclick = () => { camera.elevation = 89; sendCamera(); };
$('zoom-in').onclick = () => { camera.distance = Math.max(.3, camera.distance * .8); sendCamera(); };
$('zoom-out').onclick = () => { camera.distance = Math.min(20, camera.distance * 1.25); sendCamera(); };
$('fullscreen').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $('workspace').requestFullscreen();
  } catch (e) { error('无法进入全屏：' + e.message); }
};
document.addEventListener('fullscreenchange', () => {
  $('fullscreen').textContent = document.fullscreenElement ? '退出全屏' : '全屏';
});
$('stop').onclick = () => {
  stopping = true; feedback();
  send({action: 'stop'}).then(() => {
    stopped = true;
    $('connection').textContent = '会话已停止';
    $('dot').classList.remove('ready');
    $('status').textContent = '已停止';
    $('interaction-status').textContent = '已请求停止会话';
    clearCameraTimer(); finishDrag(false); controls();
  }).catch(() => { stopping = false; feedback(); });
};

const viewport = $('viewport');
function finishDrag(flush = true) {
  if (!drag) return;
  const {pointerId} = drag;
  drag = null;
  if (viewport.hasPointerCapture(pointerId)) viewport.releasePointerCapture(pointerId);
  if (flush) sendCamera();
}
viewport.onpointerdown = e => {
  if (!cameraEnabled() || drag || e.isPrimary === false || e.button !== 0) return;
  e.preventDefault();
  viewport.focus({preventScroll: true});
  viewport.setPointerCapture(e.pointerId);
  drag = {pointerId: e.pointerId, x: e.clientX, y: e.clientY};
};
viewport.onpointermove = e => {
  if (!drag || e.pointerId !== drag.pointerId) return;
  if (!cameraEnabled()) { finishDrag(false); return; }
  // A missed mouse-up must not turn hover movement into another rotation.
  if (!(e.buttons & 1)) { finishDrag(); return; }
  e.preventDefault();
  camera.azimuth = ((camera.azimuth - (e.clientX - drag.x) * .35 + 180) % 360 + 360) % 360 - 180;
  camera.elevation = Math.max(5, Math.min(89, camera.elevation + (e.clientY - drag.y) * .25));
  drag.x = e.clientX; drag.y = e.clientY;
  scheduleCamera();
};
viewport.onpointerup = e => {
  if (e.pointerId === drag?.pointerId && !(e.buttons & 1)) finishDrag();
};
viewport.onpointercancel = viewport.onlostpointercapture = e => {
  if (e.pointerId === drag?.pointerId) finishDrag();
};
window.addEventListener('blur', () => { finishDrag(); if (cameraTimer !== null) sendCamera(); });
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { finishDrag(); if (cameraTimer !== null) sendCamera(); }
});
viewport.ondragstart = e => e.preventDefault();
viewport.addEventListener('wheel', e => {
  if (!cameraEnabled()) return;
  e.preventDefault();
  const pixels = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? viewport.clientHeight : 1);
  camera.distance = Math.max(.3, Math.min(20, camera.distance * Math.exp(pixels * .001)));
  scheduleCamera();
}, {passive: false});
document.onkeydown = e => {
  if (e.repeat || e.isComposing || e.ctrlKey || e.altKey || e.metaKey || !connected || stopped || stopping || !state?.ready) return;
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) || e.target.isContentEditable) return;
  const key = e.key.toLowerCase();
  // Space on a focused button keeps its native activation behavior.
  if (e.code === 'Space' && e.target.closest('button,a')) return;
  const id = e.code === 'Space' ? (state.paused ? 'run' : 'pause') :
    {n: 'step', r: 'reset', f: 'camera-reset'}[key] ||
    (state.live ? {w: 'cmd-forward', s: 'cmd-back', a: 'cmd-left', d: 'cmd-right', q: 'cmd-turn-left', e: 'cmd-turn-right', x: 'cmd-zero'}[key] : null);
  if (id) { e.preventDefault(); if (!$(id).disabled) $(id).click(); }
};

function renderLive(s) {
  const l = s.live;
  $('live-panel').style.display = 'block';
  $('live-model').textContent = l.run + ' · Iter ' + l.iteration;
  $('session-note').textContent = '速度指令作用于选中环境，持续有效直到新指令或重置。指令归零仍执行策略，暂停才会停止仿真。任一环境结束后暂停全部环境，重置对应环境后继续。';
  $('live-command').textContent = '当前指令 ' + l.command.map(v => v.toFixed(2)).join(' / ') + ' · 实测 ' + l.measured_velocity.map(v => v.toFixed(2)).join(' / ');
  if (observed(pendingVelocity, s)) {
    // Preserve any newer edits made while this request was being applied.
    if (pendingVelocity.env_id === s.env_id && sameCommand(pendingVelocity.value, l.command)) {
      if (JSON.stringify(drafts.get(s.env_id)) === JSON.stringify(pendingVelocity.draft)) drafts.delete(s.env_id);
    } else error('指令请求已处理，但环境或指令已被后续操作改变；请检查当前状态');
    pendingVelocity = null;
  }
  writeVelocity(drafts.get(s.env_id) || l.command);
  velocityIds.forEach((id, i) => { $(id).min = -l.command_limits[i]; $(id).max = l.command_limits[i]; });
  for (const [id, value] of Object.entries(presets)) $(id).setAttribute('aria-pressed', String(sameCommand(value, l.command)));
  if (l.blocked_envs.length) $('status').textContent = '需重置环境 ' + l.blocked_envs.join(', ');
}

function renderState(s) {
  state = s;
  if (!s.ready) { feedback(); return; }
  $('loading').style.display = 'none'; $('physics').textContent = s.physics; $('task').textContent = s.task;
  if (!$('renderer').options.length) for (const [value, report] of Object.entries(s.renderers)) {
    const option = new Option(names[value] + (report.supported === false ? ' · 不支持此场景' : report.metadata_ok ? '' : ' · 未安装'), value);
    option.disabled = !report.metadata_ok || report.supported === false; $('renderer').add(option);
  }
  if (!$('environment').options.length) for (let i = 0; i < s.num_envs; i++) $('environment').add(new Option(`环境 ${i}`, i));
  if (observed(pendingSelection, s)) {
    if (pendingSelection.env_id !== s.env_id) error('查看环境已被其他操作改变，请检查当前环境');
    pendingSelection = null;
  }
  if (observed(pendingReset, s)) {
    if (pendingReset.env_id !== s.env_id) {
      error('重置请求已处理，但查看环境已改变；请检查目标环境'); pendingReset = null;
    } else if (s.episode !== pendingReset.episode || (s.replay && s.replay.frame === 0)) {
      drafts.delete(pendingReset.env_id); pendingReset = null;
    }
  }
  if (observed(pendingPause, s)) pendingPause = null;
  if (observed(pendingSpeed, s)) pendingSpeed = null;
  if (observed(pendingFollow, s)) pendingFollow = null;
  if (observed(rendererRequest, s)) { pendingRenderer = null; rendererRequest = null; }
  if (observed(pendingResolution, s)) pendingResolution = null;
  if (observed(pendingFPS, s)) pendingFPS = null;
  const size = pendingResolution ? `${pendingResolution.width}x${pendingResolution.height}` : `${s.render_width}x${s.render_height}`;
  if (![...$('resolution').options].some(o => o.value === size)) $('resolution').add(new Option(size.replace('x', ' × ') + ' · 自定义', size));
  $('resolution').value = size;
  const targetFPS = String(pendingFPS ? pendingFPS.value : s.target_fps);
  if (![...$('target-fps').options].some(o => o.value === targetFPS)) $('target-fps').add(new Option(targetFPS, targetFPS));
  $('target-fps').value = targetFPS;
  $('render-performance').textContent = `目标 ${s.target_fps} FPS · 实际 ${s.width} × ${s.height} · 单帧 ${(s.jpeg_bytes / 1024).toFixed(0)} KB` + (s.render_idle ? ' · 画面静止' : ` · 耗时 ${s.frame_ms.toFixed(1)} ms`);
  if (!pendingRenderer) $('renderer').value = s.renderer;
  $('environment').value = String(pendingSelection ? pendingSelection.env_id : s.env_id);
  $('speed').value = String(pendingSpeed ? pendingSpeed.value : s.speed);
  $('renderer-label').textContent = pendingRenderer ? '正在切换渲染器…' : names[s.renderer];
  $('env-label').textContent = `ENV ${s.env_id}`;
  $('status').textContent = s.paused ? '已暂停' : '运行中';
  $('camera-hint').textContent = s.camera_control ? '拖动旋转 · 滚轮缩放' : '正交调试画面';
  if (!s.camera_control) { clearCameraTimer(); finishDrag(false); }
  if (observed(pendingCamera, s)) pendingCamera = null;
  if (!drag && cameraTimer === null && !pendingCamera) camera = {azimuth: s.camera.azimuth, elevation: s.camera.elevation, distance: s.camera.distance};
  $('time').textContent = s.time.toFixed(2) + ' s'; $('episode').textContent = s.episode + ' / ' + s.step;
  $('velocity').textContent = s.velocity.toFixed(3); $('reward').textContent = s.reward == null ? '—' : s.reward.toFixed(4);
  $('fps').textContent = s.render_idle ? '静止' : s.fps.toFixed(1);
  $('position').textContent = '位置 ' + s.position.map(x => x.toFixed(3)).join(', ');
  $('frame-info').textContent = `${s.width} × ${s.height} · Frame ${s.frame_id}${s.render_idle ? ' · 画面未变化' : ''}`;
  $('follow-wrap').style.display = s.replay || s.live ? 'block' : 'none'; $('follow').checked = pendingFollow ? pendingFollow.enabled : s.follow;
  if (s.live) renderLive(s);
  if (s.replay) {
    const r = s.replay;
    $('session-note').textContent = '回放已有运动记录，不运行策略或物理。播放与单步控制所有记录，回到开头仅作用于选中记录。';
    $('replay-panel').style.display = 'block'; $('step-label').textContent = '记录帧 / 总数';
    $('episode').textContent = (r.frame + 1) + ' / ' + r.frames; $('timeline').max = r.frames - 1;
    if (!scrubbing) $('timeline').value = r.frame;
    $('replay-info').textContent = r.file + ' · ' + (r.frame + 1) + ' / ' + r.frames + ' 帧';
    $('replay-command').textContent = r.segment + ' · ' + (r.command ? '记录指令 ' + r.command.join(' / ') : '无指令记录') + ' · ' + s.time.toFixed(2) + ' / ' + r.duration.toFixed(2) + ' s';
    if (r.ended) $('status').textContent = r.terminal;
  }
  if (s.error !== lastBackendError) { error(s.error); lastBackendError = s.error; }
  feedback();
}

async function poll() {
  if (stopped) return;
  try {
    const s = await fetchJSON('/api/state', {cache: 'no-store'});
    if (stopped) return; // A late response must not reactivate a stopped page.
    connected = true; failures = 0;
    $('connection').textContent = s.ready ? '已连接' : '准备渲染器';
    $('dot').classList.toggle('ready', s.ready);
    renderState(s);
  } catch (e) {
    if (stopped) return;
    if (connected) connectionEpoch++;
    queuedCamera = null;
    connected = false; failures++;
    clearCameraTimer(); finishDrag(false); pendingCamera = null;
    $('dot').classList.remove('ready'); $('connection').textContent = '连接断开 · 正在重连';
    $('camera-hint').textContent = '连接断开，显示最后画面；未发送操作已取消'; controls();
  } finally { if (!stopped) setTimeout(poll, Math.min(5000, 250 * 2 ** Math.min(failures, 5))); }
}
$('image').onerror = () => {
  if (!stopped) setTimeout(() => { if (!stopped) $('image').src = '/stream.mjpg?t=' + Date.now(); }, 1500);
};
controls();
poll();
