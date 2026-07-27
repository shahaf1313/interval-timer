"use strict";

// ---------- constants ----------
const PHASE_COLOR = { prep: "#8b7cf6", work: "#ff6a3d", rest: "#2dd4bf", done: "#4ade80" };
const STORAGE_KEY = "itw-data";

// ---------- state ----------
let workouts = [];
let nextId = 1;

let screen = "list"; // list | editor | run
let editingId = null;
let draft = null;

let phases = [];
let phaseIdx = 0;
let remaining = 0;
let paused = false;
let tickInterval = null;

let audioCtx = null;
let wakeLock = null;

// ---------- helpers ----------
function formatTime(totalSeconds) {
  const s = Math.max(0, Math.round(totalSeconds));
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function buildPhases(prep, blocks) {
  const list = [];
  if (prep > 0) list.push({ type: "prep", duration: prep, label: "הכנה" });
  blocks.forEach((b, bi) => {
    for (let s = 0; s < b.sets; s++) {
      list.push({
        type: "work", duration: b.work, label: "עבודה",
        blockIdx: bi, setIdx: s, totalSets: b.sets, totalBlocks: blocks.length,
      });
      const isLastSetOfLastBlock = bi === blocks.length - 1 && s === b.sets - 1;
      if (!isLastSetOfLastBlock && b.rest > 0) {
        list.push({
          type: "rest", duration: b.rest, label: "מנוחה",
          blockIdx: bi, setIdx: s, totalSets: b.sets, totalBlocks: blocks.length,
        });
      }
    }
  });
  list.push({ type: "done", duration: 0, label: "סיום" });
  return list;
}

function computeSummary(prep, blocks) {
  const totalSeconds = (prep > 0 ? prep : 0) + blocks.reduce((sum, b, bi) => {
    const isLast = bi === blocks.length - 1;
    const workTotal = b.sets * b.work;
    const restTotal = isLast ? (b.sets - 1) * b.rest : b.sets * b.rest;
    return sum + workTotal + Math.max(0, restTotal);
  }, 0);
  const totalSets = blocks.reduce((sum, b) => sum + b.sets, 0);
  return { totalSeconds, totalSets };
}

function blocksPreview(blocks) {
  return blocks.map((b) => `${b.sets}×(${b.work}/${b.rest})`).join("  ·  ");
}

function escapeHtml(str) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(str).replace(/[&<>"']/g, (c) => map[c]);
}

function newBlock() {
  return { id: nextId++, sets: 4, work: 30, rest: 15 };
}

function isDraftValid() {
  return draft && draft.blocks.length > 0 && draft.blocks.every((b) => b.sets >= 1 && b.work >= 1);
}

// ---------- persistence ----------
function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) {}
  return null;
}
function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ nextId, workouts }));
  } catch (e) {}
}
function seedWorkout() {
  return {
    id: nextId++,
    name: "אימון לדוגמה",
    prep: 10,
    blocks: [
      { id: nextId++, sets: 8, work: 50, rest: 10 },
      { id: nextId++, sets: 4, work: 20, rest: 40 },
    ],
  };
}

// ---------- audio / haptics ----------
function getCtx() {
  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AC();
  }
  return audioCtx;
}
function audioUnlock() {
  const ctx = getCtx();
  if (ctx.state === "suspended") ctx.resume();
}
function beep(freq = 800, dur = 0.12, vol = 0.22, type = "sine") {
  try {
    const ctx = getCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.value = vol;
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + dur);
    osc.stop(ctx.currentTime + dur + 0.03);
  } catch (e) {}
}
function vibrate(pattern) {
  try { if (navigator.vibrate) navigator.vibrate(pattern); } catch (e) {}
}
function playTick() { beep(880, 0.07, 0.15, "square"); vibrate(25); }
function playTransition(type) {
  if (type === "work") beep(1046, 0.18, 0.26, "sine");
  else if (type === "rest") beep(659, 0.18, 0.2, "sine");
  else if (type === "prep") beep(523, 0.18, 0.2, "sine");
  else if (type === "done") {
    beep(784, 0.15, 0.26);
    setTimeout(() => beep(988, 0.15, 0.26), 160);
    setTimeout(() => beep(1318, 0.28, 0.3), 320);
  }
  vibrate(type === "done" ? [100, 60, 100, 60, 220] : 90);
}
async function requestWakeLock() {
  try { if ("wakeLock" in navigator) wakeLock = await navigator.wakeLock.request("screen"); } catch (e) {}
}
function releaseWakeLock() {
  try { wakeLock && wakeLock.release(); wakeLock = null; } catch (e) {}
}

// ---------- timer engine ----------
function startTicking() {
  if (tickInterval) return;
  tickInterval = setInterval(tick, 1000);
}
function stopTicking() {
  clearInterval(tickInterval);
  tickInterval = null;
}
function tick() {
  remaining -= 1;
  if (remaining <= 0) {
    advance();
    return;
  }
  if (remaining <= 3) playTick();
  updateRunDisplay();
}
function advance() {
  const nextIdx = phaseIdx + 1;
  if (nextIdx >= phases.length) return;
  phaseIdx = nextIdx;
  const nextPhase = phases[phaseIdx];
  remaining = nextPhase.duration;
  playTransition(nextPhase.type);
  if (nextPhase.type === "done") {
    stopTicking();
    paused = false;
    releaseWakeLock();
  }
  render();
}

function startWorkout(workout) {
  audioUnlock();
  phases = buildPhases(workout.prep, workout.blocks);
  phaseIdx = 0;
  remaining = phases[0].duration;
  paused = false;
  playTransition(phases[0].type);
  requestWakeLock();
  screen = "run";
  render();
  stopTicking();
  startTicking();
}

function handleStop() {
  stopTicking();
  paused = false;
  releaseWakeLock();
  screen = "list";
  render();
}

function handleRestart() {
  phaseIdx = 0;
  remaining = phases[0].duration;
  paused = false;
  playTransition(phases[0].type);
  requestWakeLock();
  render();
  stopTicking();
  startTicking();
}

function pauseToggle() {
  paused = !paused;
  if (paused) stopTicking(); else startTicking();
  const btn = document.getElementById("pauseBtn");
  if (btn) btn.textContent = paused ? "▶" : "⏸";
}

// ---------- workout list actions ----------
function openNew() {
  draft = { name: "", prep: 10, blocks: [newBlock()] };
  editingId = null;
  screen = "editor";
  render();
}
function openEdit(w) {
  draft = { name: w.name, prep: w.prep, blocks: w.blocks.map((b) => ({ ...b })) };
  editingId = w.id;
  screen = "editor";
  render();
}
function deleteWorkout(id) {
  if (!confirm("למחוק את האימון?")) return;
  workouts = workouts.filter((w) => w.id !== id);
  persist();
  render();
}
function saveDraft() {
  if (!isDraftValid()) return null;
  const name = (draft.name || "").trim() || "אימון ללא שם";
  const saved = { ...draft, name, id: editingId != null ? editingId : nextId++ };
  workouts = editingId != null ? workouts.map((w) => (w.id === editingId ? saved : w)) : [...workouts, saved];
  persist();
  return saved;
}

// ---------- rendering ----------
function render() {
  const app = document.getElementById("app");
  if (screen === "list") app.innerHTML = renderListHTML();
  else if (screen === "editor") app.innerHTML = renderEditorHTML();
  else app.innerHTML = renderRunHTML();
}

function renderListHTML() {
  const cards = workouts.map((w) => {
    const { totalSeconds, totalSets } = computeSummary(w.prep, w.blocks);
    return `
      <div class="workout-card">
        <button class="play-btn" data-action="start-workout" data-id="${w.id}" aria-label="התחל אימון">▶</button>
        <div class="workout-info" data-action="start-workout" data-id="${w.id}">
          <div class="workout-name">${escapeHtml(w.name)}</div>
          <div class="workout-meta">${totalSets} סטים · כ-${formatTime(totalSeconds)} דק'</div>
          <div class="workout-preview">${escapeHtml(blocksPreview(w.blocks))}</div>
        </div>
        <div class="workout-actions">
          <button class="small-icon-btn" data-action="edit-workout" data-id="${w.id}" aria-label="ערוך">✎</button>
          <button class="small-icon-btn danger" data-action="delete-workout" data-id="${w.id}" aria-label="מחק">🗑</button>
        </div>
      </div>`;
  }).join("");

  const empty = workouts.length === 0 ? `
    <div class="empty-state">
      <div class="main">עדיין אין אימונים שמורים</div>
      <div class="sub">לחץ למטה כדי ליצור את האימון הראשון שלך</div>
    </div>` : "";

  return `
    <div class="container">
      <div>
        <div class="title num-display">האימונים שלי</div>
        <div class="subtitle">בחר אימון שמור והתחל בלחיצה אחת</div>
      </div>
      ${empty}
      ${cards}
      <button class="new-workout-btn" data-action="new-workout">+ אימון חדש</button>
    </div>`;
}

function renderEditorHTML() {
  const blocksHtml = draft.blocks.map((b, i) => `
    <div class="card">
      <div class="block-header">
        <span class="block-title">בלוק ${i + 1}</span>
        ${draft.blocks.length > 1 ? `<button class="delete-btn" data-action="remove-block" data-block-id="${b.id}" aria-label="מחק בלוק">✕</button>` : ""}
      </div>
      <div class="field-row">
        <div class="field-col">
          <div class="field-label">סטים</div>
          <input class="field-input" type="number" inputmode="numeric" min="1" value="${b.sets}" data-field="sets" data-block-id="${b.id}" />
        </div>
        <div class="field-col">
          <div class="field-label" style="color:var(--work)">עבודה (שנ')</div>
          <input class="field-input" type="number" inputmode="numeric" min="1" value="${b.work}" data-field="work" data-block-id="${b.id}" />
        </div>
        <div class="field-col">
          <div class="field-label" style="color:var(--rest)">מנוחה (שנ')</div>
          <input class="field-input" type="number" inputmode="numeric" min="0" value="${b.rest}" data-field="rest" data-block-id="${b.id}" />
        </div>
      </div>
    </div>`).join("");

  const { totalSeconds, totalSets } = computeSummary(draft.prep, draft.blocks);
  const valid = isDraftValid();

  return `
    <div class="container">
      <div class="title num-display">${editingId != null ? "עריכת אימון" : "אימון חדש"}</div>

      <div class="card">
        <div class="card-label">שם האימון</div>
        <input class="name-input" type="text" value="${escapeHtml(draft.name)}" placeholder="לדוגמה: HIIT יום שני" data-field="name" />
      </div>

      <div class="card">
        <div class="card-label">זמן הכנה (שניות)</div>
        <input class="small-input" type="number" inputmode="numeric" min="0" value="${draft.prep}" data-field="prep" />
      </div>

      ${blocksHtml}

      <button class="add-block-btn" data-action="add-block">+ הוסף בלוק</button>

      <div class="summary" id="summaryText">${totalSets} סטים &nbsp;•&nbsp; סה״כ כ-${formatTime(totalSeconds)} דקות</div>

      <div style="height:90px"></div>

      <div class="editor-bottom-bar">
        <button class="cancel-btn" data-action="cancel-editor">ביטול</button>
        <button class="save-btn" id="saveBtn" data-action="save-editor" ${valid ? "" : "disabled"}>שמור</button>
        <button class="start-btn-inline" id="startBtn" data-action="save-and-start" ${valid ? "" : "disabled"}>שמור והתחל</button>
      </div>
    </div>`;
}

function renderRunHTML() {
  const phase = phases[phaseIdx];
  if (!phase) return "";

  if (phase.type === "done") {
    return `
      <div class="run-screen">
        <div class="done-wrap">
          <div class="done-emoji">🏁</div>
          <div class="done-title num-display">סיום!</div>
          <div class="done-sub">כל הכבוד, האימון הושלם</div>
          <div class="controls-row">
            <button class="pill-btn" data-action="restart-run">התחל שוב</button>
            <button class="pill-btn secondary" data-action="back-to-list-done">חזרה לרשימה</button>
          </div>
        </div>
      </div>`;
  }

  const R = 92, CIRC = 2 * Math.PI * R;
  const fraction = phase.duration > 0 ? remaining / phase.duration : 0;
  const color = PHASE_COLOR[phase.type];
  const progressTxt = phase.totalBlocks
    ? `בלוק ${phase.blockIdx + 1} מתוך ${phase.totalBlocks}  ·  סט ${phase.setIdx + 1} מתוך ${phase.totalSets}`
    : "";
  const next = phases[phaseIdx + 1];
  const nextTxt = next ? (next.type === "done" ? "הבא: סיום" : `הבא: ${next.label} · ${formatTime(next.duration)}`) : "";

  return `
    <div class="run-screen">
      <div class="progress-text" id="progressText">${progressTxt}</div>
      <div class="phase-pill" id="phasePill" style="background:${color}22;color:${color}">${phase.label}</div>
      <div class="ring-wrap">
        <svg width="240" height="240" viewBox="0 0 240 240">
          <circle class="circle-bg" cx="120" cy="120" r="${R}" fill="none" stroke-width="12"/>
          <circle id="ringProgress" cx="120" cy="120" r="${R}" fill="none" stroke="${color}" stroke-width="12"
            stroke-linecap="round" stroke-dasharray="${CIRC}" stroke-dashoffset="${CIRC * (1 - fraction)}"
            transform="rotate(-90 120 120)" style="transition: stroke-dashoffset 0.9s linear"/>
        </svg>
        <div class="big-number num-display" id="bigNumber">${formatTime(remaining)}</div>
      </div>
      <div class="next-text" id="nextText">${nextTxt}</div>
      <div class="controls-row">
        <button class="icon-btn-small" data-action="skip" aria-label="דלג">⏭</button>
        <button class="icon-btn-big" id="pauseBtn" data-action="pause-toggle" aria-label="השהה או המשך">${paused ? "▶" : "⏸"}</button>
        <button class="icon-btn-small" data-action="stop-run" aria-label="עצור">⏹</button>
      </div>
    </div>`;
}

function updateRunDisplay() {
  const phase = phases[phaseIdx];
  if (!phase || phase.type === "done") return;
  const bigNum = document.getElementById("bigNumber");
  if (bigNum) bigNum.textContent = formatTime(remaining);
  const ring = document.getElementById("ringProgress");
  if (ring) {
    const R = 92, CIRC = 2 * Math.PI * R;
    const fraction = phase.duration > 0 ? remaining / phase.duration : 0;
    ring.setAttribute("stroke-dashoffset", CIRC * (1 - fraction));
  }
}

function updateSummaryDisplay() {
  const { totalSeconds, totalSets } = computeSummary(draft.prep, draft.blocks);
  const el = document.getElementById("summaryText");
  if (el) el.textContent = `${totalSets} סטים  •  סה״כ כ-${formatTime(totalSeconds)} דקות`;
  const valid = isDraftValid();
  const saveBtn = document.getElementById("saveBtn");
  const startBtn = document.getElementById("startBtn");
  if (saveBtn) saveBtn.disabled = !valid;
  if (startBtn) startBtn.disabled = !valid;
}

// ---------- event delegation ----------
function onAppClick(e) {
  const el = e.target.closest("[data-action]");
  if (!el) return;
  const action = el.dataset.action;
  const id = el.dataset.id ? Number(el.dataset.id) : null;

  switch (action) {
    case "start-workout": {
      const w = workouts.find((w) => w.id === id);
      if (w) startWorkout(w);
      break;
    }
    case "edit-workout": {
      const w = workouts.find((w) => w.id === id);
      if (w) openEdit(w);
      break;
    }
    case "delete-workout":
      deleteWorkout(id);
      break;
    case "new-workout":
      openNew();
      break;
    case "add-block":
      draft.blocks.push(newBlock());
      render();
      break;
    case "remove-block": {
      const bid = Number(el.dataset.blockId);
      draft.blocks = draft.blocks.filter((b) => b.id !== bid);
      render();
      break;
    }
    case "cancel-editor":
      draft = null;
      screen = "list";
      render();
      break;
    case "save-editor":
      if (saveDraft()) { screen = "list"; render(); }
      break;
    case "save-and-start": {
      const saved = saveDraft();
      if (saved) startWorkout(saved);
      break;
    }
    case "pause-toggle":
      pauseToggle();
      break;
    case "skip":
      advance();
      break;
    case "stop-run":
    case "back-to-list-done":
      handleStop();
      break;
    case "restart-run":
      handleRestart();
      break;
  }
}

function onAppInput(e) {
  const el = e.target;
  if (!draft) return;
  const field = el.dataset.field;
  if (!field) return;

  if (field === "name") {
    draft.name = el.value;
    return;
  }
  if (field === "prep") {
    draft.prep = Math.max(0, Number(el.value) || 0);
    updateSummaryDisplay();
    return;
  }
  if (el.dataset.blockId) {
    const bid = Number(el.dataset.blockId);
    const min = field === "rest" ? 0 : 1;
    const val = Math.max(min, Number(el.value) || min);
    const block = draft.blocks.find((b) => b.id === bid);
    if (block) block[field] = val;
    updateSummaryDisplay();
  }
}

// ---------- init ----------
function init() {
  const loaded = loadState();
  if (loaded && Array.isArray(loaded.workouts)) {
    workouts = loaded.workouts;
    nextId = loaded.nextId || 1;
  } else {
    workouts = [seedWorkout()];
  }

  const app = document.getElementById("app");
  app.addEventListener("click", onAppClick);
  app.addEventListener("input", onAppInput);

  render();
}

document.addEventListener("DOMContentLoaded", init);
