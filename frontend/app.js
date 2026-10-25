// ---- Config ----
const USE_MOCK = true;                    // test pipeline not real model
const API_URL = "/predict";               // POST, multipart field "audio"
const MOCK_URL = "mock_response.json";
const MIN_SECONDS = 5;
const MAX_SECONDS = 10;

const $ = (id) => document.getElementById(id);
const btnRecord = $("btn-record"), btnSubmit = $("btn-submit");
const timerEl = $("timer"), playback = $("playback");

// ---- State ----
let recorder = null, stream = null, chunks = [];
let startTime = 0, tickId = null, maxId = null;
let wavBlob = null, abortCtl = null;

function showScreen(name) {
  document.querySelectorAll(".screen").forEach((s) => (s.hidden = true));
  $("screen-" + name).hidden = false;
}

function showError(msg) {
  $("error-reason").textContent = msg;
  showScreen("error");
}

function resetRecordScreen() {
  wavBlob = null;
  if (playback.src) URL.revokeObjectURL(playback.src);
  playback.removeAttribute("src");
  playback.hidden = true;
  btnSubmit.hidden = true;
  btnRecord.textContent = "Record";
  timerEl.textContent = "0:00";
  showScreen("record");
}

// ---- Recording ----
async function startRecording() {
  chunks = [];
  wavBlob = null;
  playback.hidden = true;
  btnSubmit.hidden = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    return showError("Microphone access was denied. Allow it in your browser settings and try again.");
  }
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = onRecorderStop;
  recorder.start();
  startTime = Date.now();
  btnRecord.textContent = "Stop";
  tickId = setInterval(() => {
    const s = Math.floor((Date.now() - startTime) / 1000);
    timerEl.textContent = "0:" + String(s).padStart(2, "0");
  }, 200);
  maxId = setTimeout(stopRecording, MAX_SECONDS * 1000);
}

function stopRecording() {
  if (recorder && recorder.state === "recording") recorder.stop();
}

async function onRecorderStop() {
  const seconds = (Date.now() - startTime) / 1000;
  clearInterval(tickId);
  clearTimeout(maxId);
  stream.getTracks().forEach((t) => t.stop()); // turns the mic indicator off
  stream = null;
  btnRecord.textContent = "Record";

  if (seconds < MIN_SECONDS) {
    resetRecordScreen();
    return showError(`Recording too short (${seconds.toFixed(1)} s). Hold the vowel for at least ${MIN_SECONDS} seconds.`);
  }
  try {
    wavBlob = await blobToWav(new Blob(chunks, { type: recorder.mimeType }));
  } catch (e) {
    resetRecordScreen();
    return showError("Could not process the recording: " + e.message);
  }
  playback.src = URL.createObjectURL(wavBlob);
  playback.hidden = false;
  btnSubmit.hidden = false;
}

// ---- WAV conversion (mono, 16-bit PCM) ----
async function blobToWav(blob) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
  await ctx.close();
  const mono = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < d.length; i++) mono[i] += d[i] / buf.numberOfChannels;
  }
  return encodeWav(mono, buf.sampleRate);
}

function encodeWav(samples, rate) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (o, s) => [...s].forEach((ch, i) => view.setUint8(o + i, ch.charCodeAt(0)));
  str(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE"); str(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);        // PCM
  view.setUint16(22, 1, true);        // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  str(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

// ---- Submit ----
async function submit() {
  if (!wavBlob) return;
  showScreen("processing");
  abortCtl = new AbortController();
  const opts = { signal: abortCtl.signal };
  let res;
  try {
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 800)); // fake latency
      if (abortCtl.signal.aborted) return;
      res = await fetch(MOCK_URL, { ...opts, cache: "no-store" });
    } else {
      const fd = new FormData();
      fd.append("audio", wavBlob, "recording.wav");
      res = await fetch(API_URL, { ...opts, method: "POST", body: fd });
    }
  } catch (e) {
    if (e.name === "AbortError") return;
    return showError("Could not reach the server. Check your connection and try again.");
  }
  if (!res.ok) return showError(`Server error (${res.status}). Please try again.`);

  let data;
  try { data = await res.json(); }
  catch (e) { return showError("The server sent an unreadable response."); }

  const problem = validate(data);
  if (problem) return showError("Unexpected response from server: " + problem);
  renderResults(data);
}

function cancelSubmit() {
  if (abortCtl) abortCtl.abort();
  showScreen("record"); // recording is kept, so the user can submit again
}

// Shape (mock_response.json): { risk_score, label, model_name,
// top_features: [{ name, value, shap }], warnings: [] }
function validate(d) {
  if (!d || typeof d.risk_score !== "number") return "missing risk_score";
  if (d.risk_score < 0 || d.risk_score > 1) return "risk_score out of range";
  if (typeof d.label !== "string") return "missing label";
  if (!Array.isArray(d.top_features)) return "missing top_features";
  for (const f of d.top_features) {
    if (typeof f.name !== "string" || typeof f.value !== "number" || typeof f.shap !== "number") {
      return "bad feature entry";
    }
  }
  if (d.warnings !== undefined && !Array.isArray(d.warnings)) return "bad warnings";
  return null;
}

// ---- Results (plain list inside #chart; Chart.js replaces it later) ----
function renderResults(d) {
  $("risk-score").textContent = d.risk_score.toFixed(2);
  // "higher_risk" -> "Higher risk"
  const text = d.label.replace(/_/g, " ");
  $("risk-label").textContent = text.charAt(0).toUpperCase() + text.slice(1);

  // warnings (shown above the explanation, cleared each run)
  let w = $("warnings");
  if (!w) {
    w = document.createElement("p");
    w.id = "warnings";
    $("explanation").before(w);
  }
  w.textContent = (d.warnings || []).join(" ");

  drawChart(d.top_features);
  showScreen("results");
}

let chart = null;
function drawChart(features) {
  if (chart) chart.destroy();
  chart = new Chart($("chart-canvas"), {
    type: "bar",
    data: {
      labels: features.map((f) => f.name),
      datasets: [{
        data: features.map((f) => f.shap),
        backgroundColor: features.map((f) => (f.shap > 0 ? "#d9534f" : "#4a7fb5")),
      }],
    },
    options: {
      indexAxis: "y",
      plugins: {
        legend: { display: false },
        title: { display: true, text: "Top features (red raises risk, blue lowers it)" },
      },
      scales: {
          x: { title: { display: true, text: "SHAP value" } },
          y: { ticks: { autoSkip: false } },
      },
      responsive: true,
      maintainAspectRatio: false
    },
  });
}

// ---- Wiring ----
btnRecord.onclick = () => (recorder && recorder.state === "recording" ? stopRecording() : startRecording());
btnSubmit.onclick = submit;
$("btn-cancel").onclick = cancelSubmit;
$("btn-again").onclick = resetRecordScreen;
$("btn-retry").onclick = resetRecordScreen;
resetRecordScreen();