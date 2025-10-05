<!-- src/pages/UploadPage.vue -->
<template>
  <div class="page upload" aria-labelledby="upload-title">
    <!-- Cosmic background -->
    <div class="bg" aria-hidden="true">
      <div class="nebula"></div>
      <div class="stars layer-1"></div>
      <div class="stars layer-2"></div>
      <div class="gridlines"></div>
    </div>

    <!-- Header -->
    <header class="header">
      <h1 id="upload-title" class="title">
        SpaceMonkey <span class="accent">Exoplanets</span>
      </h1>
      <p class="muted lead">
        Train a model, check exo-planets from a CSV, and evaluate predictions.
      </p>
    </header>

    <!-- Cards -->
    <section class="grid" aria-label="Workflows">
      <!-- Train Model -->
      <div class="card fade-in" :class="{ busy: busyTrain }" style="--i:0">
        <h2>🔭 Train Model</h2>
        <p>Upload a KOI CSV (must include <code>koi_disposition</code>).</p>

        <label
          class="drop"
          :class="{ active: dragTrain }"
          for="trainInput"
          @dragover.prevent="dragTrain = true"
          @dragleave.prevent="dragTrain = false"
          @drop.prevent="onDropTrain"
        >
          <input id="trainInput" class="file-input" type="file" accept=".csv,text/csv" @change="onPickTrain" />
          <span class="drop-icon">⬆️</span>
          <span class="drop-text">
            Drop CSV here or <b>browse</b>
            <small v-if="trainFile"> — {{ fileLabel(trainFile) }}</small>
          </span>
        </label>

        <button
          class="btn"
          :disabled="busyTrain || !trainFile"
          @click="doTrain"
          :aria-busy="busyTrain ? 'true' : 'false'"
        >
          <span v-if="busyTrain" class="spinner" aria-hidden="true"></span>
          {{ busyTrain ? 'Training…' : 'Train' }}
        </button>

        <div v-if="trainResult" class="panel">
          <h3>Metrics</h3>
          <pre>{{ pretty(trainResult.metrics) }}</pre>
        </div>
        <p v-if="errTrain" class="error" role="alert" aria-live="polite">{{ errTrain }}</p>
      </div>

      <!-- Check Exo Planets -->
      <div class="card fade-in" :class="{ busy: busyCheck }" style="--i:1">
        <h2>🛰️ Check Exo Planets</h2>
        <p>
          Upload a CSV to receive an enriched CSV with
          <code>kepoi_name</code>, <code>planet_like_prob</code>, <code>planet_like_label</code>, KOI fields, and <code>status</code>.
        </p>

        <label
          class="drop"
          :class="{ active: dragCheck }"
          for="checkInput"
          @dragover.prevent="dragCheck = true"
          @dragleave.prevent="dragCheck = false"
          @drop.prevent="onDropCheck"
        >
          <input id="checkInput" class="file-input" type="file" accept=".csv,text/csv" @change="onPickCheck" />
          <span class="drop-icon">⬆️</span>
          <span class="drop-text">
            Drop CSV here or <b>browse</b>
            <small v-if="checkFile"> — {{ fileLabel(checkFile) }}</small>
          </span>
        </label>

        <button
          class="btn"
          :disabled="busyCheck || !checkFile"
          @click="doCheck"
          :aria-busy="busyCheck ? 'true' : 'false'"
        >
          <span v-if="busyCheck" class="spinner" aria-hidden="true"></span>
          {{ busyCheck ? 'Generating…' : 'Generate CSV' }}
        </button>

        <p v-if="errCheck" class="error" role="alert" aria-live="polite">{{ errCheck }}</p>
      </div>

      <!-- Evaluate Predictions -->
      <div class="card fade-in" :class="{ busy: busyEval }" style="--i:2">
        <h2>📈 Evaluate Predictions</h2>
        <p>Upload the original labeled CSV and your <code>exo_check_results.csv</code>.</p>

        <div class="pair">
          <div>
            <label class="lbl">Actual CSV</label>

            <label
              class="drop small"
              :class="{ active: dragActual }"
              for="actualInput"
              @dragover.prevent="dragActual = true"
              @dragleave.prevent="dragActual = false"
              @drop.prevent="onDropActual"
            >
              <input id="actualInput" class="file-input" type="file" accept=".csv,text/csv" @change="onPickActual" />
              <span class="drop-icon sm">⬆️</span>
              <span class="drop-text">
                Drop or <b>browse</b>
                <small v-if="actualFile"> — {{ fileLabel(actualFile) }}</small>
              </span>
            </label>
          </div>

          <div>
            <label class="lbl">Pred CSV</label>

            <label
              class="drop small"
              :class="{ active: dragPred }"
              for="predInput"
              @dragover.prevent="dragPred = true"
              @dragleave.prevent="dragPred = false"
              @drop.prevent="onDropPred"
            >
              <input id="predInput" class="file-input" type="file" accept=".csv,text/csv" @change="onPickPred" />
              <span class="drop-icon sm">⬆️</span>
              <span class="drop-text">
                Drop or <b>browse</b>
                <small v-if="predFile"> — {{ fileLabel(predFile) }}</small>
              </span>
            </label>
          </div>
        </div>

        <button
          class="btn"
          :disabled="busyEval || !actualFile || !predFile"
          @click="doEvaluate"
          :aria-busy="busyEval ? 'true' : 'false'"
        >
          <span v-if="busyEval" class="spinner" aria-hidden="true"></span>
          {{ busyEval ? 'Evaluating…' : 'Evaluate' }}
        </button>

        <div v-if="evalResult" class="panel">
          <h3>Results</h3>
          <ul>
            <li><b>Compared rows:</b> {{ evalResult.compared_rows }}</li>
            <li><b>Correct:</b> {{ evalResult.correct }}</li>
            <li><b>Accuracy:</b> {{ evalResult.accuracy_percent }}%</li>
            <li><b>Threshold:</b> {{ evalResult.threshold }}</li>
          </ul>
          <h4>Confusion Matrix</h4>
          <ul>
            <li>TP: {{ evalResult.confusion_matrix.tp }}</li>
            <li>TN: {{ evalResult.confusion_matrix.tn }}</li>
            <li>FP: {{ evalResult.confusion_matrix.fp }}</li>
            <li>FN: {{ evalResult.confusion_matrix.fn }}</li>
          </ul>
        </div>

        <p v-if="errEval" class="error" role="alert" aria-live="polite">{{ errEval }}</p>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { trainCSV, checkExoCSV, evaluatePairCSV, downloadBlob } from '@/api/api'

const trainFile = ref(null)
const checkFile = ref(null)
const actualFile = ref(null)
const predFile = ref(null)

const busyTrain = ref(false)
const busyCheck = ref(false)
const busyEval = ref(false)

const trainResult = ref(null)
const evalResult = ref(null)

const errTrain = ref('')
const errCheck = ref('')
const errEval = ref('')

const dragTrain = ref(false)
const dragCheck = ref(false)
const dragActual = ref(false)
const dragPred = ref(false)

function onPickTrain(e) { trainFile.value = e.target.files?.[0] || null }
function onPickCheck(e) { checkFile.value = e.target.files?.[0] || null }
function onPickActual(e) { actualFile.value = e.target.files?.[0] || null }
function onPickPred(e) { predFile.value = e.target.files?.[0] || null }

function onDropTrain(e) { dragTrain.value = false; trainFile.value = e.dataTransfer.files?.[0] || null }
function onDropCheck(e) { dragCheck.value = false; checkFile.value = e.dataTransfer.files?.[0] || null }
function onDropActual(e) { dragActual.value = false; actualFile.value = e.dataTransfer.files?.[0] || null }
function onDropPred(e) { dragPred.value = false; predFile.value = e.dataTransfer.files?.[0] || null }

function pretty(x) { try { return JSON.stringify(x, null, 2) } catch { return String(x) } }
function fileLabel(f) { return f ? `${f.name} (${formatBytes(f.size)})` : '' }
function formatBytes(b = 0) {
  const u = ['B','KB','MB','GB']; let i = 0, n = b
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++ }
  return `${n.toFixed(n < 10 && i ? 1 : 0)} ${u[i]}`
}

async function doTrain() {
  errTrain.value = ''; trainResult.value = null; busyTrain.value = true
  try { trainResult.value = await trainCSV(trainFile.value) }
  catch (e) { errTrain.value = e?.message || 'Train failed' }
  finally { busyTrain.value = false }
}

async function doCheck() {
  errCheck.value = ''; busyCheck.value = true
  try { const blob = await checkExoCSV(checkFile.value); downloadBlob(blob, 'exo_check_results.csv') }
  catch (e) { errCheck.value = e?.message || 'Check failed' }
  finally { busyCheck.value = false }
}

async function doEvaluate() {
  errEval.value = ''; evalResult.value = null; busyEval.value = true
  try { evalResult.value = await evaluatePairCSV(actualFile.value, predFile.value) }
  catch (e) { errEval.value = e?.message || 'Evaluate failed' }
  finally { busyEval.value = false }
}
</script>

<style scoped>
/* Theme (scoped to this page wrapper) */
.upload {
  --bg: #0b1020;
  --text: #e5e7eb;
  --muted: #9ca3af;
  --card: #0b1220;
  --card-2: #0a0f1a;
  --border: #1f2937;
  --primary-1: #22d3ee;
  --primary-2: #6366f1;
  --shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
}

/* Layout */
.page.upload {
  position: relative;
  color: var(--text);
  min-height: 100svh;
  padding: clamp(20px, 3.5vw, 36px) clamp(16px, 4vw, 48px);
  overflow: clip;
}

.header {
  position: relative;
  z-index: 1;
  max-width: 1100px;
  margin: 0 auto clamp(18px, 4vh, 28px);
  text-align: center;
}

.title {
  margin: 0;
  font-weight: 900;
  letter-spacing: -0.02em;
  font-size: clamp(28px, 5vw, 44px);
  background: linear-gradient(135deg, var(--primary-1), var(--primary-2));
  background-clip: text;
  -webkit-background-clip: text;
  color: transparent;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 0 30px rgba(99,102,241,0.18);
}
.title .accent { -webkit-text-fill-color: currentColor; color: var(--text); opacity: 0.9; }

.lead { color: var(--muted); margin-top: 8px; }

/* Grid of cards */
.grid {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: clamp(14px, 2vw, 20px);
  max-width: 1100px;
  margin: 0 auto;
}

/* Card */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(8px);
  transform: translateY(0);
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.card:hover { transform: translateY(-2px); }
.card.busy { opacity: 0.95; }

.card h2 { margin: 0 0 6px; font-size: 20px; }
.card p { color: var(--muted); margin-top: 4px; }

/* Drag & Drop */
.file-input { display: none; }

.drop {
  margin: 12px 0;
  padding: 16px;
  border: 1px dashed rgba(255,255,255,0.22);
  border-radius: 12px;
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  cursor: pointer;
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: 12px;
  transition: border-color .2s ease, background .2s ease, transform .2s ease;
}
.drop:hover { transform: translateY(-1px); }
.drop.active { border-color: var(--primary-1); background: rgba(34,211,238,0.08); }
.drop.small { padding: 12px; }
.drop-icon { font-size: 20px; line-height: 0; opacity: .9; }
.drop-icon.sm { font-size: 18px; }
.drop-text { color: var(--text); }
.drop-text small { color: var(--muted); }

/* Buttons */
.btn {
  background: linear-gradient(135deg, var(--primary-1), var(--primary-2));
  border: none;
  color: white;
  padding: 10px 14px;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 600;
  box-shadow: var(--shadow);
  transition: transform .15s ease, filter .2s ease, box-shadow .2s ease, opacity .2s ease;
}
.btn:hover { transform: translateY(-1px); }
.btn:active { transform: translateY(0); }
.btn:disabled { opacity: .6; cursor: not-allowed; }

.spinner {
  display: inline-block;
  width: 16px; height: 16px;
  margin-right: 8px;
  border: 2px solid rgba(255,255,255,0.35);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin .8s linear infinite;
  vertical-align: -2px;
}

/* Results / panels */
.panel {
  background: var(--card-2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  margin-top: 12px;
  overflow: auto;
  max-height: 320px;
}
.error { color: #ef4444; margin-top: 8px; }
.lbl { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }

.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 560px) { .pair { grid-template-columns: 1fr; } }

/* Background FX */
.bg { position: absolute; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; }
.nebula {
  position: absolute; inset: -20%;
  background:
    radial-gradient(800px 600px at 20% 30%, rgba(99,102,241,0.18), transparent 60%),
    radial-gradient(700px 500px at 80% 20%, rgba(34,211,238,0.15), transparent 60%),
    radial-gradient(600px 800px at 70% 80%, rgba(147,197,253,0.08), transparent 70%);
  filter: blur(10px) saturate(120%);
  animation: drift 40s linear infinite alternate;
}
.stars {
  position: absolute; inset: -100% -100% 0 -100%;
  background-repeat: repeat; background-size: 600px 600px;
  opacity: .7; mix-blend-mode: screen;
}
.stars.layer-1 {
  background-image:
    radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.9) 99%, transparent),
    radial-gradient(1.5px 1.5px at 60% 70%, rgba(255,255,255,0.8) 99%, transparent),
    radial-gradient(1px 1px at 80% 20%, rgba(255,255,255,0.6) 99%, transparent);
  animation: starScroll 80s linear infinite;
}
.stars.layer-2 {
  background-image:
    radial-gradient(1px 1px at 30% 60%, rgba(255,255,255,0.6) 99%, transparent),
    radial-gradient(1.5px 1.5px at 70% 40%, rgba(255,255,255,0.5) 99%, transparent),
    radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.4) 99%, transparent);
  animation: starScroll 140s linear infinite reverse; opacity: .5;
}
.gridlines {
  position: absolute; inset: 0;
  background:
    linear-gradient(transparent 95%, rgba(99,102,241,0.18) 96%) 0 0 / 100% 32px,
    linear-gradient(90deg, transparent 95%, rgba(34,211,238,0.12) 96%) 0 0 / 32px 100%;
  mask: linear-gradient(#000, transparent 70%);
  opacity: .25;
  transform: perspective(800px) rotateX(55deg) translateY(35%);
  transform-origin: top center; filter: blur(0.2px);
}

/* Animations */
.fade-in { animation: fadeUp .7s ease both; animation-delay: calc(var(--i, 0) * 80ms); }
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes drift { from { transform: translate3d(-2%, -1%, 0) scale(1.02); } to { transform: translate3d(1%, 2%, 0) scale(1.04); } }
@keyframes starScroll { from { transform: translateY(0); } to { transform: translateY(600px); } }
@keyframes spin { to { transform: rotate(360deg); } }

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  .nebula, .stars.layer-1, .stars.layer-2, .fade-in, .spinner { animation: none !important; }
}
</style>
