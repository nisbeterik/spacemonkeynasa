<!-- src/pages/UploadPage.vue -->
<template>
  <div class="page">
    <h1>SpaceMonkey Exoplanets</h1>
    <p class="muted">Train a model, check exo planets from a CSV, and evaluate predictions.</p>

    <section class="grid">
      <!-- Train Model -->
      <div class="card">
        <h2>Train Model</h2>
        <p>Upload a KOI CSV (must include <code>koi_disposition</code>).</p>
        <input type="file" accept=".csv,text/csv" @change="onPickTrain" />
        <button :disabled="busyTrain || !trainFile" @click="doTrain">
          {{ busyTrain ? 'Training…' : 'Train' }}
        </button>
        <div v-if="trainResult" class="panel">
          <h3>Metrics</h3>
          <pre>{{ pretty(trainResult.metrics) }}</pre>
        </div>
        <p v-if="errTrain" class="error">{{ errTrain }}</p>
      </div>

      <!-- Check Exo Planets -->
      <div class="card">
        <h2>Check Exo Planets</h2>
        <p>Upload a CSV to receive an enriched CSV with
          <code>kepoi_name</code>, <code>planet_like_prob</code>, <code>planet_like_label</code>, KOI fields, and <code>status</code>.
        </p>
        <input type="file" accept=".csv,text/csv" @change="onPickCheck" />
        <button :disabled="busyCheck || !checkFile" @click="doCheck">
          {{ busyCheck ? 'Generating…' : 'Generate CSV' }}
        </button>
        <p v-if="errCheck" class="error">{{ errCheck }}</p>
      </div>

      <!-- Evaluate Predictions -->
      <div class="card">
        <h2>Evaluate Predictions</h2>
        <p>Upload the original labeled CSV and your <code>exo_check_results.csv</code>.</p>
        <div class="pair">
          <div>
            <label class="lbl">Actual CSV</label>
            <input type="file" accept=".csv,text/csv" @change="onPickActual" />
          </div>
          <div>
            <label class="lbl">Pred CSV</label>
            <input type="file" accept=".csv,text/csv" @change="onPickPred" />
          </div>
        </div>
        <button :disabled="busyEval || !actualFile || !predFile" @click="doEvaluate">
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

        <p v-if="errEval" class="error">{{ errEval }}</p>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import {
  trainCSV,
  checkExoCSV,
  evaluatePairCSV,
  downloadBlob,
} from '@/api/api';

const trainFile = ref(null);
const checkFile = ref(null);
const actualFile = ref(null);
const predFile = ref(null);

const busyTrain = ref(false);
const busyCheck = ref(false);
const busyEval = ref(false);

const trainResult = ref(null);
const evalResult = ref(null);

const errTrain = ref('');
const errCheck = ref('');
const errEval = ref('');

function onPickTrain(e) { trainFile.value = e.target.files?.[0] || null; }
function onPickCheck(e) { checkFile.value = e.target.files?.[0] || null; }
function onPickActual(e) { actualFile.value = e.target.files?.[0] || null; }
function onPickPred(e) { predFile.value = e.target.files?.[0] || null; }

function pretty(x) { try { return JSON.stringify(x, null, 2); } catch { return String(x); } }

async function doTrain() {
  errTrain.value = ''; trainResult.value = null; busyTrain.value = true;
  try { trainResult.value = await trainCSV(trainFile.value); }
  catch (e) { errTrain.value = e.message || 'Train failed'; }
  finally { busyTrain.value = false; }
}

async function doCheck() {
  errCheck.value = ''; busyCheck.value = true;
  try { const blob = await checkExoCSV(checkFile.value); downloadBlob(blob, 'exo_check_results.csv'); }
  catch (e) { errCheck.value = e.message || 'Check failed'; }
  finally { busyCheck.value = false; }
}

async function doEvaluate() {
  errEval.value = ''; evalResult.value = null; busyEval.value = true;
  try { evalResult.value = await evaluatePairCSV(actualFile.value, predFile.value); }
  catch (e) { errEval.value = e.message || 'Evaluate failed'; }
  finally { busyEval.value = false; }
}
</script>

<style scoped>
.page { max-width: 1100px; margin: 40px auto; padding: 0 16px; }
h1 { font-weight: 800; letter-spacing: -0.02em; margin-bottom: 8px; }
.muted { color: #6b7280; margin-bottom: 24px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
.card { background: #0b1220; color: #e5e7eb; border: 1px solid #1f2937; border-radius: 16px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.3); }
.card h2 { margin-top: 0; font-size: 20px; }
.card p { color: #9ca3af; }
input[type="file"] { margin: 10px 0; }
button {
  background: linear-gradient(135deg, #22d3ee, #6366f1);
  border: none; color: white; padding: 10px 14px; border-radius: 10px; cursor: pointer;
}
button:disabled { opacity: .6; cursor: not-allowed; }
.error { color: #ef4444; margin-top: 8px; }
.panel { background: #0a0f1a; border: 1px solid #1f2937; border-radius: 12px; padding: 12px; margin-top: 12px; overflow: auto; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.lbl { display: block; font-size: 12px; color: #9ca3af; margin-bottom: 4px; }
code { background: #111827; padding: 2px 6px; border-radius: 6px; }
</style>
