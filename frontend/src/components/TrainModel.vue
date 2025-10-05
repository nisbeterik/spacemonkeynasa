<script setup>
import { ref } from 'vue'
import { postCSV, getJSON } from '../api'

const file = ref(null)
const busy = ref(false)
const result = ref(null)
const error = ref('')
const health = ref(null)

function onFile(e){ file.value = e.target.files[0] }

async function checkHealth() {
  try { health.value = await getJSON('/api/health') }
  catch (e) { health.value = { ok:false, error: e.message } }
}

function reset() {
  result.value = null
  error.value = ''
}

async function uploadTrain() {
  if (!file.value) return
  reset()
  busy.value = true
  try {
    // simple client-side guard: must be .csv
    if (!file.value.name.toLowerCase().endsWith('.csv')) {
      throw new Error('Please select a .csv file')
    }
    result.value = await postCSV('/api/train_csv', file.value, false)
  } catch (e) {
    error.value = e.message || String(e)
  } finally {
    busy.value = false
    await checkHealth()
  }
}
</script>

<template>
  <div class="card">
    <div class="head">
      <h2>Train model</h2>
      <button class="ghost" @click="checkHealth">health</button>
    </div>
    <p>Upload a KOI CSV to train a fresh model on the server.</p>

    <input type="file" accept=".csv" @change="onFile" />
    <div class="actions">
      <button :disabled="!file || busy" @click="uploadTrain">
        {{ busy ? 'Training…' : 'Train model' }}
      </button>
      <span v-if="file" class="filename">{{ file.name }}</span>
    </div>

    <p v-if="error" class="error">⚠ {{ error }}</p>

    <div v-if="result" class="result">
      <h3>Saved</h3>
      <ul>
        <li><code>{{ result.saved_model }}</code></li>
        <li><code>{{ result.saved_schema }}</code></li>
      </ul>

      <h3>Metrics (holdout)</h3>
      <div class="metrics">
        <table>
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
          <tbody>
            <tr v-for="cls in Object.keys(result.metrics).filter(k => !['accuracy','macro avg','weighted avg'].includes(k))"
                :key="cls">
              <td>{{ cls }}</td>
              <td>{{ Number(result.metrics[cls].precision ?? 0).toFixed(3) }}</td>
              <td>{{ Number(result.metrics[cls].recall ?? 0).toFixed(3) }}</td>
              <td>{{ Number(result.metrics[cls]['f1-score'] ?? 0).toFixed(3) }}</td>
              <td>{{ result.metrics[cls].support }}</td>
            </tr>
          </tbody>
        </table>
        <div class="agg">
          <p><strong>Accuracy:</strong> {{ Number(result.metrics.accuracy ?? 0).toFixed(3) }}</p>
          <p>
            <strong>Macro avg F1:</strong> {{ Number(result.metrics['macro avg']?.['f1-score'] ?? 0).toFixed(3) }},
            <strong>Weighted avg F1:</strong> {{ Number(result.metrics['weighted avg']?.['f1-score'] ?? 0).toFixed(3) }}
          </p>
        </div>
      </div>
    </div>

    <div v-if="health" class="health">
      <h3>Health</h3>
      <pre>{{ health }}</pre>
    </div>
  </div>
</template>

<style scoped>
.card { padding: 1rem; border: 1px solid #e7e7e7; border-radius: 12px; background:#fff; }
.head { display:flex; align-items:center; justify-content:space-between; margin-bottom:.5rem }
.actions { margin-top:.5rem; display:flex; gap: .5rem; align-items:center; }
.filename { font-size:.9rem; color:#555; }
button { padding:.5rem .9rem; border:1px solid #ccc; border-radius:8px; background:#f8f8f8; cursor:pointer }
button:disabled { opacity:.6; cursor:not-allowed }
button.ghost { background:transparent }
.error { color:#b00020; margin-top:.5rem }
.metrics { margin-top:.75rem }
table { border-collapse: collapse; width:100%; }
th, td { border:1px solid #eee; padding:.4rem .5rem; text-align:left; }
.agg { margin-top:.5rem; }
.health { margin-top:1rem; }
</style>
