<script setup>
import { ref } from 'vue'
import { postCSV } from '../api'

const file = ref(null)
const busy = ref(false)
const error = ref('')
const done = ref(false)

function onFile(e){ file.value = e.target.files[0] }
function reset(){ error.value=''; done.value=false }

async function runCheck(){
  if(!file.value) return
  reset()
  busy.value = true
  try {
    if (!file.value.name.toLowerCase().endsWith('.csv')) {
      throw new Error('Please select a .csv file')
    }
    const blob = await postCSV('/api/check_exo_csv', file.value, true)
    // download
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'exo_check_results.csv'
    a.click()
    URL.revokeObjectURL(url)
    done.value = true
  } catch (e) {
    error.value = e.message || String(e)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="card">
    <h2>Check exo planets</h2>
    <p>
      Upload a CSV and we’ll predict planet-likeness and enrich with KOI metadata.<br />
      Output columns include:
      <code>kepid</code>, <code>kepoi_name</code>, <code>kepler_name</code>,
      <code>exoplanet_archive_disposition</code>, <code>disposition_using_kepler_data</code>,
      <code>planet_like_prob</code>, <code>planet_like_label</code>.
    </p>

    <input type="file" accept=".csv" @change="onFile" />
    <div class="actions">
      <button :disabled="!file || busy" @click="runCheck">
        {{ busy ? 'Processing…' : 'Check exo planets' }}
      </button>
      <span v-if="file" class="filename">{{ file.name }}</span>
    </div>

    <p v-if="error" class="error">⚠ {{ error }}</p>
    <p v-if="done" class="ok">✅ Downloaded <code>exo_check_results.csv</code></p>
  </div>
</template>

<style scoped>
.card { padding: 1rem; border: 1px solid #e7e7e7; border-radius: 12px; background:#fff; }
.actions { margin-top:.5rem; display:flex; gap:.5rem; align-items:center; }
.filename { font-size:.9rem; color:#555; }
button { padding:.5rem .9rem; border:1px solid #ccc; border-radius:8px; background:#f8f8f8; cursor:pointer }
button:disabled { opacity:.6; cursor:not-allowed }
.error { color:#b00020; margin-top:.5rem }
.ok { color:#087f5b; margin-top:.5rem }
</style>
