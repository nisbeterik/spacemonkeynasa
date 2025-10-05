<script setup>
import { ref, onMounted } from 'vue'
import TrainModel from './components/TrainModel.vue'
import CheckExoPlanets from './components/CheckExoPlanets.vue'

const tab = ref('train')
onMounted(() => { /* could ping /api/health here if desired */ })
</script>

<template>
  <main class="container">
    <h1>Exoplanet Finder</h1>
    <p class="tagline">Train a KOI model and check CSV files for exoplanet candidates.</p>

    <div class="tabs">
      <button :class="{active: tab==='train'}" @click="tab='train'">Train model</button>
      <button :class="{active: tab==='check'}" @click="tab='check'">Check exo planets</button>
    </div>

    <section v-if="tab==='train'">
      <TrainModel />
    </section>
    <section v-else>
      <CheckExoPlanets />
    </section>
  </main>
</template>

<style>
.container { max-width: 960px; margin: 2rem auto; padding: 1rem; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
h1 { margin: 0 0 .25rem 0; }
.tagline { margin: 0 0 1rem 0; color:#555; }
.tabs { display:flex; gap:.5rem; margin-bottom:1rem; }
button { padding:.55rem .95rem; border:1px solid #ccc; border-radius:10px; background:#f8f8f8; cursor:pointer; }
button.active { background:#e9f2ff; border-color:#88b4ff; }
section { margin-top:.5rem; }
code { background:#f4f4f4; padding:.1rem .3rem; border-radius:6px; }
</style>
