<script setup>
import { ref } from 'vue'

const csvFile = ref(null)
const isLoading = ref(false)

function handleFileChange(event) {
  csvFile.value = event.target.files[0]
  console.log('Selected file:', csvFile.value)
}

function handlePredict() {
  if (!csvFile.value) {
    alert('Please upload a CSV file first.')
    return
  }
  isLoading.value = true
  console.log(`Ready to predict for file: ${csvFile.value.name}`)

  // Fake loading animation for 5s
  setTimeout(() => {
    isLoading.value = false
    alert(`Prediction complete for file: ${csvFile.value.name}`)
  }, 5000)
}
</script>

<template>
  <div class="space-page">
    <div class="logo-container">
      <img alt="SpaceMonkey logo" class="logo" src="../assets/logo.svg" />

      <!-- Orbiting Earth animation -->
      <div v-if="isLoading" class="orbit">
        <div class="earth"></div>
      </div>
    </div>

    <div class="wrapper">
      <h1>Exoplanet Predictor</h1>
      <p>
        Upload a CSV file containing astronomical data. Our trained AI model
        will analyze the data and predict which entries are exoplanets,
        candidates, or neither.
      </p>

      <div class="upload-section">
        <input 
          type="file" 
          accept=".csv" 
          @change="handleFileChange"
          class="file-input"
        />
        <button @click="handlePredict">Predict</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Fullscreen layout */
.space-page {
  position: fixed;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 5vh;
  background: radial-gradient(circle at top, #0b0c29, #1c1c3c, #000);
  color: #fff;
  text-align: center;
  padding: 2rem;
}

/* Logo + orbit container */
.logo-container {
  position: relative;
  width: 250px;
  height: 250px;
  margin-bottom: 2rem;
}

.logo {
  width: 100%;
  height: auto;
  filter: drop-shadow(0 0 10px #0ff);
  position: relative;
  z-index: 2;
}

/* Orbit wrapper */
.orbit {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 200px;
  height: 200px;
  margin: -100px 0 0 -100px; /* center orbit */
  border-radius: 50%;
  animation: spin 4s linear infinite;
  z-index: 1;
}

/* Earth */
.earth {
  width: 30px;
  height: 30px;
  background: radial-gradient(circle at 30% 30%, #3cf, #036);
  border-radius: 50%;
  box-shadow: 0 0 10px #0ff;
  position: absolute;
  top: 0;
  left: 50%;
  margin-left: -15px;
}

/* Orbit spinning */
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Text styling */
.wrapper h1 {
  font-size: 2.2rem;
  margin-bottom: 1rem;
  text-shadow: 0 0 10px #0ff, 0 0 20px #08f;
}

.wrapper p {
  font-size: 1.2rem;
  margin-bottom: 2rem;
  text-shadow: 0 0 5px #0ff;
  max-width: 600px;
  line-height: 1.5;
  margin-inline: auto;
}

/* Upload section */
.upload-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2rem;
}

/* File input */
.file-input {
  background: rgba(255, 255, 255, 0.05);
  border: 2px solid #0ff;
  padding: 0.8rem 1rem;
  border-radius: 10px;
  color: #fff;
  cursor: pointer;
}

/* Button */
button {
  background: linear-gradient(180deg, #0ff, #08f);
  color: #000;
  font-weight: bold;
  padding: 1rem 2.5rem;
  border: none;
  border-radius: 50px;
  cursor: pointer;
  font-size: 1.1rem;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  box-shadow: 0 0 15px #0ff, 0 0 30px #08f inset;
}

button:hover {
  transform: scale(1.05);
  box-shadow: 0 0 25px #0ff, 0 0 50px #08f inset;
}
</style>