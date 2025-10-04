<script setup>
function handleGetStarted() {
  alert("Let's start exploring exoplanets!"); // replace with actual routing later
}
</script>

<template>
  <div class="space-page">
    <img alt="SpaceMonkey logo" class="logo" src="./assets/logo.svg" width="125" height="125" />
    <div class="wrapper">
      <h1>Welcome to SpaceMonkey Exoplanet Hunter</h1>
      <p>Discover new worlds beyond our solar system using AI-powered tools!</p>
      <button @click="handleGetStarted">Get Started</button>
    </div>
  </div>
</template>

<style scoped>
/* Force full viewport regardless of parent layout */
html, body, #app {
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
}

/* Fullscreen space-themed page */
.space-page {
  position: fixed;   /* fill the viewport and ignore parent layout */
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center; /* vertical center */
  align-items: center;     /* horizontal center */
  background: radial-gradient(circle at top, #0b0c29, #1c1c3c, #000);
  color: #fff;
  text-align: center;
  padding: 2rem;
  box-shadow: 0 0 50px rgba(0,0,0,0.8) inset;
  z-index: 9999; /* stay on top if needed */
}

/* Logo styling */
.logo {
  margin-bottom: 2rem;
  filter: drop-shadow(0 0 10px #0ff);
}

/* Text styling */
.wrapper h1 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  text-shadow: 0 0 10px #0ff, 0 0 20px #08f;
}

.wrapper p {
  font-size: 1.3rem;
  margin-bottom: 2rem;
  text-shadow: 0 0 5px #0ff;
}

/* Button styling */
button {
  background: linear-gradient(45deg, #0ff, #08f);
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
