<!-- src/pages/HomePage.vue -->
<template>
  <div class="home">
    <h1>Welcome 👋</h1>
    <p class="muted">Use the Upload page to train the model, check exo planets, and evaluate predictions.</p>
    <router-link class="btn" to="/upload">Go to Uploads</router-link>
  </div>
</template>

<script setup>
</script>

<style scoped>
.home { max-width: 760px; margin: 80px auto; text-align: center; }
.muted { color: #6b7280; margin-bottom: 24px; }
.btn {
  background: linear-gradient(135deg, #22d3ee, #6366f1);
  color: #fff; padding: 12px 18px; border-radius: 12px; text-decoration: none;
}
</style>
