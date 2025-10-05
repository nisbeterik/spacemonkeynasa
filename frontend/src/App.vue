<!-- src/App.vue -->
<template>
  <div class="main-container">
    <nav class="nav">
      <router-link to="/" class="brand">SpaceMonkey</router-link>
      <div class="links">
        <router-link to="/upload">Upload</router-link>
      </div>
    </nav>
    <router-view />
  </div>
</template>

<script setup>
</script>

<style scoped>
.main-container {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  width: 100%;
}
.nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 16px; background: #0b1220; border-bottom: 1px solid #1f2937;
}
.brand { color: #22d3ee; font-weight: 800; text-decoration: none; }
.links a { color: #e5e7eb; text-decoration: none; margin-left: 12px; }
.links a.router-link-active { color: #22d3ee; }
</style>
