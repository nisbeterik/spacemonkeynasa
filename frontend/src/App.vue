<!-- src/App.vue -->
<template>
  <div class="main-container">
    <nav class="nav">
      <router-link to="/" class="brand">
        <img :src="logoUrl" alt="Space Monkeys logo" class="logo" width="36" height="36" />
        <span>Space Monkeys</span>
      </router-link>

      <div class="links">
        <router-link to="/upload">Find Exoplanet</router-link>
      </div>
    </nav>

    <router-view />
  </div>
</template>

<script setup>
import logoUrl from '@/assets/space-monkeys-logo.png'
</script>

<style scoped>
.main-container {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  background: #0b1220;
  border-bottom: 1px solid #1f2937;
  width: 100%;
}

/* Brand + logo */
.brand {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: #22d3ee;
  font-weight: 800;
  text-decoration: none;
  white-space: nowrap;
}

.logo {
  height: 32px;   /* tweak to taste (24–40px usually looks good) */
  width: auto;
  display: block;
}

/* Links */
.links a {
  color: #e5e7eb;
  text-decoration: none;
  margin-left: 12px;
}
.links a.router-link-active { color: #22d3ee; }
</style>
