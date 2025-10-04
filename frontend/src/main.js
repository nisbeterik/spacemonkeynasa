import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import { createRouter, createWebHistory } from 'vue-router'

// import your pages
import HomePage from './pages/HomePage.vue'
import UploadPage from './pages/UploadPage.vue'

// define routes
const routes = [
  { path: '/', component: HomePage },
  { path: '/upload', component: UploadPage }
]

// create router
const router = createRouter({
  history: createWebHistory(),
  routes,
})

const app = createApp(App)

// Disable Vue DevTools
app.config.devtools = false

// use router
app.use(router)

app.mount('#app')
