// src/main.js
import { createApp } from 'vue';
import { createRouter, createWebHistory } from 'vue-router';

import App from './App.vue';
import HomePage from './pages/HomePage.vue';
import UploadPage from './pages/UploadPage.vue';
import AboutUs from './pages/AboutUs.vue'

import './assets/base.css';
import './assets/main.css';

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: HomePage },
    { path: '/upload', component: UploadPage },
    { path: '/about', component: AboutUs },

  ],
});

createApp(App).use(router).mount('#app');
