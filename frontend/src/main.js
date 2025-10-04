import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'

const app = createApp(App)

// Disable Vue DevTools
app.config.devtools = false

app.mount('#app')
