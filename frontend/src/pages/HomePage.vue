
<!-- src/pages/HomePage.vue -->
<template>
  <section class="home" aria-labelledby="title">
    <!-- Animated cosmic background -->
    <div class="bg">
      <div class="nebula"></div>
      <div class="stars layer-1" aria-hidden="true"></div>
      <div class="stars layer-2" aria-hidden="true"></div>
      <div class="gridlines" aria-hidden="true"></div>
    </div>

    <!-- Content -->
    <div class="container">
      <div class="hero">
        <!-- Planet illustration -->
        <div class="planet-wrap" aria-hidden="true">
          <svg class="planet" viewBox="0 0 200 200" role="img" aria-label="Stylized exoplanet with ring">
            <defs>
              <radialGradient id="planetGrad" cx="35%" cy="35%" r="70%">
                <stop offset="0%" stop-color="#9AE6FF"/>
                <stop offset="55%" stop-color="#6C9BFF"/>
                <stop offset="100%" stop-color="#2E2F75"/>
              </radialGradient>
              <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#22d3ee" stop-opacity="0.9"/>
                <stop offset="100%" stop-color="#6366f1" stop-opacity="0.9"/>
              </linearGradient>
              <filter id="softGlow">
                <feGaussianBlur stdDeviation="6" result="blur"/>
                <feMerge>
                  <feMergeNode in="blur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            <!-- glow -->
            <circle cx="100" cy="100" r="64" fill="url(#planetGrad)" filter="url(#softGlow)"/>
            <!-- soft terminator -->
            <ellipse cx="80" cy="85" rx="46" ry="30" fill="#ffffff" opacity="0.08"/>
            <!-- ring -->
            <g class="ring">
              <ellipse cx="100" cy="110" rx="92" ry="26" fill="url(#ringGrad)" opacity="0.9"/>
              <ellipse cx="100" cy="110" rx="92" ry="26" fill="url(#ringGrad)" opacity="0.25" filter="url(#softGlow)"/>
            </g>
            <!-- planet mask to pass ring behind the sphere -->
            <circle class="ring-mask" cx="100" cy="100" r="64" fill="url(#planetGrad)"/>
          </svg>
        </div>

        <header class="copy">
          <h1 id="title" class="title">
            Exoplanet Hunter <span class="ai-chip">AI</span>
          </h1>
          <p class="muted lead">
            Train models on stellar light curves, spot candidate exoplanets, and evaluate predictions—all in one place.
          </p>
          <div class="cta">
            <router-link class="btn primary" to="/upload" aria-label="Go to Uploads to train the model">
              🚀 Go to Uploads
            </router-link>
            <a class="btn ghost" href="#how-it-works">
              Learn more
            </a>
          </div>
          <p class="features-label">Features:</p>
          <ul class="pills" aria-label="Key capabilities">
            <li>Transit Detection</li>
            <li>Model Training</li>
            <li>Prediction Review</li>
          </ul>
        </header>
      </div>

      <!-- How it works (quick explainer) -->
      <section id="how-it-works" class="how">
        <div class="card">
          <h2>How it works</h2>
          <ol>
            <li><strong>Upload</strong> light curves (CSV, FITS, etc.)</li>
            <li><strong>Train</strong> the AI to learn transit signatures</li>
            <li><strong>Validate</strong> predictions with rich metrics &amp; charts</li>
          </ol>
        </div>
        <div class="card">
          <h2>Built for discovery</h2>
          <p>
            We combine signal processing with deep learning to surface high-confidence candidates—
            faster than traditional pipelines.
          </p>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup>
// No JavaScript required for the core experience.
// Motion/animation is CSS-based and honors prefers-reduced-motion.
</script>

<style scoped>
/* ---------- CSS Variables ---------- */
:root {
  --bg: #0b1020;
  --bg-deep: #070a16;
  --text: #e5e7eb;
  --muted: #99a1b3;
  --primary-1: #22d3ee;
  --primary-2: #6366f1;
  --glass: rgba(255, 255, 255, 0.06);
  --border: rgba(255, 255, 255, 0.08);
  --shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
}

/* ---------- Layout ---------- */
.home {
  position: relative;
  min-height: 100vhs;
  width: 100%;
  color: var(--text);
  background: radial-gradient(1200px 800px at 70% 20%, #151a33, var(--bg)) fixed;
  overflow: clip;
}

.container {
  max-width: 100%;
  margin: 0 auto;
  padding: clamp(24px, 4vw, 48px);
}

.hero {
  display: grid;
  grid-template-columns: 1.1fr 1fr;
  align-items: center;
  gap: clamp(20px, 5vw, 56px);
  margin-top: clamp(40px, 8vh, 120px);
}

@media (max-width: 940px) {
  .hero { grid-template-columns: 1fr; }
  .planet-wrap { order: -1; }
}

/* ---------- Background FX ---------- */
.bg {
  position: absolute;
  inset: 0;
  overflow: hidden;
  pointer-events: none;
  z-index: 0;
}

.nebula {
  position: absolute;
  inset: -20%;
  background:
    radial-gradient(800px 600px at 20% 30%, rgba(99,102,241,0.18), transparent 60%),
    radial-gradient(700px 500px at 80% 20%, rgba(34,211,238,0.15), transparent 60%),
    radial-gradient(600px 800px at 70% 80%, rgba(147,197,253,0.08), transparent 70%);
  filter: blur(10px) saturate(120%);
  animation: drift 40s linear infinite alternate;
}

.stars {
  position: absolute;
  inset: -100% -100% 0 -100%;
  background-repeat: repeat;
  background-size: 600px 600px;
  opacity: 0.7;
  mix-blend-mode: screen;
}

.stars.layer-1 {
  background-image:
    radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.9) 99%, transparent),
    radial-gradient(1.5px 1.5px at 60% 70%, rgba(255,255,255,0.8) 99%, transparent),
    radial-gradient(1px 1px at 80% 20%, rgba(255,255,255,0.6) 99%, transparent);
  animation: starScroll 80s linear infinite;
}

.stars.layer-2 {
  background-image:
    radial-gradient(1px 1px at 30% 60%, rgba(255,255,255,0.6) 99%, transparent),
    radial-gradient(1.5px 1.5px at 70% 40%, rgba(255,255,255,0.5) 99%, transparent),
    radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.4) 99%, transparent);
  animation: starScroll 140s linear infinite reverse;
  opacity: 0.5;
}

/* Subtle holographic grid to give an "AI" vibe */
.gridlines {
  position: absolute;
  inset: 0;
  background:
    linear-gradient(transparent 95%, rgba(99,102,241,0.18) 96%) 0 0 / 100% 32px,
    linear-gradient(90deg, transparent 95%, rgba(34,211,238,0.12) 96%) 0 0 / 32px 100%;
  mask: linear-gradient(#000, transparent 70%);
  opacity: 0.25;
  transform: perspective(800px) rotateX(55deg) translateY(35%);
  transform-origin: top center;
  filter: blur(0.2px);
}

/* ---------- Planet ---------- */
.planet-wrap {
  position: relative;
  width: min(460px, 80vw);
  margin-inline: auto;
  aspect-ratio: 1 / 1;
  transform-style: preserve-3d;
  animation: float 8s ease-in-out infinite;
  will-change: transform;
  z-index: 1;
}

.planet {
  width: 100%;
  height: auto;
  display: block;
  filter: drop-shadow(0 40px 60px rgba(0,0,0,0.55));
}

.planet .ring {
  transform-origin: 100px 110px;
  animation: wobble 16s ease-in-out infinite;
}

.planet .ring-mask {
  /* mask the back of the ring for depth illusion */
  mix-blend-mode: multiply;
}

/* ---------- Typography ---------- */
.title {
  font-size: clamp(32px, 6vw, 64px);
  line-height: 1.05;
  letter-spacing: -0.02em;
  margin: 0 0 12px;

  background: linear-gradient(135deg, var(--primary-1), var(--primary-2));
  background-clip: text;
  -webkit-background-clip: text;       /* Safari/Chrome */
  text-shadow: 0 0 30px #6366f1b0;
}

/* give the chip its own text color so it doesn’t inherit transparent */
.ai-chip {
  display: inline-block;
  font-size: 0.55em;
  padding: 0.18em 0.45em;
  border-radius: 999px;
  margin-left: 0.25em;
  background: linear-gradient(135deg, rgba(34,211,238,0.25), rgba(99,102,241,0.25));
  border: 1px solid var(--border);
  vertical-align: 0.1em;
  backdrop-filter: blur(6px);

  color: #e5e7eb;                   /* visible text */
  -webkit-text-fill-color: #e5e7eb; /* Safari */
  font-weight: 700;                  /* optional pop */
}

.lead {
  font-size: clamp(16px, 2.2vw, 20px);
  margin: 8px 0 22px;
  color: var(--muted);
}

/* ---------- Buttons & Pills ---------- */
.cta {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 44px;
  padding: 0 16px;
  border-radius: 14px;
  text-decoration: none;
  font-weight: 600;
  border: 1px solid var(--border);
  background: var(--glass);
  box-shadow: var(--shadow);
  transition: transform 0.2s ease, opacity 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
  will-change: transform;
}

.btn:hover { transform: translateY(-2px); }
.btn:active { transform: translateY(0); }

.btn.primary {
  background: linear-gradient(135deg, var(--primary-1), var(--primary-2));
  color: #fff;
  border: none;
}

.btn.ghost {
  color: var(--text);
  backdrop-filter: blur(6px);
}

.features-label {
  margin: 6px 0 2px;
  font-weight: 700;
  letter-spacing: .02em;
  color: var(--text);
  opacity: .9;
}

.pills {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  list-style: none;
  padding: 0;
  margin: 10px 0 0;
}
.pills li {
  font-size: 12px;
  letter-spacing: 0.4px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.04);
}

/* ---------- Cards Section ---------- */
.how {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: clamp(16px, 3vw, 24px);
  margin-top: clamp(40px, 8vh, 80px);
}

@media (max-width: 780px) {
  .how { grid-template-columns: 1fr; }
}

.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: clamp(16px, 2.4vw, 22px);
  box-shadow: var(--shadow);
  backdrop-filter: blur(8px);
}

.card h2 {
  margin: 0 0 10px;
  font-size: clamp(18px, 2.5vw, 22px);
}

.card p, .card li { color: var(--muted); }
.card ol {
  margin: 0;
  padding-left: 18px;
}

/* ---------- Animations ---------- */
@keyframes float {
  0%, 100% { transform: translateY(0) rotateZ(0.2deg); }
  50% { transform: translateY(-10px) rotateZ(-0.6deg); }
}
@keyframes wobble {
  0%, 100% { transform: rotateX(42deg) rotateZ(-10deg); }
  50% { transform: rotateX(48deg) rotateZ(-14deg); }
}
@keyframes drift {
  from { transform: translate3d(-2%, -1%, 0) scale(1.02); }
  to   { transform: translate3d(1%, 2%, 0) scale(1.04); }
}
@keyframes starScroll {
  from { transform: translateY(0); }
  to   { transform: translateY(600px); }
}

/* ---------- Accessibility: Reduced Motion ---------- */
@media (prefers-reduced-motion: reduce) {
  .nebula,
  .stars.layer-1,
  .stars.layer-2,
  .planet-wrap,
  .planet .ring,
  .btn { animation: none !important; transition: none !important; }
}

/* ---------- Small niceties ---------- */
.copy { position: relative; z-index: 2; }
.muted { color: var(--muted); }
</style>
