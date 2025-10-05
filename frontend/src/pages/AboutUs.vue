<!-- src/pages/AboutPage.vue -->
<template>
  <section class="about" aria-labelledby="about-title">
    <!-- Cosmic background -->
    <div class="bg" aria-hidden="true">
      <div class="nebula"></div>
      <div class="stars layer-1"></div>
      <div class="stars layer-2"></div>
      <div class="gridlines"></div>
      <div class="comet" aria-hidden="true"></div>
    </div>

    <div class="container">
      <!-- Hero -->
      <header class="hero">
        <h1 id="about-title" class="title">
          About <span class="accent">SpaceMonkeys</span>
        </h1>
        <p class="lead">
          Six curious SEM students from the University of Gothenburg chasing exoplanets with AI—
          building tools that help researchers plug in KOI data and predict planet candidates.
        </p>
      </header>

      <!-- Intro block -->
      <section class="intro fade-in" style="--i:0">
        <figure class="logo-wrap">
          <img src="../assets/space-monkeys-logo.png" alt="SpaceMonkeys logo" class="logo" />
        </figure>

        <div class="text-wrap">
          <p class="text">
            SpaceMonkeys consists of 6 curious students studying SEM (Software Engineering and Management)
            at the University of Gothenburg. We aim to maximize Galactic Impact by providing a tool that
            researchers can use to plug in KOI data and predict whether objects of interest are exoplanets—or not.
            This is powered by our flagship AI model trained on cutting-edge NASA data.
          </p>
        </div>
      </section>

      <!-- Team -->
      <section class="team fade-in" style="--i:1" aria-label="Team members">
        <h2 class="section-title">Meet the crew</h2>
        <ul class="team-grid">
          <li class="member">
            <div class="card">
              <h3 class="name">Nadman</h3>
              <p class="role">Chief of Vibes</p>
            </div>
          </li>
          <li class="member">
            <div class="card">
              <h3 class="name">Filipe</h3>
              <p class="role">Junior Viber</p>
            </div>
          </li>
          <li class="member">
            <div class="card">
              <h3 class="name">Vaibhav</h3>
              <p class="role">Senior Vibe Assistant</p>
            </div>
          </li>
          <li class="member">
            <div class="card">
              <h3 class="name">Taha</h3>
              <p class="role">Associate Vibe Partner</p>
            </div>
          </li>
          <li class="member">
            <div class="card">
              <h3 class="name">Erik</h3>
              <p class="role">Expert Vibe Accelerator</p>
            </div>
          </li>
          <li class="member">
            <div class="card">
              <h3 class="name">Zahid</h3>
              <p class="role">Technological Vibe Officer</p>
            </div>
          </li>
        </ul>
      </section>

      <!-- Group photo -->
      <section class="group fade-in" style="--i:2">
        <figure class="photo-card">
          <img src="../assets/group-pic.jpg" alt="SpaceMonkeys group photo" />
          <figcaption>SpaceMonkeys, on a mission to find distant worlds.</figcaption>
        </figure>
      </section>
    </div>
  </section>
</template>

<script setup>
// Pure CSS animations; no JS needed.
</script>

<style scoped>
/* ---------- Theme ---------- */
.about {
  --bg: #0b1020;
  --text: #e5e7eb;
  --muted: #9ca3af;
  --border: rgba(255,255,255,0.10);
  --card: rgba(255,255,255,0.06);
  --card-2: rgba(255,255,255,0.04);
  --primary-1: #22d3ee;
  --primary-2: #6366f1;
  --shadow: 0 10px 30px rgba(0,0,0,0.45);

  position: relative;
  color: var(--text);
  background: radial-gradient(1200px 800px at 70% 20%, #151a33, var(--bg)) fixed;
  min-height: 100svh;
  overflow: clip;
}

/* ---------- Background FX ---------- */
.bg { position: absolute; inset: 0; pointer-events: none; z-index: 0; }

.nebula {
  position: absolute; inset: -20%;
  background:
    radial-gradient(900px 700px at 20% 30%, rgba(99,102,241,0.18), transparent 60%),
    radial-gradient(800px 600px at 80% 20%, rgba(34,211,238,0.15), transparent 60%),
    radial-gradient(700px 900px at 70% 80%, rgba(147,197,253,0.08), transparent 70%);
  filter: blur(10px) saturate(120%);
  animation: drift 42s linear infinite alternate;
}

.stars {
  position: absolute; inset: -100% -100% 0 -100%;
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
  animation: starScroll 90s linear infinite;
}
.stars.layer-2 {
  background-image:
    radial-gradient(1px 1px at 30% 60%, rgba(255,255,255,0.6) 99%, transparent),
    radial-gradient(1.5px 1.5px at 70% 40%, rgba(255,255,255,0.5) 99%, transparent),
    radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.4) 99%, transparent);
  animation: starScroll 150s linear infinite reverse; opacity: 0.5;
}

/* Holographic grid */
.gridlines {
  position: absolute; inset: 0;
  background:
    linear-gradient(transparent 95%, rgba(99,102,241,0.18) 96%) 0 0 / 100% 32px,
    linear-gradient(90deg, transparent 95%, rgba(34,211,238,0.12) 96%) 0 0 / 32px 100%;
  mask: linear-gradient(#000, transparent 70%);
  opacity: .25;
  transform: perspective(800px) rotateX(55deg) translateY(35%);
  transform-origin: top center;
  filter: blur(0.2px);
}

/* Comet sweep */
.comet {
  position: absolute;
  width: 2px; height: 2px; border-radius: 50%;
  background: #fff;
  box-shadow: 0 0 12px rgba(255,255,255,0.9), 0 0 30px rgba(99,102,241,0.6);
  top: -40px; left: -60px;
  animation: comet 12s cubic-bezier(.32,.01,.2,1) infinite;
  opacity: .9;
}
.comet::after {
  content: "";
  position: absolute; right: 2px; top: 0;
  width: 140px; height: 2px; border-radius: 2px;
  background: linear-gradient(90deg, rgba(255,255,255,0.85), rgba(255,255,255,0));
  transform: translateX(-100%);
}

/* ---------- Layout ---------- */
.container {
  position: relative; z-index: 1;
  max-width: 1100px; margin: 0 auto;
  padding: clamp(24px, 4vw, 48px);
}

.hero { text-align: center; margin-top: clamp(12px, 4vh, 24px); }
.title {
  margin: 0 0 8px;
  font-weight: 900;
  letter-spacing: -0.02em;
  font-size: clamp(32px, 6vw, 56px);
  background: linear-gradient(135deg, var(--primary-1), var(--primary-2));
  background-clip: text; -webkit-background-clip: text;
  color: transparent; -webkit-text-fill-color: transparent;
  text-shadow: 0 0 30px rgba(99,102,241,0.18);
}
.title .accent { -webkit-text-fill-color: currentColor; color: var(--text); opacity: 0.9; }
.lead { color: var(--muted); max-width: 820px; margin: 0 auto; }

/* Intro */
.intro {
  display: grid;
  grid-template-columns: 240px 1fr;
  gap: clamp(16px, 3vw, 28px);
  align-items: center;
  margin-top: clamp(24px, 6vh, 40px);
}
.logo-wrap {
  background: linear-gradient(180deg, var(--card), var(--card-2));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  box-shadow: var(--shadow);
  display: grid; place-items: center;
  animation: float 9s ease-in-out infinite;
}
.logo { width: 100%; height: auto; max-width: 200px; display: block; }
.text-wrap .text { color: var(--text); opacity: .95; line-height: 1.7; }

/* Team */
.team { margin-top: clamp(28px, 6vh, 56px); }
.section-title {
  font-size: clamp(20px, 3vw, 24px);
  margin: 0 0 12px;
  opacity: .9;
}
.team-grid {
  list-style: none; padding: 0; margin: 0;
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.member .card {
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
  will-change: transform;
}
.member .card:hover { transform: translateY(-2px); }
.name { margin: 0; font-size: 18px; }
.role { margin: 4px 0 0; color: var(--muted); font-size: 14px; }

/* Group photo */
.group { margin-top: clamp(28px, 6vh, 56px); }
.photo-card {
  position: relative;
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  background: linear-gradient(180deg, var(--card), var(--card-2));
}
.photo-card img { width: 100%; height: auto; display: block; }
.photo-card figcaption {
  position: absolute; left: 0; right: 0; bottom: 0;
  padding: 10px 14px;
  background: linear-gradient(180deg, transparent, rgba(0,0,0,0.55));
  color: #fff; font-size: 14px;
}

/* ---------- Animations ---------- */
.fade-in { animation: fadeUp .7s ease both; animation-delay: calc(var(--i, 0) * 90ms); }
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes drift { from { transform: translate3d(-2%, -1%, 0) scale(1.02); } to { transform: translate3d(1%, 2%, 0) scale(1.04); } }
@keyframes starScroll { from { transform: translateY(0); } to { transform: translateY(600px); } }
@keyframes float { 0%,100% { transform: translateY(0) rotateZ(0.2deg); } 50% { transform: translateY(-8px) rotateZ(-0.5deg); } }
@keyframes comet { 
  0% { transform: translate(-5vw, -5vh); opacity: 0; }
  5% { opacity: .9; }
  50% { transform: translate(110vw, 60vh); opacity: .9; }
  60% { opacity: 0; }
  100% { transform: translate(110vw, 60vh); opacity: 0; }
}

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
  .intro { grid-template-columns: 1fr; }
  .logo-wrap { order: -1; max-width: 320px; margin: 0 auto; }
  .lead { padding: 0 4px; }
}

@media (max-width: 560px) {
  .container { padding: 20px; }
  .photo-card figcaption { font-size: 13px; }
}

/* ---------- Reduced Motion ---------- */
@media (prefers-reduced-motion: reduce) {
  .nebula, .stars.layer-1, .stars.layer-2, .fade-in, .logo-wrap, .comet { animation: none !important; }
}
</style>
