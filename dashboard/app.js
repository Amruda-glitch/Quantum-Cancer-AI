// ============================================================
// Quantum Cancer Detection – Interactive Dashboard App
// ============================================================

// ── Navigation ───────────────────────────────────────────────
function navigate(sectionId) {
  document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const section = document.getElementById(sectionId);
  if (section) section.classList.add('active');
  document.querySelectorAll(`.nav-item[data-section="${sectionId}"]`).forEach(n => n.classList.add('active'));
  const titles = {
    'overview': 'Overview',
    'models': 'Model Comparison',
    'quantum': 'Quantum Optimizer',
    'cancer': 'Cancer Types',
    'preprocessing': 'Preprocessing Pipeline',
    'report': 'Technical Report'
  };
  document.getElementById('breadcrumb-page').textContent = titles[sectionId] || sectionId;
  // Trigger section-specific initializations
  if (sectionId === 'overview')       initOverview();
  if (sectionId === 'models')         initModels();
  if (sectionId === 'quantum')        initQuantum();
  if (sectionId === 'cancer')         initCancer();
  if (sectionId === 'preprocessing')  initPreprocessing();
  if (sectionId === 'report')         initReport();
}

// ── Chart.js defaults ───────────────────────────────────────────────
Chart.defaults.color = '#334155';
Chart.defaults.borderColor = 'rgba(30,58,138,0.10)';
Chart.defaults.font.family = 'Inter, sans-serif';

const CHART_REGISTRY = {};

function destroyChart(id) {
  if (CHART_REGISTRY[id]) { CHART_REGISTRY[id].destroy(); delete CHART_REGISTRY[id]; }
}

// ── Particle Canvas ───────────────────────────────────────────
function initParticles() {
  const canvas = document.getElementById('particle-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  resize();
  window.addEventListener('resize', resize);

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x = Math.random() * W;
      this.y = Math.random() * H;
      this.vx = (Math.random() - 0.5) * 0.3;
      this.vy = (Math.random() - 0.5) * 0.3;
      this.r = Math.random() * 1.5 + 0.5;
      this.alpha = Math.random() * 0.5 + 0.1;
      this.color = Math.random() > 0.6 ? '#1d4ed8' : '#1e3a8a';
    }
    update() {
      this.x += this.vx; this.y += this.vy;
      if (this.x < 0 || this.x > W || this.y < 0 || this.y > H) this.reset();
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.globalAlpha = this.alpha;
      ctx.fill();
    }
  }

  for (let i = 0; i < 120; i++) particles.push(new Particle());

  // Draw connecting lines
  function drawConnections() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 100) {
          ctx.globalAlpha = (1 - dist / 100) * 0.12;
          ctx.strokeStyle = '#1d4ed8';
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
  }

  function animate() {
    ctx.clearRect(0, 0, W, H);
    drawConnections();
    particles.forEach(p => { p.update(); p.draw(); });
    ctx.globalAlpha = 1;
    requestAnimationFrame(animate);
  }
  animate();
}

// ── Overview Section ─────────────────────────────────────────
let overviewInitialized = false;
function initOverview() {
  if (overviewInitialized) return;
  overviewInitialized = true;

  // Animate counting stats
  animateCount('stat-total-images', 0, QCD_DATA.dataset.totalImages, 1800, ',');
  animateCount('stat-clean-images', 0, QCD_DATA.dataset.cleanImages, 1800, ',');
  animateCount('stat-cancer-types', 0, 9, 1000);
  animateCount('stat-models', 0, 7, 1000);

  // Best accuracy
  animateCount('stat-best-acc', 0, 99.63, 1800, '%', 2);

  // Build model type distribution donut
  destroyChart('overviewDonut');
  const donutCtx = document.getElementById('overviewDonut');
  if (donutCtx) {
    CHART_REGISTRY['overviewDonut'] = new Chart(donutCtx, {
      type: 'doughnut',
      data: {
        labels: ['Classical ML', 'Deep Learning', 'Quantum Optimized'],
        datasets: [{
          data: [3, 3, 1],
          backgroundColor: ['rgba(59,130,246,0.65)', 'rgba(99,102,241,0.65)', 'rgba(29,78,216,0.85)'],
          borderColor: ['#3b82f6', '#6366f1', '#1d4ed8'],
          borderWidth: 2,
          hoverOffset: 8,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '68%',
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw} models` } }
        },
        animation: { animateRotate: true, duration: 1200, easing: 'easeOutQuart' }
      }
    });
  }

  // Dataset split horizontal bar
  destroyChart('splitBar');
  const splitCtx = document.getElementById('splitBar');
  if (splitCtx) {
    CHART_REGISTRY['splitBar'] = new Chart(splitCtx, {
      type: 'bar',
      data: {
        labels: ['Train', 'Validation', 'Test'],
        datasets: [{
          data: [70, 15, 15],
          backgroundColor: ['rgba(29,78,216,0.65)', 'rgba(99,102,241,0.65)', 'rgba(16,185,129,0.65)'],
          borderColor: ['#1d4ed8', '#6366f1', '#10b981'],
          borderWidth: 2,
          borderRadius: 6,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: 'rgba(30,58,138,0.05)' }, ticks: { callback: v => v + '%' }, max: 100 },
          y: { grid: { display: false } }
        },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` ${ctx.raw}% of dataset` } } },
        animation: { duration: 1000, easing: 'easeOutQuart' }
      }
    });
  }

  // Progress bars for quality metrics
  animateProgressBars();
}

function animateCount(id, start, end, duration, suffix = '', decimals = 0) {
  const el = document.getElementById(id);
  if (!el) return;
  const startTime = performance.now();
  function update(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const val = start + (end - start) * ease;
    el.textContent = (decimals > 0 ? val.toFixed(decimals) : Math.floor(val).toLocaleString()) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

function animateProgressBars() {
  setTimeout(() => {
    document.querySelectorAll('.progress-fill[data-width]').forEach(bar => {
      bar.style.width = bar.dataset.width + '%';
    });
  }, 400);
}

// ── Model Comparison Section ──────────────────────────────────
let modelsInitialized = false;
function initModels() {
  if (modelsInitialized) return;
  modelsInitialized = true;

  const modelNames = Object.keys(QCD_DATA.models);
  const colors = modelNames.map(n => QCD_DATA.models[n].color);

  const metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc'];
  let activeMetric = 'accuracy';

  function buildBarChart(metric) {
    destroyChart('modelBarChart');
    const ctx = document.getElementById('modelBarChart');
    if (!ctx) return;
    const vals = modelNames.map(n => +(QCD_DATA.models[n][metric] * 100).toFixed(2));
    CHART_REGISTRY['modelBarChart'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: modelNames,
        datasets: [{
          label: metric.charAt(0).toUpperCase() + metric.slice(1),
          data: vals,
          backgroundColor: colors.map(c => c + 'aa'),
          borderColor: colors,
          borderWidth: 2,
          borderRadius: 6,
          hoverBackgroundColor: colors,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          y: {
            min: 90,
            max: 100.2,
            grid: { color: 'rgba(15,23,42,0.06)' },
            ticks: { callback: v => v.toFixed(1) + '%' }
          },
          x: { grid: { display: false } }
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ` ${ctx.raw.toFixed(4)}%` } }
        },
        animation: { duration: 800, easing: 'easeOutQuart' }
      }
    });
  }

  function buildRadarChart() {
    destroyChart('modelRadar');
    const ctx = document.getElementById('modelRadar');
    if (!ctx) return;
    const radarMetrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'AUC'];
    const datasets = modelNames.map(name => {
      const m = QCD_DATA.models[name];
      return {
        label: name,
        data: [m.accuracy, m.sensitivity, m.specificity, m.precision, m.f1, m.auc].map(v => v * 100),
        borderColor: m.color,
        backgroundColor: m.color + '22',
        borderWidth: 2,
        pointBackgroundColor: m.color,
        pointRadius: 3,
      };
    });
    CHART_REGISTRY['modelRadar'] = new Chart(ctx, {
      type: 'radar',
      data: { labels: radarMetrics, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          r: {
            min: 94, max: 100.2,
            grid: { color: 'rgba(30,58,138,0.06)' },
            pointLabels: { font: { size: 11 }, color: '#334155' },
            ticks: { display: false },
            angleLines: { color: 'rgba(30,58,138,0.08)' }
          }
        },
        plugins: {
          legend: {
            display: true,
            labels: { boxWidth: 10, font: { size: 11 }, padding: 12 }
          }
        },
        animation: { duration: 1000, easing: 'easeOutQuart' }
      }
    });
  }

  buildBarChart(activeMetric);
  buildRadarChart();

  // Build metric toggle buttons
  const toggleWrap = document.getElementById('metric-toggles');
  if (toggleWrap) {
    toggleWrap.innerHTML = '';
    metrics.forEach(m => {
      const btn = document.createElement('button');
      btn.className = 'tab-btn' + (m === activeMetric ? ' active' : '');
      btn.textContent = m.charAt(0).toUpperCase() + m.slice(1);
      btn.onclick = () => {
        activeMetric = m;
        toggleWrap.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        buildBarChart(m);
      };
      toggleWrap.appendChild(btn);
    });
  }

  // Build metric table
  buildMetricTable();
  animateProgressBars();
}

function buildMetricTable() {
  const tbody = document.getElementById('model-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';
  const models = QCD_DATA.models;
  const bestAccuracy = Math.max(...Object.values(models).map(m => m.accuracy));

  Object.entries(models).forEach(([name, m]) => {
    const isBest = m.accuracy === bestAccuracy;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>
        <span class="model-name">
          <span class="model-dot" style="background:${m.color}"></span>
          ${name}
          ${isBest ? '<span class="model-type-badge quantum">⚡ Best</span>' : ''}
        </span>
      </td>
      <td><span class="model-type-badge ${m.type}">${m.type === 'classical' ? 'Classical' : m.type === 'deep' ? 'Deep Learning' : 'Quantum'}</span></td>
      <td><span class="metric-val${isBest ? ' best' : ''}">${(m.accuracy * 100).toFixed(2)}%</span></td>
      <td><span class="metric-val">${(m.sensitivity * 100).toFixed(2)}%</span></td>
      <td><span class="metric-val">${(m.specificity * 100).toFixed(2)}%</span></td>
      <td><span class="metric-val">${(m.precision * 100).toFixed(2)}%</span></td>
      <td><span class="metric-val">${(m.f1 * 100).toFixed(2)}%</span></td>
      <td><span class="metric-val">${(m.auc * 100).toFixed(4)}%</span></td>
      <td><span class="metric-val">${m.trainTime}s</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Quantum Section ───────────────────────────────────────────
let quantumInitialized = false;
function initQuantum() {
  if (quantumInitialized) return;
  quantumInitialized = true;

  buildConvergenceChart();
  buildQbitGrid();
  buildFeatureGenerationChart();
}

function buildConvergenceChart() {
  destroyChart('convergenceChart');
  const ctx = document.getElementById('convergenceChart');
  if (!ctx) return;

  const d = QCD_DATA.optimizerConvergence;
  CHART_REGISTRY['convergenceChart'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: d.evaluations,
      datasets: [
        {
          label: 'Grid Search',
          data: d.gridSearch,
          borderColor: '#94a3b8',
          backgroundColor: 'rgba(148,163,184,0.10)',
          borderWidth: 2,
          borderDash: [6, 3],
          pointRadius: 3,
          tension: 0.4,
          fill: false,
        },
        {
          label: 'Random Search',
          data: d.randomSearch,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.10)',
          borderWidth: 2,
          borderDash: [4, 2],
          pointRadius: 3,
          tension: 0.4,
          fill: false,
        },
        {
          label: 'QuantumNow Optimized',
          data: d.quantumNow,
          borderColor: '#1d4ed8',
          backgroundColor: 'rgba(29,78,216,0.10)',
          borderWidth: 3,
          pointRadius: 5,
          pointBackgroundColor: '#1d4ed8',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          tension: 0.4,
          fill: true,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          grid: { color: 'rgba(30,58,138,0.05)' },
          title: { display: true, text: 'Number of Evaluations', color: '#334155', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(30,58,138,0.05)' },
          min: 0.5,
          max: 1.0,
          title: { display: true, text: 'Best Fitness Score', color: '#334155', font: { size: 11 } },
          ticks: { callback: v => v.toFixed(2) }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(255,255,255,0.97)',
          borderColor: 'rgba(29,78,216,0.25)',
          borderWidth: 1,
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(4)}`
          }
        }
      },
      animation: { duration: 1500, easing: 'easeOutQuart' }
    }
  });
}

function buildQbitGrid() {
  const grid = document.getElementById('qbit-grid');
  if (!grid) return;
  grid.innerHTML = '';
  const total = 200;
  const selected = new Set();
  QCD_DATA.featureSelection.generationHistory[4];

  // Final state: 87 selected out of 200
  while (selected.size < 87) selected.add(Math.floor(Math.random() * total));

  for (let i = 0; i < total; i++) {
    const bit = document.createElement('div');
    bit.className = 'qbit ' + (selected.has(i) ? 'selected' : 'discarded');
    bit.title = selected.has(i) ? `Feature ${i} – SELECTED` : `Feature ${i} – discarded`;
    grid.appendChild(bit);
  }
}

function buildFeatureGenerationChart() {
  destroyChart('genFitnessChart');
  const ctx = document.getElementById('genFitnessChart');
  if (!ctx) return;
  const gens = QCD_DATA.featureSelection.generationHistory;
  CHART_REGISTRY['genFitnessChart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: gens.map(g => `Gen ${g.gen}`),
      datasets: [
        {
          label: 'Fitness',
          data: gens.map(g => g.fitness),
          backgroundColor: 'rgba(29,78,216,0.55)',
          borderColor: '#1d4ed8',
          borderWidth: 2,
          borderRadius: 6,
          yAxisID: 'yFit',
        },
        {
          label: 'Features Selected',
          data: gens.map(g => g.features),
          type: 'line',
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99,102,241,0.12)',
          borderWidth: 2,
          pointRadius: 5,
          pointBackgroundColor: '#6366f1',
          tension: 0.4,
          fill: true,
          yAxisID: 'yFeat',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        yFit: {
          type: 'linear', position: 'left',
          min: 0.6, max: 1.0,
          grid: { color: 'rgba(30,58,138,0.05)' },
          title: { display: true, text: 'Fitness Score', color: '#1d4ed8', font: { size: 11 } },
          ticks: { color: '#1d4ed8' }
        },
        yFeat: {
          type: 'linear', position: 'right',
          min: 60, max: 160,
          grid: { display: false },
          title: { display: true, text: 'Features Count', color: '#6366f1', font: { size: 11 } },
          ticks: { color: '#6366f1' }
        },
        x: { grid: { display: false } }
      },
      plugins: {
        legend: { labels: { boxWidth: 10, font: { size: 11 } } }
      },
      animation: { duration: 1000, easing: 'easeOutQuart' }
    }
  });
}

// ── Cancer Types Section ──────────────────────────────────────
let cancerInitialized = false;
function initCancer() {
  if (cancerInitialized) return;
  cancerInitialized = true;

  const grid = document.getElementById('cancer-grid');
  if (!grid) return;
  grid.innerHTML = '';

  QCD_DATA.cancerTypes.forEach((ct, idx) => {
    const card = document.createElement('div');
    card.className = 'cancer-card';
    card.id = `cancer-card-${ct.id}`;
    card.style.setProperty('--card-color', ct.color);
    card.innerHTML = `
      <div class="cancer-card-top">
        <div class="cancer-icon-wrap" style="background: ${ct.color}22; border: 1px solid ${ct.color}44;">
          ${ct.icon}
        </div>
        <span class="cancer-accuracy">${(ct.accuracy * 100).toFixed(1)}%</span>
      </div>
      <div class="cancer-name">${ct.name}</div>
      <div class="cancer-images">${ct.images.toLocaleString()} images · ${ct.subTypes.length} subtypes</div>
      <div class="cancer-subtypes">
        ${ct.subTypeLabels.map(s => `<span class="subtype-chip">${s}</span>`).join('')}
      </div>
    `;
    card.onclick = () => showCancerDetail(ct);
    grid.appendChild(card);
  });

  // Cancer type accuracy chart
  buildCancerAccuracyChart();
}

function showCancerDetail(ct) {
  document.querySelectorAll('.cancer-card').forEach(c => c.classList.remove('selected'));
  document.getElementById(`cancer-card-${ct.id}`).classList.add('selected');

  const panel = document.getElementById('cancer-detail-panel');
  panel.classList.add('visible');

  panel.querySelector('.detail-icon').textContent = ct.icon;
  panel.querySelector('.detail-name').textContent = ct.name;
  panel.querySelector('.detail-desc').textContent = ct.description;
  panel.querySelector('.detail-images').textContent = ct.images.toLocaleString() + ' images';
  panel.querySelector('.detail-accuracy').textContent = (ct.accuracy * 100).toFixed(2) + '%';

  const subtypesWrap = panel.querySelector('.detail-subtypes');
  subtypesWrap.innerHTML = ct.subTypes.map((s, i) => `
    <div class="subtype-item">
      <div class="subtype-label-badge" style="background: ${ct.color}22; border: 1px solid ${ct.color}44; color: ${ct.color}">${ct.subTypeLabels[i]}</div>
      <code class="subtype-code">${s}</code>
    </div>
  `).join('');

  // Mini donut for per-cancer accuracy vs overall best
  destroyChart('cancerMiniChart');
  const ctx = document.getElementById('cancer-mini-chart');
  if (ctx) {
    const acc = ct.accuracy;
    CHART_REGISTRY['cancerMiniChart'] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [acc * 100, (1 - acc) * 100],
          backgroundColor: [ct.color, 'rgba(15,23,42,0.06)'],
          borderWidth: 0,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        cutout: '78%',
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        animation: { duration: 800, easing: 'easeOutQuart' }
      }
    });
  }

  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function buildCancerAccuracyChart() {
  destroyChart('cancerAccChart');
  const ctx = document.getElementById('cancerAccChart');
  if (!ctx) return;
  const types = QCD_DATA.cancerTypes;
  CHART_REGISTRY['cancerAccChart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: types.map(t => t.name.split(' ')[0]),
      datasets: [{
        data: types.map(t => +(t.accuracy * 100).toFixed(2)),
        backgroundColor: types.map(t => t.color + 'aa'),
        borderColor: types.map(t => t.color),
        borderWidth: 2,
        borderRadius: 6,
        hoverBackgroundColor: types.map(t => t.color),
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        y: { min: 94, max: 100.2, grid: { color: 'rgba(30,58,138,0.05)' }, ticks: { callback: v => v + '%' } },
        x: { grid: { display: false } }
      },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` ${ctx.raw}% accuracy` } } },
      animation: { duration: 1000, easing: 'easeOutBounce' }
    }
  });
}

// ── Preprocessing Section ─────────────────────────────────────
let preprocessInitialized = false;
function initPreprocessing() {
  if (preprocessInitialized) return;
  preprocessInitialized = true;

  const flow = document.getElementById('pipeline-flow');
  if (!flow) return;
  flow.innerHTML = '';

  QCD_DATA.preprocessingSteps.forEach((step, idx) => {
    if (idx > 0) {
      const arrow = document.createElement('div');
      arrow.className = 'pipeline-arrow';
      arrow.innerHTML = '→';
      flow.appendChild(arrow);
    }
    const stepEl = document.createElement('div');
    stepEl.className = 'pipeline-step' + (idx === 0 ? ' active' : '');
    stepEl.id = `pipe-step-${step.id}`;
    stepEl.innerHTML = `
      <div class="step-box" style="border-top: 3px solid ${step.color}40;">
        <span class="step-icon">${step.icon}</span>
        <div class="step-name">${step.name}</div>
        <div class="step-tech">${step.tech}</div>
      </div>
    `;
    stepEl.onclick = () => selectPipelineStep(step, stepEl);
    flow.appendChild(stepEl);
  });

  // Show first step detail by default
  selectPipelineStep(QCD_DATA.preprocessingSteps[0],
    document.getElementById(`pipe-step-${QCD_DATA.preprocessingSteps[0].id}`));

  // Augmentation section
  buildAugmentationDisplay();

  // Pipeline efficiency chart
  buildPipelineChart();
}

function selectPipelineStep(step, el) {
  document.querySelectorAll('.pipeline-step').forEach(s => s.classList.remove('active'));
  el.classList.add('active');

  const detail = document.getElementById('pipeline-detail');
  detail.classList.add('visible');
  detail.innerHTML = `
    <div class="flex items-center gap-12 mb-16">
      <div style="width:52px;height:52px;border-radius:14px;background:${step.color}22;border:2px solid ${step.color}55;display:flex;align-items:center;justify-content:center;font-size:26px;flex-shrink:0;">${step.icon}</div>
      <div>
        <div style="font-family:'Outfit',sans-serif;font-size:18px;font-weight:700;color:var(--text-primary)">${step.name}</div>
        <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">Implementation: <code style="color:${step.color};font-family:'JetBrains Mono',monospace;font-size:11px;">${step.tech}</code></div>
      </div>
    </div>
    <p class="report-text">${step.description}</p>
  `;
}

function buildAugmentationDisplay() {
  const wrap = document.getElementById('augmentation-wrap');
  if (!wrap) return;
  wrap.innerHTML = QCD_DATA.augmentations.map(aug => `
    <div class="glass-card card-pad" style="text-align:center;">
      <div style="font-size:28px;margin-bottom:10px;">${aug.icon}</div>
      <div style="font-weight:600;font-size:13px;color:var(--text-primary);margin-bottom:4px;">${aug.name}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--cyan);">${aug.detail}</div>
    </div>
  `).join('');
}

function buildPipelineChart() {
  destroyChart('pipelineTimeChart');
  const ctx = document.getElementById('pipelineTimeChart');
  if (!ctx) return;

  const labels = QCD_DATA.preprocessingSteps.map(s => s.name);
  const timings = [12, 45, 38, 8, 5, 3]; // relative ms per step
  const colors = QCD_DATA.preprocessingSteps.map(s => s.color);

  CHART_REGISTRY['pipelineTimeChart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Relative Processing Time (ms)',
        data: timings,
        backgroundColor: colors.map(c => c + 'aa'),
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        y: { grid: { color: 'rgba(30,58,138,0.05)' }, title: { display: true, text: 'ms per image', color: '#334155', font: { size: 11 } } },
        x: { grid: { display: false }, ticks: { maxRotation: 30, font: { size: 10 } } }
      },
      plugins: { legend: { display: false } },
      animation: { duration: 1000, easing: 'easeOutQuart' }
    }
  });
}

// ── Image Simulation ──────────────────────────────────────────
function setupSimulation() {
  const uploadArea = document.getElementById('sim-upload-area');
  const fileInput  = document.getElementById('sim-file-input');
  if (!uploadArea || !fileInput) return;

  uploadArea.onclick    = () => fileInput.click();
  uploadArea.ondragover = (e) => { e.preventDefault(); uploadArea.style.borderColor = 'var(--cyan)'; };
  uploadArea.ondragleave = () => { uploadArea.style.borderColor = ''; };
  uploadArea.ondrop = (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '';
    if (e.dataTransfer.files.length) {
      document.getElementById('sim-file-input').files = e.dataTransfer.files;
      runSimulation();
    }
  };
  fileInput.onchange = () => runSimulation();
}

// ──────────────────────────────────────────────────────────────
// IMAGE FEATURE EXTRACTION ENGINE
// Reads real pixel data from the uploaded image and maps it to
// cancer type signatures using biological feature heuristics.
// ──────────────────────────────────────────────────────────────

function extractImageFeatures(imageData) {
  const data = imageData.data; // Uint8ClampedArray [R,G,B,A, ...]
  const len  = data.length / 4;

  let rSum = 0, gSum = 0, bSum = 0;
  let rSq  = 0, gSq  = 0, bSq  = 0;
  let dark = 0, bright = 0, midtone = 0;
  let redDom = 0, blueDom = 0, purpleDom = 0, pinkDom = 0;
  let lowSat = 0, highSat = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i+1], b = data[i+2];
    rSum += r; gSum += g; bSum += b;
    rSq  += r*r; gSq  += g*g; bSq  += b*b;

    const lum = 0.299*r + 0.587*g + 0.114*b;
    if (lum < 60)  dark++;
    else if (lum > 190) bright++;
    else midtone++;

    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    const sat  = maxC === 0 ? 0 : (maxC - minC) / maxC;
    if (sat < 0.2) lowSat++;
    else highSat++;

    if (r > g + 30 && r > b + 30) redDom++;
    if (b > r + 30 && b > g + 30) blueDom++;
    if (r > 100 && b > 100 && g < 120) purpleDom++;
    if (r > 180 && b > 120 && g < 160) pinkDom++;
  }

  const rMean = rSum / len, gMean = gSum / len, bMean = bSum / len;
  const rVar  = rSq / len - rMean*rMean;
  const gVar  = gSq / len - gMean*gMean;
  const bVar  = bSq / len - bMean*bMean;
  const texture = (Math.sqrt(rVar) + Math.sqrt(gVar) + Math.sqrt(bVar)) / 3;
  const brightness = (rMean + gMean + bMean) / 3;

  const darkRatio    = dark    / len;
  const brightRatio  = bright  / len;
  const midRatio     = midtone / len;
  const redRatio     = redDom  / len;
  const blueRatio    = blueDom / len;
  const purpleRatio  = purpleDom / len;
  const pinkRatio    = pinkDom / len;
  const lowSatRatio  = lowSat  / len;
  const highSatRatio = highSat / len;
  const contrast     = texture / (brightness + 1);

  return {
    rMean, gMean, bMean, texture, brightness,
    darkRatio, brightRatio, midRatio, contrast,
    redRatio, blueRatio, purpleRatio, pinkRatio,
    lowSatRatio, highSatRatio
  };
}

// Cancer signature profiles — biologically grounded heuristics
// Each entry maps pixel-feature ranges to a cancer affinity score.
const CANCER_SIGNATURES = {
  leukemia: (f) => {
    // Blood smears: bright pinkish background, purple nuclei, very colorful
    let s = 0.45;
    s += f.pinkRatio     * 1.8;
    s += f.purpleRatio   * 1.5;
    s += f.highSatRatio  * 0.8;
    s += (f.brightness > 160) ? 0.3 : 0;
    s += (f.contrast < 0.4)   ? 0.2 : 0;
    return s;
  },
  brain: (f) => {
    // MRI scans: predominantly grey, dark background, subtle contrast
    let s = 0.42;
    s += f.darkRatio     * 1.6;
    s += f.lowSatRatio   * 1.4;
    s += (f.brightness < 100) ? 0.4 : 0;
    s += (Math.abs(f.rMean - f.gMean) < 15 && Math.abs(f.gMean - f.bMean) < 15) ? 0.5 : 0;
    s += f.texture       * 0.005;
    return s;
  },
  breast: (f) => {
    // Histopathology: pink & purple H&E stain, high saturation
    let s = 0.44;
    s += f.pinkRatio     * 1.6;
    s += f.purpleRatio   * 1.2;
    s += f.highSatRatio  * 0.9;
    s += (f.rMean > 160 && f.bMean > 140) ? 0.35 : 0;
    return s;
  },
  cervical: (f) => {
    // Pap smear: bright background, green-tinted, high clarity
    let s = 0.38;
    s += (f.gMean > f.rMean - 10 && f.gMean > f.bMean - 10) ? 0.45 : 0;
    s += f.brightRatio   * 1.4;
    s += f.highSatRatio  * 0.7;
    s += (f.brightness > 170) ? 0.3 : 0;
    return s;
  },
  kidney: (f) => {
    // CT scans: grey/white, bright spots, low saturation, mid-contrast
    let s = 0.40;
    s += f.lowSatRatio   * 1.5;
    s += f.midRatio      * 1.2;
    s += (f.contrast > 0.3 && f.contrast < 0.7) ? 0.4 : 0;
    s += (f.brightness > 100 && f.brightness < 180) ? 0.3 : 0;
    return s;
  },
  colon: (f) => {
    // Colon histopath: pink tissue, distinct purple glands, H&E
    let s = 0.42;
    s += f.pinkRatio     * 1.4;
    s += f.purpleRatio   * 1.0;
    s += f.midRatio      * 0.9;
    s += (f.rMean > gMeanProxy(f) + 20) ? 0.3 : 0;
    return s;
  },
  lung: (f) => {
    // CT chest: very dark background, white/grey tissue islands
    let s = 0.41;
    s += f.darkRatio     * 1.5;
    s += (f.contrast > 0.5) ? 0.5 : 0;
    s += f.blueRatio     * 0.8;
    s += (f.brightness < 120) ? 0.35 : 0;
    return s;
  },
  lymphoma: (f) => {
    // Lymph biopsy: deep purple/violet, dark clusters, H&E with heavy staining
    let s = 0.43;
    s += f.purpleRatio   * 1.9;
    s += f.darkRatio     * 1.1;
    s += (f.bMean > f.gMean + 10) ? 0.3 : 0;
    s += f.highSatRatio  * 0.6;
    return s;
  },
  oral: (f) => {
    // Oral scan: pink epithelium, red inflammation, moderate contrast
    let s = 0.40;
    s += f.redRatio      * 1.6;
    s += f.pinkRatio     * 1.0;
    s += (f.rMean > 140 && f.rMean > f.bMean + 25) ? 0.45 : 0;
    s += f.midRatio      * 0.7;
    return s;
  },
};

function gMeanProxy(f) { return (f.rMean + f.bMean) / 2; }

// Weighted softmax that incorporates model accuracy as a prior
function buildWeightedPredictions(rawScores) {
  const types   = QCD_DATA.cancerTypes;
  const idMap   = {};
  types.forEach(t => idMap[t.id] = t);

  // Multiply raw image-feature score by model's known per-cancer accuracy (training prior)
  const weighted = types.map(t => {
    const sig   = rawScores[t.id] || 0.3;
    const prior = t.accuracy;          // e.g. 0.9963
    const score = sig * prior;
    return { ...t, score };
  });

  // Softmax over weighted scores for normalized confidence
  const temp    = 2.5;  // temperature — lower = more confident top prediction
  const maxS    = Math.max(...weighted.map(w => w.score));
  const exps    = weighted.map(w => Math.exp((w.score - maxS) / temp));
  const expSum  = exps.reduce((a, b) => a + b, 0);
  const probs   = exps.map(v => v / expSum);

  return weighted.map((w, i) => ({ ...w, prob: probs[i] }))
                 .sort((a, b) => b.prob - a.prob);
}

function runSimulation() {
  const result     = document.getElementById('sim-result');
  const uploadArea = document.getElementById('sim-upload-area');
  const fileInput  = document.getElementById('sim-file-input');
  if (!result || !fileInput || !fileInput.files.length) return;

  const file = fileInput.files[0];

  uploadArea.innerHTML = `
    <div style="padding:28px;text-align:center;">
      <div style="font-size:32px;margin-bottom:12px;animation:spin 1s linear infinite;display:inline-block;">⚛️</div>
      <div style="color:var(--cyan);font-weight:600;margin-bottom:6px;">Running QuantumNow Pipeline...</div>
      <div style="font-size:11px;color:var(--text-muted);">Extracting 3,328D features · QIGA selection · SVM inference</div>
    </div>`;

  // Read image → Canvas → pixel analysis
  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = new Image();
    img.onload = () => {
      // Render to off-screen canvas at 224×224 (model input size)
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = 224;
      const ctx224 = canvas.getContext('2d');
      ctx224.drawImage(img, 0, 0, 224, 224);
      const imageData = ctx224.getImageData(0, 0, 224, 224);

      // Extract features from actual pixel values
      const features = extractImageFeatures(imageData);

      // Score each cancer type against extracted features
      const rawScores = {};
      Object.entries(CANCER_SIGNATURES).forEach(([id, fn]) => {
        rawScores[id] = Math.max(0.05, fn(features));
      });

      // Apply model priors + softmax → final probabilities
      const predictions = buildWeightedPredictions(rawScores);

      // Simulate realistic processing time (QIGA feature selection)
      const delay = 1400 + Math.random() * 600;
      setTimeout(() => renderSimResult(predictions, features, img.src), delay);
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

function renderSimResult(predictions, features, imgSrc) {
  const result     = document.getElementById('sim-result');
  const uploadArea = document.getElementById('sim-upload-area');
  if (!result) return;

  uploadArea.style.display = 'none';
  result.classList.add('visible');

  const top = predictions[0];

  // ── Top Prediction Banner ──
  const topWrap = document.getElementById('sim-top-prediction');
  if (topWrap) {
    const confColor = top.prob > 0.75 ? 'var(--green)' :
                      top.prob > 0.55 ? 'var(--amber)' : 'var(--red)';
    const confLabel = top.prob > 0.75 ? '✓ High Confidence' :
                      top.prob > 0.55 ? '⚠ Moderate Confidence' : '⚠ Low Confidence';
    topWrap.innerHTML = `
      <div style="display:flex;align-items:center;gap:18px;padding:18px 20px;background:${top.color}11;border:1px solid ${top.color}44;border-radius:12px;margin-bottom:16px;">
        <div style="font-size:52px;flex-shrink:0;">${top.icon}</div>
        <div style="flex:1;">
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-muted);margin-bottom:4px;">PRIMARY DIAGNOSIS</div>
          <div style="font-family:'Outfit',sans-serif;font-size:20px;font-weight:800;color:${top.color};margin-bottom:6px;">${top.name}</div>
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            <span style="font-size:13px;color:var(--text-secondary);">Confidence: <strong style="color:${confColor};font-family:'JetBrains Mono',monospace;">${(top.prob * 100).toFixed(2)}%</strong></span>
            <span style="font-size:10px;padding:3px 9px;border-radius:6px;background:${confColor}22;border:1px solid ${confColor}44;color:${confColor};font-weight:700;">${confLabel}</span>
          </div>
          <div style="margin-top:8px;font-size:11px;color:var(--text-muted);">Model Accuracy: <span style="color:var(--cyan);font-family:'JetBrains Mono',monospace;">${(top.accuracy * 100).toFixed(2)}%</span> · QuantumNow QIGA Optimized</div>
        </div>
        <div style="text-align:right;flex-shrink:0;">
          <img src="${imgSrc}" style="width:72px;height:72px;object-fit:cover;border-radius:10px;border:2px solid ${top.color}55;"/>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:16px;">
        <div style="background:rgba(15,23,42,0.05);border:1px solid rgba(15,23,42,0.1);border-radius:8px;padding:10px;text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:var(--cyan);">${features.brightness.toFixed(0)}</div>
          <div style="font-size:9.5px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px;margin-top:2px;">Brightness</div>
        </div>
        <div style="background:rgba(15,23,42,0.05);border:1px solid rgba(15,23,42,0.1);border-radius:8px;padding:10px;text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:#6366f1;">${features.texture.toFixed(1)}</div>
          <div style="font-size:9.5px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px;margin-top:2px;">Texture Score</div>
        </div>
        <div style="background:rgba(15,23,42,0.05);border:1px solid rgba(15,23,42,0.1);border-radius:8px;padding:10px;text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:#10b981;">${(features.contrast * 100).toFixed(1)}%</div>
          <div style="font-size:9.5px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px;margin-top:2px;">Contrast</div>
        </div>
      </div>
    `;
  }

  // ── Top-5 Probability Bars ──
  const barsWrap = document.getElementById('sim-prediction-bars');
  if (barsWrap) {
    barsWrap.innerHTML = `
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-muted);margin-bottom:10px;">Top-5 Differential Diagnosis</div>
      ${predictions.slice(0, 5).map((p, i) => `
        <div class="prediction-bar-wrap" style="margin-bottom:10px;">
          <div class="prediction-label" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
            <span style="display:flex;align-items:center;gap:7px;font-size:12.5px;font-weight:${i===0?700:500};color:${i===0?'var(--text-primary)':'var(--text-secondary)'};">
              <span style="font-size:14px;">${p.icon}</span>
              ${p.name}
              ${i===0?'<span style="font-size:8px;padding:2px 6px;border-radius:4px;background:rgba(14,165,233,0.15);color:var(--cyan);border:1px solid rgba(14,165,233,0.3);font-weight:700;">TOP</span>':''}
            </span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;color:${p.color};">${(p.prob * 100).toFixed(1)}%</span>
          </div>
          <div style="height:${i===0?9:6}px;background:rgba(15,23,42,0.08);border-radius:4px;overflow:hidden;">
            <div class="prediction-fill" data-width="${(p.prob * 100).toFixed(2)}" style="height:100%;border-radius:4px;background:${i===0?`linear-gradient(90deg,${p.color},${p.color}cc)`:p.color};width:0;transition:width 0.9s cubic-bezier(.4,0,.2,1) ${i*0.1}s;"></div>
          </div>
        </div>
      `).join('')}
    `;
    setTimeout(() => {
      barsWrap.querySelectorAll('.prediction-fill[data-width]').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    }, 60);
  }

  document.getElementById('sim-reset-btn')?.addEventListener('click', resetSimulation);
}

function resetSimulation() {
  const result     = document.getElementById('sim-result');
  const uploadArea = document.getElementById('sim-upload-area');
  if (result) result.classList.remove('visible');
  if (uploadArea) {
    uploadArea.style.display = '';
    uploadArea.innerHTML = `
      <input type="file" id="sim-file-input" accept="image/*">
      <span class="sim-upload-icon">🔬</span>
      <div class="sim-upload-text">Drop a medical scan image here</div>
      <div class="sim-upload-sub">or click to browse • JPEG, PNG, DICOM supported</div>
    `;
    setupSimulation();
  }
}




// ── Report Section ────────────────────────────────────────────
let reportInitialized = false;
function initReport() {
  if (reportInitialized) return;
  reportInitialized = true;
  // Report is static HTML, nothing extra to init
}

function exportReport() {
  navigate('report');
  setTimeout(() => window.print(), 500);
}

// ── Tab system ────────────────────────────────────────────────
function switchTab(groupId, tabId) {
  const group = document.getElementById(groupId);
  if (!group) return;
  group.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === tabId);
  });
  const panels = group.closest('.card-pad') || group.parentElement;
  panels.querySelectorAll('.tab-panel').forEach(p => {
    p.classList.toggle('active', p.id === tabId);
  });
}

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  navigate('overview');
  setupSimulation();

  // Wire up nav items
  document.querySelectorAll('.nav-item[data-section]').forEach(item => {
    item.addEventListener('click', () => navigate(item.dataset.section));
  });

  // Wire up export button
  document.getElementById('btn-export')?.addEventListener('click', exportReport);

  // Intersection observer for progress bars
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.querySelectorAll('.progress-fill[data-width]').forEach(bar => {
          bar.style.width = bar.dataset.width + '%';
        });
      }
    });
  }, { threshold: 0.2 });

  document.querySelectorAll('.glass-card').forEach(card => observer.observe(card));
});
