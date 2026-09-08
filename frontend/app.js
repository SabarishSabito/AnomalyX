/**
 * AnomalyX - Main Dashboard Application Controller
 * Handles real-time telemetry streaming, Chart.js visualizations,
 * model parameter tuning, dataset uploading, alert feed, and explainability modal.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Lucide Icons
  if (window.lucide) {
    window.lucide.createIcons();
  }

  // API Base Configuration
  const API_BASE = '/api';

  // Global State
  const state = {
    activeModel: 'isolation_forest',
    contamination: 0.05,
    category: 'server',
    isStreaming: true,
    streamTimer: null,
    streamSpeedMs: 1500,
    telemetryHistory: [], // Max 30 points
    scoreHistory: [],
    alerts: [],
    metricsSummary: {
      totalAnalyzed: 0,
      anomaliesCount: 0,
      confidenceSum: 0,
      criticalCount: 0,
      highCount: 0,
      mediumCount: 0,
      latencyMs: 0
    },
    activeFilter: 'all',
    searchQuery: ''
  };

  // DOM Elements
  const elements = {
    statusIndicator: document.getElementById('statusIndicator'),
    statusText: document.getElementById('statusText'),
    activeModelBadge: document.getElementById('activeModelBadge'),
    streamToggleBtn: document.getElementById('streamToggleBtn'),
    streamIcon: document.getElementById('streamIcon'),
    streamBtnText: document.getElementById('streamBtnText'),
    sampleDataBtn: document.getElementById('sampleDataBtn'),

    // KPI Elements
    kpiTotalAnalyzed: document.getElementById('kpiTotalAnalyzed'),
    kpiAnomaliesCount: document.getElementById('kpiAnomaliesCount'),
    kpiSeverityBreakdown: document.getElementById('kpiSeverityBreakdown'),
    kpiAnomalyRate: document.getElementById('kpiAnomalyRate'),
    kpiLatency: document.getElementById('kpiLatency'),

    // Controls
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    algorithmSelect: document.getElementById('algorithmSelect'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    applyTuningBtn: document.getElementById('applyTuningBtn'),

    // Alerts
    alertCountBadge: document.getElementById('alertCountBadge'),
    alertTableBody: document.getElementById('alertTableBody'),
    alertSearchInput: document.getElementById('alertSearchInput'),
    filterButtons: document.querySelectorAll('.filter-btn'),
    exportCsvBtn: document.getElementById('exportCsvBtn'),

    // Modal
    modalBackdrop: document.getElementById('modalBackdrop'),
    modalAlertId: document.getElementById('modalAlertId'),
    modalCloseBtn: document.getElementById('modalCloseBtn'),
    modalBodyContent: document.getElementById('modalBodyContent')
  };

  // Charts References
  let telemetryChart = null;
  let scoreChart = null;

  // Initialize Charts
  initCharts();

  // Attach Event Listeners
  attachEventListeners();

  // Start Real-Time Stream Loop
  startStream();

  /**
   * Initialize Chart.js Instances
   */
  function initCharts() {
    const telemetryCtx = document.getElementById('telemetryChart').getContext('2d');
    const scoreCtx = document.getElementById('scoreChart').getContext('2d');

    // Multi-metric Telemetry Chart
    telemetryChart = new Chart(telemetryCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'CPU Usage %',
            data: [],
            borderColor: '#06B6D4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            borderWidth: 2,
            tension: 0.35,
            pointRadius: 0
          },
          {
            label: 'Memory Usage %',
            data: [],
            borderColor: '#8B5CF6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            borderWidth: 2,
            tension: 0.35,
            pointRadius: 0
          },
          {
            label: 'Network IO',
            data: [],
            borderColor: '#3B82F6',
            borderWidth: 1.5,
            tension: 0.35,
            pointRadius: 0,
            hidden: true
          },
          {
            label: 'Latency ms',
            data: [],
            borderColor: '#F59E0B',
            borderWidth: 1.5,
            tension: 0.35,
            pointRadius: 0,
            hidden: true
          },
          {
            label: 'Anomalies',
            data: [], // [{x, y}] scatter points
            type: 'scatter',
            backgroundColor: '#EF4444',
            borderColor: '#FFF',
            borderWidth: 2,
            pointRadius: 7,
            pointHoverRadius: 9
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748B', font: { family: 'Inter', size: 11 } }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748B', font: { family: 'Inter', size: 11 } },
            beginAtZero: true
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0F172A',
            titleColor: '#F8FAFC',
            bodyColor: '#94A3B8',
            borderColor: 'rgba(6, 182, 212, 0.4)',
            borderWidth: 1,
            padding: 10
          }
        }
      }
    });

    // Score & Threshold Chart
    scoreChart = new Chart(scoreCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Anomaly Score',
            data: [],
            borderColor: '#8B5CF6',
            backgroundColor: 'rgba(139, 92, 246, 0.2)',
            fill: true,
            borderWidth: 2,
            tension: 0.3
          },
          {
            label: 'Threshold',
            data: [],
            borderColor: '#EF4444',
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
          x: { display: false },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748B', font: { family: 'Inter', size: 11 } },
            min: 0,
            max: 1.0
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0F172A',
            titleColor: '#F8FAFC',
            bodyColor: '#94A3B8',
            borderColor: 'rgba(239, 68, 68, 0.4)',
            borderWidth: 1
          }
        }
      }
    });
  }

  /**
   * Event Listeners Setup
   */
  function attachEventListeners() {
    // Stream Toggle
    elements.streamToggleBtn.addEventListener('click', toggleStream);

    // Sample Benchmark Dataset Button
    elements.sampleDataBtn.addEventListener('click', loadSampleDataset);

    // Category Preset Buttons
    document.querySelectorAll('.preset-buttons button').forEach(btn => {
      btn.addEventListener('click', (e) => {
        state.category = e.target.dataset.category || 'server';
        loadSampleDataset();
      });
    });

    // Threshold Slider Sync
    elements.thresholdSlider.addEventListener('input', (e) => {
      state.contamination = parseFloat(e.target.value);
      elements.thresholdValue.textContent = `${(state.contamination * 100).toFixed(1)}%`;
    });

    // Model Apply Button
    elements.applyTuningBtn.addEventListener('click', updateModelConfig);

    // Drag & Drop Handling
    elements.dropZone.addEventListener('click', () => elements.fileInput.click());
    elements.dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      elements.dropZone.classList.add('dragover');
    });
    elements.dropZone.addEventListener('dragleave', () => elements.dropZone.classList.remove('dragover'));
    elements.dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      elements.dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
    elements.fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) {
        handleFileUpload(e.target.files[0]);
      }
    });

    // Alert Search & Severity Filters
    elements.alertSearchInput.addEventListener('input', (e) => {
      state.searchQuery = e.target.value.toLowerCase();
      renderAlertsTable();
    });

    elements.filterButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        elements.filterButtons.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        state.activeFilter = e.target.dataset.severity;
        renderAlertsTable();
      });
    });

    // Export CSV
    elements.exportCsvBtn.addEventListener('click', exportAlertsCSV);

    // Modal Close
    elements.modalCloseBtn.addEventListener('click', closeModal);
    elements.modalBackdrop.addEventListener('click', (e) => {
      if (e.target === elements.modalBackdrop) closeModal();
    });
  }

  /**
   * Real-Time Stream Loop
   */
  function startStream() {
    if (state.streamTimer) clearInterval(state.streamTimer);
    state.isStreaming = true;
    updateStreamUI();
    fetchNextStreamFrame();
    state.streamTimer = setInterval(fetchNextStreamFrame, state.streamSpeedMs);
  }

  function pauseStream() {
    state.isStreaming = false;
    if (state.streamTimer) clearInterval(state.streamTimer);
    updateStreamUI();
  }

  function toggleStream() {
    if (state.isStreaming) pauseStream();
    else startStream();
  }

  function updateStreamUI() {
    if (state.isStreaming) {
      elements.streamBtnText.textContent = 'Pause Stream';
      elements.streamIcon.setAttribute('data-lucide', 'pause');
      elements.statusIndicator.className = 'status-indicator online';
      elements.statusText.textContent = 'API ONLINE';
    } else {
      elements.streamBtnText.textContent = 'Resume Stream';
      elements.streamIcon.setAttribute('data-lucide', 'play');
      elements.statusIndicator.className = 'status-indicator offline';
      elements.statusText.textContent = 'STREAM PAUSED';
    }
    if (window.lucide) window.lucide.createIcons();
  }

  /**
   * Fetch Next Frame from API
   */
  async function fetchNextStreamFrame() {
    try {
      const response = await fetch(`${API_BASE}/stream/next`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: state.activeModel,
          contamination: state.contamination,
          category: state.category
        })
      });

      if (!response.ok) throw new Error('API Error');
      const data = await response.json();
      processFrame(data.frame, data.result, data.summary);
    } catch (err) {
      console.warn('Stream fetch error:', err);
      elements.statusIndicator.className = 'status-indicator offline';
      elements.statusText.textContent = 'DISCONNECTED';
    }
  }

  /**
   * Process Single Frame and Update Dashboard
   */
  function processFrame(frame, result, summary) {
    const timestamp = frame.timestamp || new Date().toLocaleTimeString();

    // 1. Update Telemetry History
    state.telemetryHistory.push({
      timestamp,
      cpu: frame.cpu_usage || frame.transaction_amount || frame.temperature_celsius || 0,
      memory: frame.memory_usage || frame.tx_frequency_per_min || frame.vibration_g || 0,
      network: frame.network_io || frame.risk_score || 0,
      latency: frame.latency_ms || frame.location_delta_km || 0,
      isAnomaly: result.is_anomaly,
      score: result.anomaly_score
    });

    if (state.telemetryHistory.length > 30) {
      state.telemetryHistory.shift();
    }

    // 2. Update Charts
    updateCharts();

    // 3. Update Metrics KPI
    state.metricsSummary.totalAnalyzed = summary.total_analyzed;
    state.metricsSummary.anomaliesCount = summary.anomalies_count;
    state.metricsSummary.latencyMs = summary.latency_ms;

    if (result.is_anomaly) {
      if (result.severity === 'critical') state.metricsSummary.criticalCount++;
      else if (result.severity === 'high') state.metricsSummary.highCount++;
      else state.metricsSummary.mediumCount++;

      // Log to Alert Feed
      const alertObj = {
        id: `ALT-${Math.random().toString(36).substr(2, 6).toUpperCase()}`,
        timestamp,
        severity: result.severity,
        anomaly_score: result.anomaly_score,
        confidence_pct: result.confidence_pct,
        top_contributor: result.explainability.top_contributor,
        metrics: result.metrics,
        explainability: result.explainability
      };

      state.alerts.unshift(alertObj);
      if (state.alerts.length > 100) state.alerts.pop();
      renderAlertsTable();
    }

    updateKPIs();
  }

  /**
   * Update Chart.js Datasets
   */
  function updateCharts() {
    const labels = state.telemetryHistory.map(item => item.timestamp);
    const cpuData = state.telemetryHistory.map(item => item.cpu);
    const memoryData = state.telemetryHistory.map(item => item.memory);
    const networkData = state.telemetryHistory.map(item => item.network);
    const latencyData = state.telemetryHistory.map(item => item.latency);

    // Scatter points for anomalies
    const anomalyScatter = [];
    state.telemetryHistory.forEach((item, index) => {
      if (item.isAnomaly) {
        anomalyScatter.push({ x: item.timestamp, y: item.cpu });
      }
    });

    telemetryChart.data.labels = labels;
    telemetryChart.data.datasets[0].data = cpuData;
    telemetryChart.data.datasets[1].data = memoryData;
    telemetryChart.data.datasets[2].data = networkData;
    telemetryChart.data.datasets[3].data = latencyData;
    telemetryChart.data.datasets[4].data = anomalyScatter;
    telemetryChart.update('none');

    // Score Chart
    const scoreData = state.telemetryHistory.map(item => item.score);
    const thresholdData = state.telemetryHistory.map(() => state.contamination * 4.0);

    scoreChart.data.labels = labels;
    scoreChart.data.datasets[0].data = scoreData;
    scoreChart.data.datasets[1].data = thresholdData;
    scoreChart.update('none');
  }

  /**
   * Update KPI Cards UI
   */
  function updateKPIs() {
    elements.kpiTotalAnalyzed.textContent = state.metricsSummary.totalAnalyzed.toLocaleString();
    elements.kpiAnomaliesCount.textContent = state.metricsSummary.anomaliesCount.toLocaleString();

    const rate = state.metricsSummary.totalAnalyzed > 0
      ? ((state.metricsSummary.anomaliesCount / state.metricsSummary.totalAnalyzed) * 100).toFixed(1)
      : '0.0';
    elements.kpiAnomalyRate.textContent = `${rate}%`;

    elements.kpiSeverityBreakdown.textContent = `Critical: ${state.metricsSummary.criticalCount} | High: ${state.metricsSummary.highCount}`;
    elements.kpiLatency.textContent = `${state.metricsSummary.latencyMs} ms`;
    elements.alertCountBadge.textContent = `${state.alerts.length} Alerts`;
  }

  /**
   * Render Alerts Table with Search & Filter
   */
  function renderAlertsTable() {
    const filtered = state.alerts.filter(item => {
      const matchesSeverity = state.activeFilter === 'all' || item.severity.toLowerCase() === state.activeFilter;
      const matchesSearch = !state.searchQuery ||
        item.id.toLowerCase().includes(state.searchQuery) ||
        item.top_contributor.toLowerCase().includes(state.searchQuery);
      return matchesSeverity && matchesSearch;
    });

    if (filtered.length === 0) {
      elements.alertTableBody.innerHTML = `
        <tr class="empty-row">
          <td colspan="8">No matching anomalies found. Monitoring stream...</td>
        </tr>`;
      return;
    }

    elements.alertTableBody.innerHTML = filtered.map((item) => `
      <tr>
        <td class="font-mono">${item.id}</td>
        <td>${item.timestamp}</td>
        <td><span class="badge-sev ${item.severity}">${item.severity}</span></td>
        <td><strong>${item.anomaly_score.toFixed(3)}</strong></td>
        <td>${item.confidence_pct}%</td>
        <td class="text-cyan"><strong>${item.top_contributor}</strong></td>
        <td><span class="font-mono text-dim">${formatMetricsSnippet(item.metrics)}</span></td>
        <td>
          <button class="btn btn-sm btn-outline inspect-btn" data-id="${item.id}">
            Inspect Cause
          </button>
        </td>
      </tr>
    `).join('');

    // Attach click listener for modal
    document.querySelectorAll('.inspect-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const id = e.target.closest('button').dataset.id;
        const targetAlert = state.alerts.find(a => a.id === id);
        if (targetAlert) openExplainabilityModal(targetAlert);
      });
    });
  }

  function formatMetricsSnippet(metrics) {
    return Object.entries(metrics).slice(0, 2).map(([k, v]) => `${k}: ${v}`).join(' | ');
  }

  /**
   * Update Model Configuration via Backend
   */
  async function updateModelConfig() {
    state.activeModel = elements.algorithmSelect.value;
    const modelName = elements.algorithmSelect.options[elements.algorithmSelect.selectedIndex].text;
    elements.activeModelBadge.textContent = modelName.split('(')[0].trim();

    try {
      const response = await fetch(`${API_BASE}/models/configure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: state.activeModel,
          contamination: state.contamination
        })
      });

      if (response.ok) {
        showNotification('Model updated successfully');
      }
    } catch (err) {
      console.error('Failed to configure model:', err);
    }
  }

  /**
   * Ingest Sample Dataset
   */
  async function loadSampleDataset() {
    try {
      const response = await fetch(`${API_BASE}/datasets/sample?category=${state.category}`);
      if (!response.ok) throw new Error('Failed to load sample data');
      const data = await response.json();

      state.telemetryHistory = [];
      data.records.forEach(rec => {
        state.telemetryHistory.push({
          timestamp: rec.timestamp || new Date().toLocaleTimeString(),
          cpu: rec.metrics.cpu_usage || rec.metrics.transaction_amount || rec.metrics.temperature_celsius || 0,
          memory: rec.metrics.memory_usage || rec.metrics.tx_frequency_per_min || rec.metrics.vibration_g || 0,
          network: rec.metrics.network_io || rec.metrics.risk_score || 0,
          latency: rec.metrics.latency_ms || rec.metrics.location_delta_km || 0,
          isAnomaly: rec.is_anomaly,
          score: rec.anomaly_score
        });

        if (rec.is_anomaly) {
          const alertObj = {
            id: `ALT-${Math.random().toString(36).substr(2, 6).toUpperCase()}`,
            timestamp: rec.timestamp || new Date().toLocaleTimeString(),
            severity: rec.severity,
            anomaly_score: rec.anomaly_score,
            confidence_pct: rec.confidence_pct,
            top_contributor: rec.explainability.top_contributor,
            metrics: rec.metrics,
            explainability: rec.explainability
          };
          state.alerts.unshift(alertObj);
        }
      });

      updateCharts();
      renderAlertsTable();
      showNotification(`Loaded ${data.total_records} sample benchmark records`);
    } catch (err) {
      console.error(err);
    }
  }

  /**
   * Drag & Drop File Upload Handler
   */
  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      showNotification(`Uploading and analyzing ${file.name}...`, 'info');
      const response = await fetch(`${API_BASE}/datasets/upload?model_id=${state.activeModel}&contamination=${state.contamination}`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Upload failed');
      }
      const data = await response.json();

      // Pause live stream so uploaded data remains visible on dashboard
      pauseStream();

      // Clear & populate telemetry history with uploaded records
      state.telemetryHistory = [];
      let criticalCount = 0;
      let highCount = 0;
      let mediumCount = 0;

      // Find dynamic numeric metric keys for chart visualizer
      const firstRecordMetrics = data.results[0]?.metrics || {};
      const numericKeys = Object.keys(firstRecordMetrics).filter(k => {
        const v = firstRecordMetrics[k];
        return typeof v === 'number' || (!isNaN(parseFloat(v)) && isFinite(v));
      });

      const key1 = numericKeys.find(k => /voltage|cpu|salary|val|amount/i.test(k)) || numericKeys[0];
      const key2 = numericKeys.find(k => /current|memory|age|freq/i.test(k)) || numericKeys[1];
      const key3 = numericKeys.find(k => /power|network|temp|io/i.test(k)) || numericKeys[2];
      const key4 = numericKeys.find(k => /factor|latency|demand|humidity/i.test(k)) || numericKeys[3];

      data.results.forEach((rec, idx) => {
        const getNum = (k) => {
          if (!k || !(k in rec.metrics)) return 0;
          const v = rec.metrics[k];
          return typeof v === 'number' ? v : parseFloat(v) || 0;
        };

        state.telemetryHistory.push({
          timestamp: rec.timestamp || `Rec #${idx + 1}`,
          cpu: getNum(key1),
          memory: getNum(key2),
          network: getNum(key3),
          latency: getNum(key4),
          isAnomaly: rec.is_anomaly,
          score: rec.anomaly_score
        });

        if (rec.is_anomaly) {
          if (rec.severity === 'critical') criticalCount++;
          else if (rec.severity === 'high') highCount++;
          else mediumCount++;

          const alertObj = {
            id: `ALT-${Math.random().toString(36).substr(2, 6).toUpperCase()}`,
            timestamp: rec.timestamp || `Rec #${idx + 1}`,
            severity: rec.severity,
            anomaly_score: rec.anomaly_score,
            confidence_pct: rec.confidence_pct,
            top_contributor: rec.explainability.top_contributor,
            metrics: rec.metrics,
            explainability: rec.explainability
          };
          state.alerts.unshift(alertObj);
        }
      });

      // Update KPIs
      state.metricsSummary.totalAnalyzed = data.total_records;
      state.metricsSummary.anomaliesCount = data.anomalies_found;
      state.metricsSummary.criticalCount = criticalCount;
      state.metricsSummary.highCount = highCount;
      state.metricsSummary.mediumCount = mediumCount;
      state.metricsSummary.latencyMs = 3.8;

      updateCharts();
      updateKPIs();
      renderAlertsTable();

      showNotification(`Uploaded ${data.filename}: Found ${data.anomalies_found} anomalies in ${data.total_records} records`, 'success');
    } catch (err) {
      console.error(err);
      showNotification(`File upload failed: ${err.message}`, 'error');
    }
  }

  /**
   * Explainability Modal Drawer
   */
  function openExplainabilityModal(alertItem) {
    elements.modalAlertId.textContent = `Root Cause Analysis: ${alertItem.id}`;

    const exp = alertItem.explainability;
    const breakDownHtml = exp.feature_breakdown.map(item => `
      <div class="attr-item">
        <div class="attr-header">
          <span><strong>${item.feature}</strong> (Val: ${item.raw_value}, Z: ${item.z_score})</span>
          <span><strong>${item.contribution_pct}%</strong> Contribution</span>
        </div>
        <div class="attr-bar-bg">
          <div class="attr-bar-fill" style="width: ${item.contribution_pct}%"></div>
        </div>
      </div>
    `).join('');

    elements.modalBodyContent.innerHTML = `
      <div class="modal-summary-grid">
        <p><strong>Flagged Timestamp:</strong> ${alertItem.timestamp}</p>
        <p><strong>Severity Rating:</strong> <span class="badge-sev ${alertItem.severity}">${alertItem.severity}</span></p>
        <p><strong>Anomaly Confidence:</strong> ${alertItem.confidence_pct}%</p>
        <p><strong>Top Anomaly Driver:</strong> <span class="text-cyan"><strong>${exp.top_contributor}</strong></span></p>
      </div>

      <div class="card p-4">
        <h3>Metric & Feature Breakdown</h3>
        <div class="attribution-bar-container mt-3">
          ${breakDownHtml}
        </div>
      </div>

      <div class="card p-4">
        <h3>Recommended Remediation</h3>
        <p class="text-dim mt-1">
          The machine learning model identified <strong>${exp.top_contributor}</strong> as the primary anomaly driver.
          Review the record details in the table or isolate the data ingestion source.
        </p>
      </div>
    `;

    elements.modalBackdrop.classList.add('open');
  }

  function closeModal() {
    elements.modalBackdrop.classList.remove('open');
  }

  /**
   * Export CSV
   */
  function exportAlertsCSV() {
    if (!state.alerts.length) return alert('No alerts available to export.');

    const headers = ['Alert ID', 'Timestamp', 'Severity', 'Anomaly Score', 'Confidence %', 'Top Contributor'];
    const rows = state.alerts.map(a => [
      a.id, a.timestamp, a.severity, a.anomaly_score, a.confidence_pct, a.top_contributor
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map(e => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `anomalyx_alerts_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function showNotification(msg, type = 'info') {
    console.log('[AnomalyX]', msg);
    let container = document.getElementById('toastContainer');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toastContainer';
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = `toast-notification ${type}`;
    toast.innerHTML = `<i data-lucide="${type === 'error' ? 'alert-circle' : 'check-circle'}"></i> <span>${msg}</span>`;
    container.appendChild(toast);
    if (window.lucide) window.lucide.createIcons();

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }
});
