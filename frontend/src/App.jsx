import { useEffect, useRef, useState } from 'react';
import {
  Activity, AlertCircle, AlertTriangle, ArrowRight, ArrowUpRight,
  Check, CheckCircle2, ChevronDown, CircleHelp, Clock3, Cpu,
  Download, FileImage, Fingerprint, FlaskConical, Gauge, Image as ImageIcon,
  LayoutDashboard, LoaderCircle, Menu, Microscope, RotateCcw,
  ScanSearch, Shield, ShieldCheck, Sparkles, Upload, X, Zap
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://127.0.0.1:8000';
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];

function formatScore(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '—';
}

function imageSource(value) {
  if (!value) return '';
  if (typeof value === 'string' && value.startsWith('data:')) return value;
  if (typeof value === 'string' && value.startsWith('http')) return value;
  return '';
}

function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState('');
  const [view, setView] = useState('dashboard');
  const [visualTab, setVisualTab] = useState('overlay');
  const [apiStatus, setApiStatus] = useState('checking');
  const [modelInfo, setModelInfo] = useState(null);
  const [history, setHistory] = useState([]);
  const [mobileNav, setMobileNav] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function checkBackend() {
      try {
        const [healthResponse, modelResponse] = await Promise.all([
          fetch(`${API_BASE}/api/health`),
          fetch(`${API_BASE}/api/model-info`)
        ]);

        if (!healthResponse.ok) throw new Error('API health check failed');

        const health = await healthResponse.json();
        const model = modelResponse.ok ? await modelResponse.json() : null;

        if (!cancelled) {
          setApiStatus(health.success ? 'online' : 'offline');
          setModelInfo(model?.data ?? model?.model_info ?? model ?? null);
        }
      } catch {
        if (!cancelled) {
          setApiStatus('offline');
          setModelInfo(null);
        }
      }
    }

    checkBackend();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  function selectFile(selected) {
    if (!selected) return;

    setError('');
    setResult(null);

    if (!ALLOWED_TYPES.includes(selected.type)) {
      setError('Unsupported file format. Please use PNG, JPG, WEBP or BMP.');
      return;
    }

    if (selected.size > MAX_FILE_SIZE) {
      setError('Image exceeds the 10 MB limit. Please choose a smaller file.');
      return;
    }

    if (preview) URL.revokeObjectURL(preview);
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setVisualTab('overlay');
  }

  async function inspectImage() {
    if (!file) {
      setError('Choose an image before starting an inspection.');
      return;
    }

    if (apiStatus !== 'online') {
      setError('The AI backend is unavailable. Start the FastAPI server on port 8000 and retry.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const body = new FormData();
      body.append('file', file);

      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body
      });

      const payload = await response.json();

      if (!response.ok || payload.success === false) {
        throw new Error(payload.detail || payload.message || 'The inspection request failed.');
      }

      const prediction = payload.result ?? payload.data ?? payload;
      setResult(prediction);

      const record = {
        id: `${Date.now()}`,
        filename: prediction.filename || file.name,
        prediction: prediction.prediction || (prediction.is_anomaly ? 'anomaly' : 'normal'),
        score: prediction.image_score ?? prediction.score,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setHistory(previous => [record, ...previous].slice(0, 8));
      setView('dashboard');
    } catch (err) {
      setError(err.message || 'Could not connect to the VisionGuard AI API.');
    } finally {
      setLoading(false);
    }
  }

  function resetInspection() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview('');
    setResult(null);
    setError('');
    setVisualTab('overlay');
    if (inputRef.current) inputRef.current.value = '';
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  }

  function downloadImage(src, name) {
    if (!src) return;
    const link = document.createElement('a');
    link.href = src;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  const prediction = String(result?.prediction || (result?.is_anomaly ? 'anomaly' : 'normal')).toLowerCase();
  const isAnomaly = result ? (result.is_anomaly ?? prediction.includes('anomaly')) : false;
  const score = result?.image_score ?? result?.score;
  const threshold = result?.threshold ?? result?.image_threshold ?? modelInfo?.image_threshold;
  const imageTabs = [
    { id: 'original', label: 'Original', src: preview },
    { id: 'heatmap', label: 'Heatmap', src: imageSource(result?.heatmap || result?.anomaly_map) },
    { id: 'overlay', label: 'Overlay', src: imageSource(result?.overlay) },
    { id: 'mask', label: 'Defect mask', src: imageSource(result?.mask || result?.defect_mask) }
  ];

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-mark"><ScanSearch size={23} strokeWidth={2.1} /></div>
          <div>
            <div className="brand-name">VisionGuard<span> AI</span></div>
            <div className="brand-subtitle">INDUSTRIAL VISION LAB</div>
          </div>
          <button className="icon-button mobile-close" onClick={() => setMobileNav(false)} aria-label="Close menu"><X size={19} /></button>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav className="side-nav">
          <button className={`nav-link ${view === 'dashboard' ? 'active' : ''}`} onClick={() => { setView('dashboard'); setMobileNav(false); }}>
            <LayoutDashboard size={18} /><span>Inspection studio</span><span className="nav-dot" />
          </button>
          <button className={`nav-link ${view === 'history' ? 'active' : ''}`} onClick={() => { setView('history'); setMobileNav(false); }}>
            <Clock3 size={18} /><span>Recent inspections</span>
            {history.length > 0 && <span className="nav-count">{history.length}</span>}
          </button>
        </nav>

        <div className="nav-label model-label">SYSTEM</div>
        <div className="system-card">
          <div className="system-card-icon"><Cpu size={18} /></div>
          <div className="system-card-copy">
            <strong>PatchCore-style v2</strong>
            <span>ResNet18 · Feature memory</span>
          </div>
          <span className={`status-dot ${apiStatus === 'online' ? 'dot-online' : apiStatus === 'offline' ? 'dot-offline' : ''}`} />
        </div>

        <div className="sidebar-spacer" />
        <div className="sidebar-note">
          <div className="note-icon"><ShieldCheck size={19} /></div>
          <strong>Built for visual quality</strong>
          <p>Detect unusual patterns in product imagery with pretrained visual features.</p>
          <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">Explore API docs <ArrowUpRight size={13} /></a>
        </div>
        <div className="sidebar-footer"><span className="footer-orb" /> Research preview <span>v1.0.0</span></div>
      </aside>

      {mobileNav && <button className="mobile-backdrop" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" onClick={() => setMobileNav(true)} aria-label="Open menu"><Menu size={20} /></button>
            <div className="breadcrumb">VisionGuard <span>/</span> <strong>{view === 'dashboard' ? 'Inspection studio' : 'Recent inspections'}</strong></div>
          </div>
          <div className="topbar-right">
            <div className={`api-pill ${apiStatus}`}><span className="status-dot" />{apiStatus === 'online' ? 'API connected' : apiStatus === 'checking' ? 'Connecting…' : 'API offline'}</div>
            <div className="avatar">VG</div>
          </div>
        </header>

        <div className="page-content">
          {view === 'dashboard' ? (
            <>
              <section className="welcome-row">
                <div>
                  <div className="eyebrow"><span className="eyebrow-line" /> COMPUTER VISION · ANOMALY DETECTION</div>
                  <h1>Inspection <span>studio.</span></h1>
                  <p className="page-subtitle">See beyond the surface. Analyze product images for visual anomalies.</p>
                </div>
                <div className="welcome-actions">
                  <a className="text-action" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer"><CircleHelp size={16} /> API documentation <ArrowUpRight size={14} /></a>
                </div>
              </section>

              <section className="stats-grid">
                <article className="stat-card">
                  <div className="stat-top"><span>FEATURE EXTRACTOR</span><div className="stat-icon indigo"><Fingerprint size={18} /></div></div>
                  <div className="stat-value">ResNet18</div>
                  <div className="stat-foot"><span className="stat-indicator" /> Pretrained visual backbone</div>
                </article>
                <article className="stat-card">
                  <div className="stat-top"><span>MEMORY BANK</span><div className="stat-icon teal"><Microscope size={18} /></div></div>
                  <div className="stat-value">{(modelInfo?.memory_patch_count ?? 5000).toLocaleString()} <small>patches</small></div>
                  <div className="stat-foot"><span className="stat-indicator" /> Reference feature descriptors</div>
                </article>
                <article className="stat-card">
                  <div className="stat-top"><span>TRAINING REFERENCE</span><div className="stat-icon amber"><FlaskConical size={18} /></div></div>
                  <div className="stat-value">{modelInfo?.training_images ?? 48} <small>images</small></div>
                  <div className="stat-foot"><span className="stat-indicator" /> Normal samples · bottle category</div>
                </article>
                <article className="stat-card">
                  <div className="stat-top"><span>INFERENCE DEVICE</span><div className="stat-icon blue"><Zap size={18} /></div></div>
                  <div className="stat-value">{modelInfo?.device || 'CPU'}</div>
                  <div className="stat-foot"><span className="stat-indicator" /> Local model inference</div>
                </article>
              </section>

              <section className="workspace-grid">
                <div className="panel upload-panel">
                  <div className="panel-heading">
                    <div>
                      <div className="section-kicker">01 / INPUT</div>
                      <h2>Product inspection</h2>
                      <p>Upload an image to begin the analysis.</p>
                    </div>
                    <div className="heading-icon"><ImageIcon size={19} /></div>
                  </div>

                  <input
                    ref={inputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/webp,image/bmp"
                    className="visually-hidden"
                    onChange={event => selectFile(event.target.files?.[0])}
                  />

                  {!preview ? (
                    <button
                      type="button"
                      className={`dropzone ${dragging ? 'dragging' : ''}`}
                      onClick={() => inputRef.current?.click()}
                      onDragOver={event => { event.preventDefault(); setDragging(true); }}
                      onDragLeave={() => setDragging(false)}
                      onDrop={handleDrop}
                    >
                      <div className="upload-graphic"><div className="upload-graphic-inner"><Upload size={23} /></div><span className="upload-spark spark-one" /><span className="upload-spark spark-two" /></div>
                      <strong>Drop your image here</strong>
                      <span className="dropzone-caption">or click to browse your files</span>
                      <span className="file-types"><FileImage size={14} /> PNG, JPG, WEBP, BMP <i /> Max 10 MB</span>
                    </button>
                  ) : (
                    <div className="selected-image">
                      <img src={preview} alt="Selected product for inspection" />
                      <div className="selected-image-footer">
                        <div className="selected-file-icon"><FileImage size={17} /></div>
                        <div className="selected-file-info"><strong title={file?.name}>{file?.name}</strong><span>{file ? (file.size / 1024).toFixed(1) + ' KB' : ''}</span></div>
                        <button className="icon-button remove-file" onClick={resetInspection} aria-label="Remove selected image"><X size={17} /></button>
                      </div>
                    </div>
                  )}

                  {error && <div className="error-message"><AlertCircle size={16} /><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={14} /></button></div>}

                  <div className="inspection-options">
                    <div className="option-copy"><Shield size={16} /><span>Analysis engine</span></div>
                    <div className="engine-tag">PatchCore-style <ChevronDown size={13} /></div>
                  </div>
                  <div className="privacy-note"><ShieldCheck size={15} /><span>Images are sent to your local API for analysis.</span></div>

                  <button className="primary-button" disabled={!file || loading || apiStatus !== 'online'} onClick={inspectImage}>
                    {loading ? <><LoaderCircle size={18} className="spin" /> Analyzing image…</> : <><Sparkles size={17} /> Run AI inspection <ArrowRight size={17} /></>}
                  </button>
                  {apiStatus === 'offline' && <p className="backend-hint">Start the FastAPI server to enable inspections.</p>}
                </div>

                <div className="panel results-panel">
                  <div className="panel-heading">
                    <div>
                      <div className="section-kicker">02 / ANALYSIS</div>
                      <h2>Inspection results</h2>
                      <p>Visual evidence and anomaly scoring.</p>
                    </div>
                    <div className="result-state-icon">{result ? (isAnomaly ? <AlertTriangle size={19} /> : <CheckCircle2 size={19} />) : <Activity size={19} />}</div>
                  </div>

                  {!result ? (
                    <div className="empty-results">
                      <div className="empty-visual">
                        <div className="empty-circle circle-back" /><div className="empty-circle circle-front" />
                        <div className="empty-scan"><ScanSearch size={31} strokeWidth={1.4} /></div>
                        <span className="empty-point point-a" /><span className="empty-point point-b" /><span className="empty-point point-c" />
                      </div>
                      <h3>Waiting for an image</h3>
                      <p>Your anomaly map and inspection score will appear here after analysis.</p>
                      <div className="empty-steps"><span><i>1</i> Upload image</span><ArrowRight size={13} /><span><i>2</i> Run inspection</span></div>
                    </div>
                  ) : (
                    <div className="results-content">
                      <div className={`prediction-banner ${isAnomaly ? 'anomaly-banner' : 'normal-banner'}`}>
                        <div className="prediction-symbol">{isAnomaly ? <AlertTriangle size={20} /> : <CheckCircle2 size={20} />}</div>
                        <div className="prediction-copy"><span>MODEL PREDICTION</span><strong>{isAnomaly ? 'Anomaly detected' : 'No anomaly detected'}</strong><small>{result.filename || file?.name || 'Inspected image'}</small></div>
                        <span className="prediction-chip">{isAnomaly ? 'REVIEW' : 'NORMAL'}</span>
                      </div>

                      <div className="score-card">
                        <div className="score-header"><span>IMAGE ANOMALY SCORE</span><span className="score-unit">Mean patch distance</span></div>
                        <div className="score-numbers"><strong>{formatScore(score)}</strong><span>Threshold <b>{formatScore(threshold)}</b></span></div>
                        <div className="score-track"><div className={`score-fill ${isAnomaly ? 'fill-anomaly' : 'fill-normal'}`} style={{ width: `${Math.min(100, Math.max(3, (Number(score) / Math.max(Number(threshold) * 2, 0.0001)) * 100))}%` }} /></div>
                        <div className="score-scale"><span>Lower anomaly score</span><span>Higher anomaly score</span></div>
                      </div>

                      <div className="visual-section">
                        <div className="visual-heading"><strong>Visual analysis</strong><span><Activity size={13} /> 224 × 224 analysis</span></div>
                        <div className="visual-tabs">
                          {imageTabs.map(tab => (
                            <button key={tab.id} disabled={tab.id !== 'original' && !tab.src} className={visualTab === tab.id ? 'visual-tab active' : 'visual-tab'} onClick={() => setVisualTab(tab.id)}>{tab.label}</button>
                          ))}
                        </div>
                        <div className="visual-image-wrap">
                          {imageTabs.find(tab => tab.id === visualTab)?.src ? (
                            <img src={imageTabs.find(tab => tab.id === visualTab).src} alt={`${visualTab} visualization`} />
                          ) : (
                            <div className="visual-unavailable"><ImageIcon size={24} /><span>This visualization was not returned by the API.</span></div>
                          )}
                          <span className="visual-corner-label">{visualTab.toUpperCase()}</span>
                          {imageTabs.find(tab => tab.id === visualTab)?.src && <button className="download-visual" title="Download visualization" onClick={() => downloadImage(imageTabs.find(tab => tab.id === visualTab).src, `visionguard-${visualTab}.png`)}><Download size={15} /></button>}
                        </div>
                      </div>

                      <div className="result-footnote"><ShieldCheck size={15} /><span>Model output is a research preview, not a certified quality-control decision.</span></div>
                      <button className="secondary-button" onClick={resetInspection}><RotateCcw size={15} /> Inspect another image</button>
                    </div>
                  )}
                </div>
              </section>

              <section className="bottom-grid">
                <div className="panel methodology-panel">
                  <div className="mini-panel-title"><div className="mini-icon"><Gauge size={17} /></div><div><h3>How the model works</h3><p>Pretrained features · nearest-neighbor anomaly scoring</p></div></div>
                  <div className="method-flow">
                    <div className="method-step"><span>01</span><strong>Extract</strong><small>ResNet18 features</small></div><ArrowRight size={15} />
                    <div className="method-step"><span>02</span><strong>Compare</strong><small>Memory bank distance</small></div><ArrowRight size={15} />
                    <div className="method-step"><span>03</span><strong>Localize</strong><small>Patch anomaly map</small></div>
                  </div>
                </div>
                <div className="panel connection-panel">
                  <div className="connection-top"><div className="mini-icon teal"><Activity size={17} /></div><span className={`connection-status ${apiStatus}`}>{apiStatus === 'online' ? 'OPERATIONAL' : apiStatus === 'checking' ? 'CHECKING' : 'DISCONNECTED'}</span></div>
                  <h3>Inference service</h3><p>FastAPI · Local CPU runtime</p>
                  <div className="connection-url"><span className={`status-dot ${apiStatus === 'online' ? 'dot-online' : apiStatus === 'offline' ? 'dot-offline' : ''}`} />127.0.0.1:8000 <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" aria-label="Open API documentation"><ArrowUpRight size={14} /></a></div>
                </div>
              </section>

              <footer className="page-footer"><span>VISIONGUARD AI <i /> COMPUTER VISION RESEARCH</span><span>Built for transparent visual inspection</span></footer>
            </>
          ) : (
            <>
              <section className="welcome-row history-welcome">
                <div><div className="eyebrow"><span className="eyebrow-line" /> WORKSPACE ACTIVITY</div><h1>Recent <span>inspections.</span></h1><p className="page-subtitle">Review results generated during this browser session.</p></div>
                <button className="secondary-button history-back" onClick={() => setView('dashboard')}><ArrowRight size={15} /> Back to studio</button>
              </section>
              <section className="panel history-panel">
                <div className="history-panel-heading"><div><h2>Inspection activity</h2><p>{history.length} {history.length === 1 ? 'inspection' : 'inspections'} in this session</p></div>{history.length > 0 && <button className="text-action" onClick={() => setHistory([])}><X size={15} /> Clear history</button>}</div>
                {history.length === 0 ? (
                  <div className="history-empty"><div className="empty-history-icon"><Clock3 size={24} /></div><h3>No inspections yet</h3><p>Your completed inspections will appear here during this session.</p><button className="primary-button" onClick={() => setView('dashboard')}>Start an inspection <ArrowRight size={16} /></button></div>
                ) : (
                  <div className="history-list">
                    {history.map(item => (
                      <div className="history-row" key={item.id}>
                        <div className="history-file-icon"><FileImage size={19} /></div>
                        <div className="history-file-copy"><strong>{item.filename}</strong><span>Completed at {item.timestamp}</span></div>
                        <div className="history-score"><span>ANOMALY SCORE</span><strong>{formatScore(item.score)}</strong></div>
                        <span className={`history-badge ${item.prediction.toLowerCase().includes('anomaly') ? 'history-anomaly' : 'history-normal'}`}>{item.prediction.toLowerCase().includes('anomaly') ? 'Anomaly' : 'Normal'}</span>
                      </div>
                    ))}
                  </div>
                )}
              </section>
              <footer className="page-footer"><span>VISIONGUARD AI <i /> COMPUTER VISION RESEARCH</span><span>Session-only activity · Not persisted to a database</span></footer>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
