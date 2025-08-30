from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import pandas as pd
import joblib
import io
import json
import os
import subprocess
import signal
from pathlib import Path
import time
from datetime import datetime


MODEL_PATH = "best_model.joblib"
INFO_PATH = "api_info.json"

FEATURE_COLS = [
    "has_appointment",
    "has_observation",
    "has_encounter",
    "has_btg_access",
    "has_care_access",
    "num_btg_events",
    "num_care_events",
    "avg_time_between_events",
]

STREAM_SCRIPT = "stream_from_csv.py"
PID_FILE = Path(".stream_pid")
STREAM_LOG = Path("stream_stdout.log")            
INGEST_LOG = Path("csv_ingest_results.log")       
DEFAULT_CSV = "/Users/melisadzanovic/Documents/ML - Thesis/hcs-synthetic-data-generator-main/labeled_events_full-24h.csv"


_last_proc_count: Optional[int] = None
_last_proc_at: Optional[float] = None


model = joblib.load(MODEL_PATH)

if not os.path.exists(INFO_PATH):
    raise FileNotFoundError(f"Metadata file '{INFO_PATH}' not found.")
with open(INFO_PATH, "r") as f:
    MODEL_META = json.load(f)


app = FastAPI(
    title="Insider Threat Demo API – Melisa",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)


def _items_to_dicts(items: List["PredictItem"]):
    out = []
    for i in items:
        if hasattr(i, "model_dump"):
            out.append(i.model_dump())
        else:
            out.append(i.dict())
    return out

def _to_frame(items: List["PredictItem"]) -> pd.DataFrame:
    df = pd.DataFrame(_items_to_dicts(items))
    for col in ["has_appointment","has_observation","has_encounter","has_btg_access","has_care_access"]:
        df[col] = df[col].astype(int)
    df["num_btg_events"] = df["num_btg_events"].fillna(0).astype(int)
    df["num_care_events"] = df["num_care_events"].fillna(0).astype(int)
    df["avg_time_between_events"] = df["avg_time_between_events"].fillna(0.0).astype(float)
    return df[FEATURE_COLS]

def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def _parse_stream_tail(tail: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Parse last 'progress=XXXX/YYYY' and 'anomalies=ZZ' from the tail text.
    Returns (processed, total, last_batch_anoms).
    """
    processed = total = last_batch = None
    if not tail:
        return processed, total, last_batch
  
    import re
    prog_matches = re.findall(r"progress=(\d+)\s*/\s*(\d+)", tail)
    if prog_matches:
        p, t = prog_matches[-1]
        processed, total = int(p), int(t)
    
    anom_matches = re.findall(r"anomalies=(\d+)", tail)
    if anom_matches:
        last_batch = int(anom_matches[-1])
    return processed, total, last_batch

def _fmt_time_now() -> str:
    return datetime.now().strftime("%H.%M.%S")


HOME_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Insider Threat Demo – Melisa</title>
<style>
  :root{
    --bg:#f6f8fb; --fg:#0b1220; --muted:#5b667a; --card:#ffffff; --line:#e6eaf2; --accent:#1f6feb;
    --ok:#047857; --warn:#b45309;
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#0b1220; --fg:#e5e7eb; --muted:#9aa4b2; --card:#101828; --line:#1f2a37; --accent:#60a5fa; }
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;background:var(--bg);color:var(--fg);line-height:1.5}
  header{padding:28px 28px 10px;background:linear-gradient(120deg,rgba(255,255,255,.6),rgba(255,255,255,0));border-bottom:1px solid var(--line)}
  .wrap{max-width:1200px;margin:0 auto;padding:24px}
  h1{margin:0 0 4px;font-size:28px}
  p.lead{margin:0;color:var(--muted)}
  .grid{display:grid;gap:20px;grid-template-columns:repeat(12,minmax(0,1fr))}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
  .span-4{grid-column:span 4}
  .span-6{grid-column:span 6}
  .span-8{grid-column:span 8}
  .span-12{grid-column:span 12}
  .kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px}
  .kpi{border:1px solid var(--line);border-radius:12px;padding:12px;background:var(--card)}
  .kpi .label{font-size:12px;color:var(--muted)}
  .kpi .val{font-size:24px;font-weight:800;letter-spacing:0.2px;line-height:1.2;margin-top:2px}
  .row{display:grid;grid-template-columns:1fr auto;gap:12px}
  .progress{width:100%;height:10px;background:rgba(100,116,139,.2);border-radius:999px;overflow:hidden}
  .progress > div{height:100%;background:var(--accent);width:0%}
  .pill{display:inline-block;padding:4px 10px;border-radius:999px;background:rgba(31,111,235,.12);color:var(--accent);font-size:12px;font-weight:600;text-decoration:none}
  input[type="text"],input[type="number"]{padding:10px;border:1px solid var(--line);border-radius:10px;background:transparent;color:var(--fg);width:100%}
  textarea{padding:10px;border:1px solid var(--line);border-radius:10px;background:transparent;color:var(--fg);width:100%;min-height:180px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  button{appearance:none;border:0;background:var(--accent);color:#fff;padding:10px 14px;border-radius:10px;font-weight:600;cursor:pointer}
  button.secondary{background:#334155}
  pre{background:#0f172a;color:#e5e7eb;padding:14px;border-radius:10px;overflow:auto;max-height:320px;white-space:pre-wrap;word-break:break-word}
  .muted{color:var(--muted)}
  .ok{color:var(--ok);font-weight:600}
  .warn{color:var(--warn);font-weight:600}
  .kv{display:grid;grid-template-columns:200px 1fr;gap:8px}
  .kv div{padding:8px 0;border-bottom:1px dashed var(--line)}
  .section-h{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
  .spark{width:100%;height:44px}
  .controls{display:grid;grid-template-columns:1fr 120px 120px 120px 140px;gap:10px}
  .controls label{display:block;font-size:12px;color:var(--muted);margin-bottom:4px}
  .toolbar{display:flex;gap:10px;align-items:center}
  .toggle{display:inline-flex;align-items:center;gap:8px;font-size:12px}
  @media (max-width:1100px){
    .span-4,.span-6,.span-8,.span-12{grid-column:span 12}
    .kpis{grid-template-columns:repeat(2,minmax(0,1fr))}
    .controls{grid-template-columns:1fr 1fr 1fr}
  }
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>Insider Threat Demo API</h1>
    <p class="lead">Trained XGBoost anomaly detector. Upload data, stream events, monitor detections.</p>
  </div>
</header>

<main class="wrap">
  <!-- KPIs -->
  <section class="card span-12">
    <div class="section-h">
      <h2 style="margin:0;font-size:18px">Overview</h2>
      <div class="toolbar">
        <a class="pill" href="/reference">API Reference</a>
        <a class="pill" href="/metrics">/metrics</a>
        <a class="pill" href="/summary">/summary</a>
      </div>
    </div>
    <div class="kpis" id="kpis">
      <div class="kpi"><div class="label">Model</div><div class="val" id="m_name">-</div></div>
      <div class="kpi"><div class="label">Training samples</div><div class="val" id="m_rows">-</div></div>
      <div class="kpi"><div class="label">Precision / Recall / F1</div><div class="val" id="m_scores">-</div></div>
      <div class="kpi"><div class="label">Processed</div><div class="val" id="s_processed">-</div></div>
      <div class="kpi"><div class="label">Total anomalies</div><div class="val" id="s_anoms">-</div></div>
      <div class="kpi"><div class="label">Anomaly rate</div><div class="val" id="s_rate">-</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 280px;gap:16px;margin-top:12px">
      <div>
        <div class="muted" style="margin-bottom:6px">Progress</div>
        <div class="progress"><div id="s_progbar"></div></div>
        <div class="muted" id="s_progtext" style="margin-top:4px">-</div>
      </div>
      <div>
        <div class="muted" style="margin-bottom:6px">Top anomaly probabilities (last 10)</div>
        <svg class="spark" id="spark" viewBox="0 0 100 44" preserveAspectRatio="none"></svg>
      </div>
    </div>
  </section>

  <!-- Status & Controls -->
  <div class="grid" style="margin-top:18px">
    <section class="card span-6">
      <div class="section-h"><h2 style="margin:0;font-size:18px">Live status</h2><a class="pill" href="/stream/status">/stream/status</a></div>
      <div class="kv">
        <div>Status</div><div id="st_status">-</div>
        <div>PID</div><div id="st_pid">-</div>
        <div>Last batch anomalies</div><div id="s_last_batch">-</div>
        <div>Last result (anomaly?)</div><div id="s_last_result">-</div>
        <div>Throughput (records/min)</div><div id="s_speed">-</div>
        <div>Max anomaly probability</div><div id="s_topprob">-</div>
        <div>Last update</div><div id="s_update">-</div>
      </div>
    </section>

    <section class="card span-6">
      <div class="section-h"><h2 style="margin:0;font-size:18px">Stream control</h2></div>
      <div class="controls">
        <div>
          <label>CSV path</label>
          <input type="text" id="in_csv" placeholder="/path/to/labeled_events_full-24h.csv"/>
        </div>
        <div>
          <label>Threshold</label>
          <input type="text" id="in_threshold" value="0.5"/>
        </div>
        <div>
          <label>Batch size</label>
          <input type="number" id="in_batch" value="500"/>
        </div>
        <div>
          <label>Sleep (s)</label>
          <input type="text" id="in_sleep" value="0.0"/>
        </div>
        <div>
          <label>Start offset</label>
          <input type="number" id="in_offset" value="0"/>
        </div>
      </div>
      <div class="toolbar" style="margin-top:12px">
        <button id="btnStart">Start</button>
        <button class="secondary" id="btnStop">Stop</button>
        <label class="toggle"><input type="checkbox" id="autoScroll" checked/> Auto-scroll logs</label>
      </div>
      <div style="margin-top:12px">
        <div class="muted" style="margin-bottom:6px">Last log lines</div>
        <pre id="logTail"></pre>
      </div>
    </section>
  </div>

  <!-- Predict JSON -->
  <section class="card span-12" style="margin-top:18px">
    <div class="section-h">
      <h2 style="margin:0;font-size:18px">Predict (JSON)</h2>
      <span class="pill">POST /predict</span>
    </div>
    <div class="row">
      <textarea id="jsonInput">[
  {
    "has_appointment": true,
    "has_observation": false,
    "has_encounter": true,
    "has_btg_access": false,
    "has_care_access": false,
    "num_btg_events": 0,
    "num_care_events": 0,
    "avg_time_between_events": 4.2
  }
]</textarea>
      <div>
        <div class="muted">Threshold</div>
        <input id="thInput" type="text" value="0.5"/>
        <div style="height:8px"></div>
        <button id="sendJson">Run prediction</button>
      </div>
    </div>
    <div style="margin-top:12px">
      <div class="muted">Result</div>
      <pre id="jsonOut"></pre>
    </div>
  </section>

  <!-- Predict CSV -->
  <section class="card span-12" style="margin-top:18px">
    <div class="section-h">
      <h2 style="margin:0;font-size:18px">Batch predict (CSV)</h2>
      <span class="pill">POST /predict_csv</span>
    </div>
    <form id="csvForm">
      <div class="row">
        <input type="file" id="csvFile" accept=".csv" required/>
        <div class="row" style="gap:8px">
          <div>
            <div class="muted">Threshold</div>
            <input type="text" id="csvThreshold" value="0.5" style="width:120px"/>
          </div>
          <div>
            <div class="muted">Top K</div>
            <input type="number" id="topK" min="1" step="1" value="20" style="width:120px"/>
          </div>
          <div style="display:flex;align-items:flex-end">
            <button type="submit" class="secondary">Upload & run</button>
          </div>
        </div>
      </div>
    </form>
    <div style="margin-top:12px">
      <div class="muted">Summary</div>
      <pre id="csvOut"></pre>
    </div>
  </section>
</main>

<script>
function parseFloatLocale(s, fallback){ if(s==null||s==="") return fallback; return parseFloat(String(s).replace(",", ".")); }

function sparkline(values){
  const svg = document.getElementById('spark');
  while(svg.firstChild) svg.removeChild(svg.firstChild);
  if(!values || !values.length){ return; }
  const n = values.length;
  const max = Math.max.apply(null, values.concat(1));
  const min = Math.min.apply(null, values.concat(0));
  const pad = 3;
  const w = 100, h = 44;
  const xs = values.map((_,i)=> pad + (i*(w-2*pad))/(n-1));
  const ys = values.map(v => h-pad - ((v-min)/(max-min||1))*(h-2*pad));
  let d = "M "+xs[0]+" "+ys[0];
  for(let i=1;i<n;i++) d += " L "+xs[i]+" "+ys[i];
  const path = document.createElementNS("http://www.w3.org/2000/svg","path");
  path.setAttribute("d", d);
  path.setAttribute("fill","none");
  path.setAttribute("stroke","currentColor");
  path.setAttribute("stroke-width","1.5");
  svg.appendChild(path);
}

async function loadSummaryAndStatus(){
  try{
    const [mr, sr, tr] = await Promise.all([fetch('/metrics'), fetch('/summary'), fetch('/stream/status')]);
    const m = await mr.json(); const s = await sr.json(); const t = await tr.json();

    // KPIs
    document.getElementById('m_name').textContent = m.model_name ?? '-';
    document.getElementById('m_rows').textContent = m.trained_on_rows?.toLocaleString?.() ?? '-';
    document.getElementById('m_scores').textContent = [m.precision, m.recall, m.f1].map(x => x ?? '-').join(' / ');
    const processed = t.processed ?? 0, total = t.total ?? 0;
    document.getElementById('s_processed').textContent = total ? processed.toLocaleString() : '-';
    document.getElementById('s_anoms').textContent = (s.anomalies ?? 0).toLocaleString();
    const ratePct = (s.share_anomaly ?? 0)*100;
    document.getElementById('s_rate').textContent = ratePct ? ratePct.toFixed(4) : '-';

    // Progress
    const progPct = (total>0)? Math.round((processed/total)*1000)/10 : 0;
    const bar = document.getElementById('s_progbar'); bar.style.width = (progPct>100?100:progPct) + '%';
    document.getElementById('s_progtext').textContent = total ? `${progPct}% (${processed.toLocaleString()} / ${total.toLocaleString()})` : '-';

    // Status
    document.getElementById('st_status').innerHTML = t.running ? '<span class="ok">RUNNING</span>' : '<span class="warn">STOPPED</span>';
    document.getElementById('st_pid').textContent = t.pid ?? '-';
    document.getElementById('s_last_batch').textContent = (t.last_batch_anoms ?? '-');
    document.getElementById('s_last_result').textContent = (t.last_batch_anoms!=null) ? (t.last_batch_anoms>0?'1 (anomaly present)':'0 (no anomaly)'):'-';
    document.getElementById('s_speed').textContent = t.rate_per_min ? t.rate_per_min.toLocaleString(undefined,{maximumFractionDigits:3}) : '-';
    document.getElementById('s_topprob').textContent = (s.top10_probs && s.top10_probs.length>0) ? s.top10_probs[0] : '-';
    document.getElementById('s_update').textContent = s.now || '-';

    // Logs
    const pre = document.getElementById('logTail');
    const oldLen = pre.textContent.length;
    pre.textContent = t.log_tail || '';
    const auto = document.getElementById('autoScroll').checked;
    if(auto && pre.textContent.length!==oldLen){ pre.scrollTop = pre.scrollHeight; }

    // Sparkline
    sparkline((s.top10_probs||[]).map(Number));
  }catch(e){ console.error(e); }
}

// JSON predict
document.getElementById('sendJson').addEventListener('click', async () => {
  const payloadRaw = document.getElementById('jsonInput').value;
  const th = parseFloatLocale(document.getElementById('thInput').value, 0.5);
  try{
    const payload = JSON.parse(payloadRaw);
    const r = await fetch('/predict?threshold='+th,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const j = await r.json();
    document.getElementById('jsonOut').textContent = JSON.stringify(j,null,2);
  }catch(e){ document.getElementById('jsonOut').textContent = 'Error: '+e; }
});

// CSV predict
document.getElementById('csvForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const f = document.getElementById('csvFile').files[0];
  if(!f) return;
  const th = parseFloatLocale(document.getElementById('csvThreshold').value, 0.5);
  const topK = parseInt(document.getElementById('topK').value || '20', 10);
  const fd = new FormData(); fd.append('file', f); fd.append('top_k', topK); fd.append('threshold', th);
  try{
    const r = await fetch('/predict_csv', {method:'POST', body: fd});
    const j = await r.json();
    document.getElementById('csvOut').textContent = JSON.stringify(j,null,2);
  }catch(e){ document.getElementById('csvOut').textContent = 'Error: '+e; }
});

// Stream buttons
document.getElementById('btnStart').addEventListener('click', async () => {
  const csv = document.getElementById('in_csv').value || '';
  const threshold = parseFloatLocale(document.getElementById('in_threshold').value, 0.5);
  const batch = parseInt(document.getElementById('in_batch').value || '0', 10);
  const sleep = parseFloatLocale(document.getElementById('in_sleep').value, 0.0);
  const start_offset = parseInt(document.getElementById('in_offset').value || '0', 10);
  const body = {};
  if(csv) body.csv = csv;
  if(!Number.isNaN(threshold)) body.threshold = threshold;
  if(batch>0) body.batch = batch;
  if(!Number.isNaN(sleep)) body.sleep = sleep;
  if(start_offset>0) body.start_offset = start_offset;
  await fetch('/stream/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  setTimeout(loadSummaryAndStatus, 300);
});
document.getElementById('btnStop').addEventListener('click', async () => {
  await fetch('/stream/stop',{method:'POST'});
  setTimeout(loadSummaryAndStatus, 300);
});

// poll
loadSummaryAndStatus();
setInterval(loadSummaryAndStatus, 2500);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return HTMLResponse(content=HOME_HTML)

@app.get("/reference", include_in_schema=False)
async def custom_reference_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="API Reference – Insider Threat Demo",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@latest/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@latest/swagger-ui.css",
        swagger_ui_parameters={
            "layout": "BaseLayout",
            "deepLinking": False,
            "displayOperationId": True,
            "defaultModelsExpandDepth": -1,
            "defaultModelExpandDepth": 0,
            "docExpansion": "none",
            "displayRequestDuration": True,
            "tryItOutEnabled": True
        },
    )


@app.get("/info")
def info():
    return MODEL_META

@app.get("/metrics")
def metrics():
    m = MODEL_META.get("metrics", {}) or {}
    return {
        "model_name": MODEL_META.get("model_name") or MODEL_META.get("model_class"),
        "trained_on_rows": MODEL_META.get("trained_on_rows"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "f1": m.get("f1") or m.get("f1_score"),
        "accuracy": m.get("accuracy"),
    }

@app.get("/summary")
def summary():
    """
    Reads csv_ingest_results.log and reports:
      - total rows seen
      - total anomalies
      - anomaly share
      - top10 highest anomaly probabilities
      - now: server time string for UI
    """
    path = str(INGEST_LOG)
    if not os.path.exists(path):
        return {"total": 0, "anomalies": 0, "share_anomaly": 0.0, "top10_probs": [], "now": _fmt_time_now()}
    total = 0
    anomalies = 0
    probs = []
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            total += 1
            p = float(rec.get("pred_prob_anomaly", 0.0))
            if rec.get("pred_label") == "Anomaly":
                anomalies += 1
                probs.append(p)
    probs.sort(reverse=True)
    top10 = [float(f"{x:.6f}") for x in probs[:10]]
    share = float(anomalies / total) if total else 0.0
    return {"total": total, "anomalies": anomalies, "share_anomaly": share, "top10_probs": top10, "now": _fmt_time_now()}


class PredictItem(BaseModel):
    has_appointment: bool = Field(...)
    has_observation: bool = Field(...)
    has_encounter: bool = Field(...)
    has_btg_access: bool = Field(...)
    has_care_access: bool = Field(...)
    num_btg_events: int = 0
    num_care_events: int = 0
    avg_time_between_events: float = 0.0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(items: List[PredictItem], threshold: float = 0.5):
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    X = _to_frame(items)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        y = (proba >= threshold).astype(int)
    else:
        proba = None
        y = model.predict(X)
    results = []
    for i, yi in enumerate(y):
        rec = {"pred_label": "Anomaly" if int(yi) == 1 else "Normal"}
        if proba is not None:
            rec["pred_prob_anomaly"] = float(proba[i])
        results.append(rec)
    return {"n": len(results), "threshold": threshold, "results": results}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), top_k: int = 20, threshold: float = 0.5):
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="file must be .csv")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not read CSV: {e}")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing columns: {missing}")
    for col in ["has_appointment","has_observation","has_encounter","has_btg_access","has_care_access"]:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif df[col].dtype == object:
            df[col] = (df[col].astype(str).str.strip().str.lower().map({"true":1,"false":0})).fillna(0).astype(int)
    df["avg_time_between_events"] = df["avg_time_between_events"].fillna(0.0)
    X = df[FEATURE_COLS].copy()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        y = (proba >= threshold).astype(int)
    else:
        proba = None
        y = model.predict(X)
    total = int(len(df))
    anomalies = int((y == 1).sum())
    share = float(anomalies / total) if total else 0.0
    id_cols = [c for c in ["patient_id","practitioner_id","period","period_start","period_end"] if c in df.columns]
    top_list = []
    if proba is not None:
        tmp = pd.DataFrame({"_proba": proba, "_label": y})
        if id_cols:
            tmp[id_cols] = df[id_cols].reset_index(drop=True)
        top = tmp.sort_values("_proba", ascending=False).head(top_k)
        for _, row in top.iterrows():
            rec = {
                "pred_prob_anomaly": float(row["_proba"]),
                "label": "Anomaly" if int(row["_label"]) == 1 else "Normal"
            }
            for k in id_cols:
                rec[k] = str(row[k])
            top_list.append(rec)
    return {
        "rows": total,
        "anomalies": anomalies,
        "share_anomaly": share,
        "threshold": threshold,
        "top": top_list
    }


class StreamStartBody(BaseModel):
    csv: Optional[str] = None
    threshold: Optional[float] = None
    batch: Optional[int] = None
    sleep: Optional[float] = None
    start_offset: Optional[int] = None

@app.get("/stream/status")
def stream_status():
    running = False
    pid = None
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            running = _pid_running(pid)
        except Exception:
            pid = None
            running = False
    last = ""
    if STREAM_LOG.exists():
        try:
            with open(STREAM_LOG, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(size - 7000, 0))
                last = f.read().decode(errors="ignore")[-3500:]
        except Exception:
            last = ""
    processed, total, last_batch = _parse_stream_tail(last)

   
    global _last_proc_count, _last_proc_at
    rate_per_min = None
    if processed is not None:
        now = time.time()
        if _last_proc_count is not None and _last_proc_at is not None:
            dp = processed - _last_proc_count
            dt = now - _last_proc_at
            if dt > 0:
                rate_per_min = max(0.0, (dp / dt) * 60.0)
        _last_proc_count = processed
        _last_proc_at = now

    return {
        "running": running,
        "pid": pid,
        "log_tail": last,
        "processed": processed,
        "total": total,
        "last_batch_anoms": last_batch,
        "rate_per_min": rate_per_min
    }

@app.post("/stream/start")
def stream_start(body: StreamStartBody):
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if _pid_running(pid):
                return {"running": True, "pid": pid, "message": "already running"}
        except Exception:
            pass
    csv = body.csv or DEFAULT_CSV
    args = ["python", STREAM_SCRIPT, "--csv", csv]
    if body.threshold is not None:
        args += ["--threshold", str(body.threshold)]
    if body.batch is not None:
        args += ["--batch", str(body.batch)]
    if body.sleep is not None:
        args += ["--sleep", str(body.sleep)]
    if body.start_offset is not None:
        args += ["--start_offset", str(body.start_offset)]
    logf = open(STREAM_LOG, "ab")
    p = subprocess.Popen(args, stdout=logf, stderr=logf)
    PID_FILE.write_text(str(p.pid))
    time.sleep(0.3)
    # reset speed state
    global _last_proc_count, _last_proc_at
    _last_proc_count = None
    _last_proc_at = None
    return {"running": True, "pid": p.pid, "args": args}

@app.post("/stream/stop")
def stream_stop():
    if not PID_FILE.exists():
        return {"running": False, "message": "not running"}
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        PID_FILE.unlink(missing_ok=True)
        return {"running": False, "message": "stale pidfile removed"}
    if _pid_running(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
        time.sleep(0.5)
    PID_FILE.unlink(missing_ok=True)
    return {"running": False, "stopped_pid": pid}
