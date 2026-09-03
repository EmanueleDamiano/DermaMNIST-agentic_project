"""Local DermaAgent interface."""

from __future__ import annotations

import base64
import binascii
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = PROJECT_ROOT / "main"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "platform"
sys.path.insert(0, str(MAIN_DIR))

import derma_agent as project  # noqa: E402


JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
ALLOWED_FILES = {"attribution.png", "prediction.json"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>DermaAgent</title>
  <style>
    :root { --ink:#13222f; --muted:#607080; --line:#dce7eb; --paper:#fff; --blue:#145b72; --aqua:#45b8a6; --pale:#edf7f5; --warn:#a85d12; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:linear-gradient(135deg,#e7f4f1 0,#f7f3ea 55%,#e8f1f4 100%); font:16px/1.5 Inter,Segoe UI,sans-serif; min-height:100vh; }
    main { width:min(1080px,calc(100% - 32px)); margin:36px auto; }
    header { display:flex; justify-content:space-between; align-items:end; gap:24px; margin-bottom:24px; }
    h1 { margin:0; font-size:clamp(32px,6vw,58px); letter-spacing:-.045em; line-height:1; }
    header p { color:var(--muted); max-width:520px; margin:8px 0 0; }
    .badge { white-space:nowrap; background:var(--ink); color:#fff; border-radius:999px; padding:8px 14px; font-size:13px; }
    .grid { display:grid; grid-template-columns:minmax(300px,.8fr) minmax(360px,1.2fr); gap:20px; }
    .card { background:rgba(255,255,255,.92); border:1px solid rgba(255,255,255,.9); border-radius:22px; box-shadow:0 18px 50px rgba(30,70,80,.12); padding:24px; }
    h2 { font-size:18px; margin:0 0 14px; }
    .drop { display:grid; place-items:center; min-height:230px; text-align:center; border:2px dashed #9ab9bc; border-radius:18px; background:var(--pale); padding:20px; cursor:pointer; transition:.2s; }
    .drop:hover,.drop.drag { border-color:var(--aqua); transform:translateY(-2px); }
    .drop input { display:none; }
    .drop strong { display:block; font-size:20px; margin-bottom:5px; }
    #preview { width:150px; height:150px; object-fit:cover; border-radius:14px; display:none; margin-bottom:14px; image-rendering:auto; }
    button { width:100%; border:0; border-radius:13px; color:#fff; background:var(--blue); padding:14px 18px; margin-top:15px; font-weight:700; font-size:15px; cursor:pointer; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .modes { display:grid; grid-template-columns:1fr 1fr; gap:9px; margin-bottom:15px; }
    .modes button { margin:0; background:#e8f0f2; color:var(--ink); padding:13px; text-align:left; }
    .modes button.active { color:#fff; background:var(--blue); }
    .modes strong,.modes span { display:block; }
    .modes span { font-size:11px; font-weight:400; opacity:.75; margin-top:3px; }
    .settings { display:none; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:15px; }
    .settings label { color:var(--muted); font-size:12px; }
    .settings input { width:100%; border:1px solid var(--line); border-radius:10px; padding:10px; margin-top:4px; color:var(--ink); background:#fff; }
    .notice { color:var(--muted); font-size:13px; margin:12px 2px 0; }
    .readiness { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 18px; }
    .ready { color:#17634f; background:#e4f4ee; border-radius:99px; padding:6px 10px; font-size:12px; }
    .ready.missing { color:#8a4d13; background:#fff0dc; }
    .bar { height:10px; background:#e6edef; border-radius:99px; overflow:hidden; margin:12px 0 18px; }
    .bar span { display:block; width:0; height:100%; background:linear-gradient(90deg,var(--blue),var(--aqua)); transition:width .35s; }
    .stages { display:grid; gap:9px; margin-bottom:18px; }
    .stage { display:flex; align-items:center; gap:11px; color:#87929c; font-size:14px; }
    .dot { width:12px; height:12px; border:2px solid #b8c5ca; border-radius:50%; }
    .stage.active { color:var(--ink); font-weight:700; }
    .stage.active .dot { border-color:var(--aqua); box-shadow:0 0 0 4px rgba(69,184,166,.15); }
    .stage.done .dot { border-color:var(--aqua); background:var(--aqua); }
    #result { display:none; border-top:1px solid var(--line); padding-top:20px; }
    .result-head { display:flex; justify-content:space-between; gap:14px; align-items:start; }
    .result-head h3 { margin:0; font-size:25px; line-height:1.15; }
    .confidence { color:var(--blue); font-weight:800; white-space:nowrap; }
    .result-grid { display:grid; grid-template-columns:190px 1fr; gap:18px; margin-top:18px; }
    #heatmap { width:190px; height:190px; object-fit:cover; border-radius:14px; background:#eee; }
    .prob { display:grid; grid-template-columns:1fr 45px; gap:7px; margin:0 0 8px; font-size:12px; }
    .prob-track { grid-column:1/-1; height:5px; background:#e8edef; border-radius:9px; overflow:hidden; }
    .prob-track i { display:block; height:100%; background:var(--aqua); }
    .error { color:#9b2d30; background:#fff0f0; border-radius:10px; padding:12px; display:none; }
    @media (max-width:800px) { header { align-items:start; flex-direction:column; } .grid { grid-template-columns:1fr; } .result-grid { grid-template-columns:1fr; } #heatmap { width:100%; height:auto; } }
  </style>
</head>
<body><main>
  <header><div><h1>DermaAgent</h1><p>Dermoscopic image classification with visible workflow progress.</p></div><span class="badge">Research and education only</span></header>
  <div class="readiness"><span class="ready" id="modelState">Checking model</span><span class="ready" id="dataState">Checking data</span></div>
  <div class="grid">
    <section class="card">
      <h2>Process</h2>
      <div class="modes"><button class="mode active" data-mode="saved"><strong>Test trained model</strong><span>Prediction and attribution</span></button><button class="mode" data-mode="full"><strong>Run full process</strong><span>Train, test, and predict</span></button></div>
      <div class="settings" id="settings"><label>Target accuracy<input id="target" type="number" value="0.70" min="0.01" max="1" step="0.01"></label><label>Maximum epochs<input id="epochs" type="number" value="3" min="1" max="100" step="1"></label></div>
      <h2>Input image</h2>
      <label class="drop" id="drop"><input id="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp"><div><img id="preview" alt="Selected image preview"><strong id="fileName">Choose an image</strong><span>PNG, JPG, WebP, or BMP · up to 10 MB</span></div></label>
      <button id="run" disabled>Test trained model</button>
      <p class="notice">Results are saved under output/platform.</p>
    </section>
    <section class="card">
      <h2>Workflow</h2>
      <div class="bar"><span id="bar"></span></div>
      <div class="stages" id="stages"></div>
      <p id="status">Waiting for an image.</p><p class="error" id="error"></p>
      <div id="result"><div class="result-head"><h3 id="prediction"></h3><span class="confidence" id="confidence"></span></div><p id="uncertain"></p><p id="runInfo"></p><div class="result-grid"><img id="heatmap" alt="Attribution image"><div id="probabilities"></div></div><p id="classInfo"></p></div>
    </section>
  </div>
</main>
<script>
const file=document.querySelector('#file'),run=document.querySelector('#run'),drop=document.querySelector('#drop'),preview=document.querySelector('#preview');
const statusText=document.querySelector('#status'),bar=document.querySelector('#bar'),error=document.querySelector('#error'),result=document.querySelector('#result');
const stageLists={saved:[[5,'Read input'],[10,'Prepare 28 × 28 RGB image'],[20,'Agent2 classification'],[45,'Agent3 attribution'],[70,'Agent3 class information'],[85,'Agent3 result']],full:[[3,'Create work-package files'],[8,'Check DermaMNIST'],[18,'Start Agent1 training'],[30,'Train model 2'],[42,'Train model 3'],[58,'Agent2 model selection'],[68,'Agent2 final test'],[82,'Agent3 input result'],[92,'Agent3 review'],[97,'Export results']]};
let selected=null,mode='saved',modelReady=true;
function renderStages(){document.querySelector('#stages').innerHTML=stageLists[mode].map(([at,name])=>`<div class="stage" data-at="${at}"><span class="dot"></span>${name}</div>`).join('');}
function refreshRun(){run.disabled=!selected||(mode==='saved'&&!modelReady);if(mode==='saved'&&!modelReady)statusText.textContent='No trained model. Select Run full process.';}
document.querySelectorAll('.mode').forEach(button=>button.onclick=()=>{mode=button.dataset.mode;document.querySelectorAll('.mode').forEach(item=>{item.classList.toggle('active',item===button);item.setAttribute('aria-pressed',item===button)});document.querySelector('#settings').style.display=mode==='full'?'grid':'none';run.textContent=mode==='full'?'Run full process':'Test trained model';result.style.display='none';error.style.display='none';error.textContent='';update(0,'Waiting for an image.');renderStages();refreshRun();});
function selectFile(f){ if(!f)return; selected=f; document.querySelector('#fileName').textContent=f.name; preview.src=URL.createObjectURL(f); preview.style.display='block'; refreshRun(); }
file.onchange=()=>selectFile(file.files[0]); drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag')}; drop.ondragleave=()=>drop.classList.remove('drag'); drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');selectFile(e.dataTransfer.files[0])};
function update(percent,message){ bar.style.width=percent+'%'; statusText.textContent=message; document.querySelectorAll('.stage').forEach(s=>{const at=+s.dataset.at;s.classList.toggle('done',percent>at);s.classList.toggle('active',percent===at)}); }
function readData(f){return new Promise((ok,bad)=>{const r=new FileReader();r.onload=()=>ok(r.result);r.onerror=bad;r.readAsDataURL(f)})}
function show(data,id,summary){ document.querySelector('#prediction').textContent=data.prediction; document.querySelector('#confidence').textContent=(data.confidence*100).toFixed(1)+'% confidence'; document.querySelector('#uncertain').textContent=data.uncertain?'Uncertainty flag: yes':'Uncertainty flag: no'; document.querySelector('#runInfo').textContent=summary?`Selected model: ${summary.selected_model} · Test accuracy: ${(summary.test_accuracy*100).toFixed(1)}% · Review: ${summary.review_decision}`:''; document.querySelector('#heatmap').src='/api/file?job='+encodeURIComponent(id)+'&name=attribution.png&t='+Date.now(); document.querySelector('#classInfo').textContent=data.class_information; const box=document.querySelector('#probabilities'); box.innerHTML=''; Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]).forEach(([name,value])=>{const row=document.createElement('div');row.className='prob';row.innerHTML='<span></span><b></b><div class="prob-track"><i></i></div>';row.children[0].textContent=name;row.children[1].textContent=(value*100).toFixed(1)+'%';row.querySelector('i').style.width=(value*100)+'%';box.appendChild(row)}); result.style.display='block'; }
async function poll(id){const response=await fetch('/api/status?id='+encodeURIComponent(id));const job=await response.json();update(job.percent,job.stage);if(job.status==='complete'){show(job.result,id,job.summary);modelReady=true;const model=document.querySelector('#modelState');model.textContent='Model ready';model.classList.remove('missing');refreshRun();return}if(job.status==='error'){throw Error(job.error)}setTimeout(()=>poll(id).catch(fail),500)}
function fail(e){error.textContent=e.message||String(e);error.style.display='block';statusText.textContent='Stopped';refreshRun()}
run.onclick=async()=>{try{run.disabled=true;result.style.display='none';error.style.display='none';error.textContent='';const target=+document.querySelector('#target').value,epochs=+document.querySelector('#epochs').value;if(mode==='full'&&(!(target>0&&target<=1)||!(Number.isInteger(epochs)&&epochs>=1&&epochs<=100)))throw Error('Use a target accuracy above 0 and up to 1, and 1 to 100 epochs.');update(1,'Uploading image');const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:selected.name,data:await readData(selected),mode,targetAccuracy:target,maxEpochs:epochs})});const body=await response.json();if(!response.ok)throw Error(body.error);poll(body.job).catch(fail)}catch(e){fail(e)}};
renderStages();
fetch('/api/health').then(response=>response.json()).then(health=>{modelReady=health.model;const model=document.querySelector('#modelState'),data=document.querySelector('#dataState');model.textContent=health.model?'Model ready':'Model not found';data.textContent=health.data?'Dataset ready':'Dataset downloads during full process';model.classList.toggle('missing',!health.model);data.classList.toggle('missing',!health.data);refreshRun();}).catch(()=>{document.querySelector('#modelState').textContent='Status unavailable';document.querySelector('#dataState').textContent='Status unavailable';});
</script></body></html>"""


def update_job(job_id: str, **values: object) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(values)


def system_health() -> dict[str, bool]:
    try:
        manifest = project.read_json(project.deliverable_path("D4.3"))
        model_path = project.saved_model_path(manifest)
        model_ready = model_path.is_file() and project.file_hash(model_path) == manifest["model_sha256"]
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        model_ready = False
    return {"model": model_ready, "data": project.DATA_PATH.is_file()}


def run_job(job_id: str, image_path: Path, output_dir: Path, mode: str, target_accuracy: float, max_epochs: int) -> None:
    try:
        def progress(percent: int, stage: str) -> None:
            update_job(job_id, percent=percent, stage=stage)

        with MODEL_LOCK:
            if mode == "full":
                completed = project.run_full_process(image_path, target_accuracy, max_epochs, output_dir, progress)
                result, summary = completed["prediction"], completed["summary"]
            else:
                config = project.load_config()
                manifest_path = project.deliverable_path("D4.3")
                if not manifest_path.is_file():
                    raise FileNotFoundError("No trained model was found. Choose Run full process.")
                result = project.run_user_image(
                    image_path,
                    project.read_json(manifest_path),
                    config,
                    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    output_dir,
                    progress,
                )
                summary = None
        update_job(job_id, status="complete", percent=100, stage="Result ready", result=result, summary=summary)
    except Exception as exc:
        update_job(job_id, status="error", stage="Stopped", error=str(exc))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload: object) -> None:
        self.send_bytes(status, "application/json; charset=utf-8", json.dumps(payload).encode())

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(200, "text/html; charset=utf-8", PAGE.encode())
            return
        if parsed.path == "/favicon.ico":
            self.send_bytes(204, "image/x-icon", b"")
            return
        if parsed.path == "/api/health":
            self.send_json(200, system_health())
            return
        query = parse_qs(parsed.query)
        if parsed.path == "/api/status":
            job_id = query.get("id", [""])[0]
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            self.send_json(200 if job else 404, job or {"error": "Job not found"})
            return
        if parsed.path == "/api/file":
            job_id = query.get("job", [""])[0]
            name = query.get("name", [""])[0]
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            if not job or name not in ALLOWED_FILES:
                self.send_json(404, {"error": "File not found"})
                return
            target = OUTPUT_ROOT / str(job["folder"]) / name
            if not target.is_file():
                self.send_json(404, {"error": "File not found"})
                return
            content_type = "image/png" if target.suffix == ".png" else "application/json"
            self.send_bytes(200, content_type, target.read_bytes())
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES * 2:
                raise ValueError("Upload is empty or too large")
            payload = json.loads(self.rfile.read(length))
            mode = str(payload.get("mode", "saved"))
            if mode not in {"saved", "full"}:
                raise ValueError("Choose Test trained model or Run full process")
            config = project.load_config()
            target_accuracy = float(payload.get("targetAccuracy", config["training"]["default_target_accuracy"]))
            max_epochs = int(payload.get("maxEpochs", config["training"]["default_max_epochs"]))
            if not 0 < target_accuracy <= 1:
                raise ValueError("Target accuracy must be greater than 0 and no more than 1")
            if not 1 <= max_epochs <= 100:
                raise ValueError("Maximum epochs must be from 1 to 100")
            if mode == "saved" and not system_health()["model"]:
                self.send_json(409, {"error": "No trained model was found. Choose Run full process."})
                return
            encoded = str(payload.get("data", ""))
            if "," not in encoded:
                raise ValueError("Invalid image upload")
            raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
            if not raw or len(raw) > MAX_UPLOAD_BYTES:
                raise ValueError("Image must be no more than 10 MB")
            suffix = Path(str(payload.get("name", "image.png"))).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                raise ValueError("Use a PNG, JPG, WebP, or BMP image")
            job_id = uuid.uuid4().hex
            folder = "full_process" if mode == "full" else "final_output"
            output_dir = OUTPUT_ROOT / folder
            output_dir.mkdir(parents=True, exist_ok=True)
            image_path = output_dir / ("input" + suffix)
            image_path.write_bytes(raw)
            with JOBS_LOCK:
                JOBS[job_id] = {"status": "running", "percent": 1, "stage": "Upload received", "mode": mode, "folder": folder}
            threading.Thread(target=run_job, args=(job_id, image_path, output_dir, mode, target_accuracy, max_epochs), daemon=True).start()
            self.send_json(202, {"job": job_id})
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error) as exc:
            self.send_json(400, {"error": str(exc)})


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("DermaAgent platform: http://127.0.0.1:8000", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
