"""Read-only, artifact-backed local dashboard for PoseNeRF experiments.

The server never opens checkpoints or CUDA. It only serves files atomically
published by the trainer, so it is safe to run beside an active experiment.
"""
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .telemetry import read_latest


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>PoseNeRF Live Viewer</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b1020;color:#e8edf7;font:14px Inter,system-ui,sans-serif}.app{min-height:100vh;display:grid;grid-template-columns:285px minmax(0,1fr)}aside{padding:24px 18px;background:#111a2d;border-right:1px solid #26344f;position:sticky;top:0;height:100vh}main{padding:28px;min-width:0}.brand{font-weight:800;letter-spacing:.08em;color:#a9c8ff}.subtitle,.muted{color:#9ba9c2}.control,label{display:block;margin:14px 0}select,button,input{font:inherit}select,button{width:100%;border:1px solid #3a4c70;border-radius:8px;background:#18253d;color:#edf3ff;padding:9px}button{cursor:pointer;background:#3467b8}button:hover{background:#4379d0}.status{font-weight:700;margin:18px 0}.live{color:#6ee7a1}.history{color:#ffc96b}.cards,.grid{display:grid;gap:14px}.cards{grid-template-columns:repeat(6,minmax(110px,1fr));margin:20px 0}.grid{grid-template-columns:minmax(0,1fr) minmax(380px,.82fr)}.card,.panel{border:1px solid #263b60;background:#111a2d;border-radius:12px;padding:15px;box-shadow:0 8px 28px #0002}.metric{font-size:23px;font-weight:750;margin-top:5px}.metric-label,h2{margin:0;color:#9ba9c2;font-size:12px;letter-spacing:.07em;text-transform:uppercase}h1{margin:3px 0 7px;font-size:28px}.panel h2{margin-bottom:12px}.preview{min-height:320px;background:#080d18;display:grid;place-items:center;border-radius:9px;overflow:auto}.preview img{max-width:100%;height:auto;display:block}.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:12px}.tag{padding:4px 7px;border-radius:99px;background:#1c2b47}.tag.pred{background:#354b22}.tag.obs{background:#243f62}.chart{width:100%;height:185px;background:#080d18;border-radius:8px}.scene{width:100%;height:min(67vh,760px);min-height:520px;background:#080d18;border-radius:9px;cursor:grab}.toggles{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}.toggles label{margin:0;padding:6px 8px;border-radius:7px;background:#18253d}.toggles input{vertical-align:middle}details{margin-top:14px}pre{max-height:300px;overflow:auto;padding:10px;background:#080d18;border-radius:8px;white-space:pre-wrap}.presets{display:grid;grid-template-columns:1fr 1fr;gap:8px}.note{font-size:12px;color:#9ba9c2;line-height:1.45}@media(max-width:1050px){.app{grid-template-columns:1fr}aside{height:auto;position:static;border-right:0;border-bottom:1px solid #26344f}.grid{grid-template-columns:1fr}.cards{grid-template-columns:repeat(3,1fr)}main{padding:18px}}@media(max-width:580px){.cards{grid-template-columns:repeat(2,1fr)}main{padding:12px}.scene{min-height:380px}}
</style></head><body><div class="app"><aside><div class="brand">POSENERF / LIVE</div><p class="subtitle">Artifact-backed reconstruction viewer</p><label class="control">Experiment<select id="run"></select></label><label class="control">Published checkpoint<select id="step"></select></label><button id="reload">Refresh artifacts</button><p id="mode" class="status"></p><div class="panel"><h2>3D presets</h2><div class="presets"><button data-preset="nerf">NeRF only</button><button data-preset="observed">Observed only</button><button data-preset="compare">Compare</button><button data-preset="cameras">Cameras</button></div></div><details><summary>Raw status JSON</summary><pre id="status"></pre></details></aside><main><header><h1>Neural reconstruction workspace</h1><p id="headline" class="subtitle">Loading published telemetry…</p></header><section class="cards"><div class="card"><div class="metric-label">Status</div><div id="m-state" class="metric">—</div></div><div class="card"><div class="metric-label">Step</div><div id="m-step" class="metric">—</div></div><div class="card"><div class="metric-label">Validation PSNR</div><div id="m-psnr" class="metric">—</div></div><div class="card"><div class="metric-label">Depth MAE</div><div id="m-depth" class="metric">—</div></div><div class="card"><div class="metric-label">Loss</div><div id="m-loss" class="metric">—</div></div><div class="card"><div class="metric-label">GPU reserved</div><div id="m-gpu" class="metric">—</div></div></section><section class="grid"><div class="panel"><h2>Validation reconstruction</h2><div class="legend"><span class="tag obs">Observed RGB</span><span class="tag pred">NeRF RGB</span><span class="tag">Predicted depth</span></div><div id="preview" class="preview muted">No published preview.</div><p class="note">This is the genuine trainer-published composite for the selected checkpoint. Separate ground-truth-depth and error-image artifacts were not historically published, so the viewer does not fabricate them.</p></div><div class="panel"><h2>Training analytics</h2><canvas id="chart" class="chart" width="680" height="185"></canvas><div class="legend"><span class="tag">loss</span><span class="tag pred">validation PSNR</span><span class="tag obs">depth MAE</span></div><p id="metrics-note" class="note"></p></div></section><section class="panel"><h2>Interactive 3D workspace</h2><div class="toggles"><label><input id="obs" type="checkbox" checked>Observed RGB-D reference</label><label><input id="pred" type="checkbox" checked>NeRF-predicted points</label><label><input id="frustum" type="checkbox" checked>Camera frustums</label><label><input id="trajectory" type="checkbox" checked>Trajectory</label><button id="reset" style="width:auto">Reset camera</button></div><canvas id="view" class="scene" width="1280" height="700"></canvas><p id="provenance" class="note"></p></section><section class="grid"><details class="panel"><summary>Run manifest</summary><pre id="manifest"></pre></details><details class="panel"><summary>Recent telemetry JSON</summary><pre id="events"></pre></details></section></main></div><script>
const $=s=>document.querySelector(s);let O={points:[]},N={points:[]},events=[],pan=[0,0],drag,yaw=0,pitch=0,zoom=4;
async function J(u){let r=await fetch(u,{cache:'no-store'});return r.ok?await r.json():null}const fmt=(v,n=2)=>Number.isFinite(v)?v.toFixed(n):'—';
async function reload(){let rs=await J('/api/runs')||[],s=$('#run'),old=s.value;s.innerHTML='';for(let r of rs){let o=new Option(r.name+' · '+r.state+(r.stage?' · '+r.stage:''),r.name);o.selected=r.name===old;s.add(o)}await loadRun()}
async function loadRun(follow=false){let old=$('#step').value,x=await J('/api/run?name='+encodeURIComponent($('#run').value));if(!x)return;let s=$('#step');s.innerHTML='';for(let a of x.snapshots){let o=new Option('Step '+a.step+(a.current?' · current':''),a.step);o.selected=String(a.step)===old||((!old||follow)&&a.current);s.add(o)}await load()}
function last(kind){return [...events].reverse().find(e=>e.event===kind)||{}}function img(p){return p?'<img alt="Published validation reconstruction" src="/'+p+'?t='+Date.now()+'">':'No preview was published for this checkpoint.'}
async function load(){let x=await J('/api/run?name='+encodeURIComponent($('#run').value)+'&step='+encodeURIComponent($('#step').value));if(!x)return;events=x.history||x.events||[];let t=last('train_step'),v=last('validation'),s=x.status,m=x.metrics||{};$('#mode').textContent=s.state==='running'?'LIVE RUNNING':'HISTORICAL REPLAY';$('#mode').className='status '+(s.state==='running'?'live':'history');$('#headline').textContent=$('#run').value+' · selected checkpoint '+(x.selected_step??'—');$('#status').textContent=JSON.stringify(s,null,2);$('#manifest').textContent=JSON.stringify(x.manifest||{},null,2);$('#events').textContent=JSON.stringify((x.events||[]),null,2);$('#preview').innerHTML=img(x.preview);$('#m-state').textContent=s.state||'—';$('#m-step').textContent=x.selected_step??s.step??'—';$('#m-psnr').textContent=fmt(m.average_psnr??v.validation_psnr)+' dB';$('#m-depth').textContent=fmt(m.average_depth_mae??v.validation_depth_mae,3)+' m';$('#m-loss').textContent=fmt(t.loss,4);$('#m-gpu').textContent=fmt(t.cuda_reserved_mb,0)+' MB';$('#metrics-note').textContent='LR '+fmt(t.lr,7)+' · CUDA allocated '+fmt(t.cuda_allocated_mb,1)+' MB · selected snapshot metrics use its published JSON.';O=await J('/'+x.observed_scene)||{points:[]};N=await J('/'+x.nerf_scene)||{points:[]};$('#provenance').textContent='Selected step '+(x.selected_step??'—')+'. Observed RGB-D Reference uses captured depth. NeRF-Predicted 3D Reconstruction: '+(N.provenance?.source||'not published')+'; checkpoint step '+(N.provenance?.checkpoint_step??'—')+'.';chart();draw()}
function chart(){let c=$('#chart'),g=c.getContext('2d'),w=c.width,h=c.height;g.clearRect(0,0,w,h);g.fillStyle='#080d18';g.fillRect(0,0,w,h);let sets=[['loss','#b6c4e0',e=>e.loss],['PSNR','#7ee5a5',e=>e.validation_psnr],['Depth','#79aaff',e=>e.validation_depth_mae]],valid=sets.map(([n,c,f])=>[n,c,events.map(e=>[e.step,f(e)]).filter(p=>Number.isFinite(p[1]))]);for(let [n,col,a] of valid){if(a.length<2)continue;let lo=Math.min(...a.map(p=>p[1])),hi=Math.max(...a.map(p=>p[1]));if(lo===hi)hi=lo+1;g.strokeStyle=col;g.lineWidth=2;g.beginPath();a.forEach(([step,y],i)=>{let x=26+(w-36)*i/(a.length-1),py=h-20-(h-38)*(y-lo)/(hi-lo);i?g.lineTo(x,py):g.moveTo(x,py)});g.stroke()}g.fillStyle='#9ba9c2';g.font='11px system-ui';g.fillText('Telemetry history (independent scales)',10,14)}
function draw(){let c=$('#view'),g=c.getContext('2d'),a=yaw*Math.PI/180,b=pitch*Math.PI/180,z=zoom,q=v=>{let x=v[0]*Math.cos(a)-v[2]*Math.sin(a),zz=v[0]*Math.sin(a)+v[2]*Math.cos(a),y=v[1]*Math.cos(b)-zz*Math.sin(b);return[c.width/2+pan[0]+x*140/z,c.height/2+pan[1]-y*140/z]},cloud=(s,on,alpha)=>{if(!on)return;g.globalAlpha=alpha;for(let v of(s.points||[])){let p=q(v.position);g.fillStyle=v.color;g.fillRect(p[0],p[1],2,2)}},line=(xs,color)=>{if(xs.length<2)return;g.strokeStyle=color;g.globalAlpha=.9;g.beginPath();xs.map(q).forEach((p,i)=>i?g.lineTo(...p):g.moveTo(...p));g.stroke()};g.clearRect(0,0,c.width,c.height);cloud(O,$('#obs').checked,.27);cloud(N,$('#pred').checked,1);if($('#trajectory').checked){line(O.cameras||[],'#79aaff');line(N.cameras||[],'#ffd06a')}if($('#frustum').checked)for(let s of[O,N])for(let f of(s.frustums||[])){let p=f.map(q);g.strokeStyle='#ffbd4a';g.globalAlpha=.75;g.beginPath();g.moveTo(...p[0]);for(let i=1;i<p.length;i++){g.lineTo(...p[i]);g.lineTo(...p[0])}g.stroke()}g.globalAlpha=1}
function preset(p){$('#obs').checked=p!=='nerf'&&p!=='cameras';$('#pred').checked=p!=='observed'&&p!=='cameras';$('#frustum').checked=p==='cameras'||p==='compare';$('#trajectory').checked=p==='cameras';draw()}for(let b of document.querySelectorAll('[data-preset]'))b.onclick=()=>preset(b.dataset.preset);for(let id of['obs','pred','frustum','trajectory'])$('#'+id).oninput=draw;$('#run').onchange=loadRun;$('#step').onchange=load;$('#reload').onclick=reload;$('#reset').onclick=()=>{pan=[0,0];yaw=0;pitch=0;zoom=4;draw()};$('#view').onpointerdown=e=>{drag=[e.clientX,e.clientY,e.shiftKey];e.target.setPointerCapture(e.pointerId)};$('#view').onpointermove=e=>{if(drag){let dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(drag[2]){yaw+=dx*.35;pitch=Math.max(-80,Math.min(80,pitch+dy*.35))}else{pan[0]+=dx;pan[1]+=dy}drag=[e.clientX,e.clientY,drag[2]];draw()}};$('#view').onpointerup=()=>drag=null;$('#view').onwheel=e=>{e.preventDefault();zoom=Math.max(1,Math.min(14,zoom*(e.deltaY>0?1.12:.89)));draw()};setInterval(async()=>{let x=await J('/api/run?name='+encodeURIComponent($('#run').value));if(x&&x.status.state==='running')await loadRun(true);else await load()},2500);reload();
</script></body></html>"""


def _safe_run(root: Path, name: str) -> Path | None:
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_dir() and (candidate / "live").is_dir() else None


def _manifest(run: Path) -> dict:
    try:
        return json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _snapshots(run: Path, status: dict) -> list[dict]:
    """Return only steps with both a preview and a predicted-scene artifact."""
    preview_dir, scene_dir = run / "visualizations" / "train_validation", run / "live"
    result = []
    for preview in preview_dir.glob("hierarchical_nerf_val_*.png") if preview_dir.is_dir() else ():
        try:
            step = int(preview.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        if (scene_dir / f"nerf_predicted_step_{step:05d}.json").is_file():
            result.append({"step": step, "current": step == status.get("step")})
    return sorted(result, key=lambda item: item["step"], reverse=True)


def _runs(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    result = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "live").is_dir():
            status = read_latest(child / "live") or {}
            result.append({"name": child.name, "state": status.get("state", "waiting"), "step": status.get("step"), "stage": _manifest(child).get("stage")})
    return result


def _relative(run_name: str, path: Path) -> str:
    return f"{run_name}/{path.as_posix()}"


def _events(run: Path, limit: int | None = None) -> list[dict]:
    try:
        items = [json.loads(line) for line in (run / "live" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError):
        return []
    return items[-limit:] if limit else items


def _snapshot_metrics(run: Path, step: int | None) -> dict:
    if step is None:
        return {}
    path = run / "visualizations" / "train_validation" / f"hierarchical_nerf_val_{step:05d}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _payload(root: Path, name: str, requested_step: str | None) -> dict | None:
    run = _safe_run(root, name)
    if run is None:
        return None
    status, snapshots = read_latest(run / "live") or {}, _snapshots(run, read_latest(run / "live") or {})
    try:
        selected = int(requested_step) if requested_step else int(status.get("step"))
    except (TypeError, ValueError):
        selected = None
    chosen = next((item for item in snapshots if item["step"] == selected), snapshots[0] if snapshots else None)
    if chosen:
        selected = chosen["step"]
        preview = Path("visualizations/train_validation") / f"hierarchical_nerf_val_{selected:05d}.png"
        nerf_scene = Path("live") / f"nerf_predicted_step_{selected:05d}.json"
    else:
        preview = Path(status["preview"]) if status.get("preview") else None
        nerf_scene = Path(status["nerf_scene"]) if status.get("nerf_scene") else None
    observed = Path(status["observed_scene"]) if status.get("observed_scene") else None
    history = _events(run)
    events = history[-20:]
    def published(path: Path | None) -> str | None:
        return _relative(name, path) if path and (run / path).is_file() else None
    return {"status": status, "manifest": _manifest(run), "events": events, "history": history[-600:], "metrics": _snapshot_metrics(run, selected), "snapshots": snapshots, "selected_step": selected,
            "preview": published(preview), "observed_scene": published(observed), "nerf_scene": published(nerf_scene)}


def serve(root: Path, host: str, port: int) -> None:
    class DashboardHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def _reply(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:
            request = urlparse(self.path)
            if request.path == "/api/runs":
                self._reply(_runs(root)); return
            if request.path == "/api/run":
                query = parse_qs(request.query)
                payload = _payload(root, query.get("name", [""])[0], query.get("step", [None])[0])
                self._reply(payload or {"error": "unknown run"}, HTTPStatus.OK if payload else HTTPStatus.NOT_FOUND); return
            if request.path in ("/", "/index.html"):
                self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(PAGE.encode("utf-8")); return
            super().do_GET()

    print(f"PoseNeRF unified viewer: http://{host}:{port}")
    ThreadingHTTPServer((host, port), DashboardHandler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only unified PoseNeRF artifact viewer.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("workspace/outputs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.outputs_dir, args.host, args.port)


if __name__ == "__main__":
    main()
