"""PoseNeRF Studio: local Three.js viewer plus a bounded, isolated render worker.

This intentionally coexists with ``posenerf.dashboard``.  Studio serves only
127.0.0.1, never writes experiment artifacts, and has one serialized GPU job.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import threading
import time
from dataclasses import fields
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from .config import ExperimentConfig
from .checkpoints import discover_checkpoints, resolve_checkpoint
from .dashboard import _payload, _runs, _safe_run
from .model import HierarchicalNeRF, get_ray_directions, render_image
from .navigation import presets
from .utils import get_device, safe_torch_load


QUALITY = {"fast": (80, 60, 16, 16), "balanced": (160, 120, 48, 48), "quality": (320, 240, 48, 48)}
CV_TO_THREE = np.diag([1.0, -1.0, -1.0])
BUILD_ID = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
IMPORT_MAP = '<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js"}}</script>'
STUDIO_PATCH = r'''<script>(async()=>{const $=s=>document.querySelector(s),api='/api/checkpoints?name=';let selected='';const label=document.createElement('label');label.id='renderCheckpointLabel';label.innerHTML='Rendering checkpoint<select id="renderCheckpoint"></select><small id="checkpointMeta"></small>';$('#step').parentElement.after(label);async function fill(){let run=$('#run').value;if(!run){setTimeout(fill,75);return}let x=await fetch(api+encodeURIComponent(run)).then(r=>r.json()),s=$('#renderCheckpoint');s.innerHTML='';for(let c of x){let text=`${c.kind.toUpperCase()} · model step ${c.step} · ${c.architecture} · ${c.id}`;let o=new Option(text,c.id);o.selected=c.default;s.add(o)}selected=s.value;meta()}function meta(){let o=$('#renderCheckpoint').selectedOptions[0];$('#checkpointMeta').textContent=o?`Available for fresh rendering. ${o.text}`:'No renderable checkpoint';selected=$('#renderCheckpoint').value}$('#renderCheckpoint').onchange=meta;$('#run').addEventListener('change',()=>setTimeout(fill,0));const nativeFetch=window.fetch;window.fetch=async function(input,init){if(String(input).endsWith('/api/render')&&init&&init.body){let body=JSON.parse(init.body);body.checkpoint_id=selected;init={...init,body:JSON.stringify(body)}}return nativeFetch(input,init)};await fill()})()</script>'''
STUDIO_PATCH = STUDIO_PATCH.replace("for(let c of x)", "for(let c of (Array.isArray(x)?x:[]))")
STUDIO_NAV_PATCH = r'''<script>(function wait(){if(!window.PoseNeRFStudio||!document.querySelector('#run').value){return setTimeout(wait,80)}const $=s=>document.querySelector(s),bridge=window.PoseNeRFStudio,Matrix4=bridge.camera.matrix.constructor,Vector3=bridge.camera.position.constructor,box=document.createElement('section');box.innerHTML='<hr><b>Camera Navigation</b><label>Training camera<select id="trainPreset"></select></label><label>Validation camera<select id="valPreset"></select></label><button id="jumpPreset">Jump to selected camera</button><button id="frameScene">Frame scene</button><button id="centerScene">Look at scene center</button><hr><b>Bookmarks</b><input id="bookmarkName" placeholder="Bookmark name"><button id="saveBookmark">Save current</button><select id="bookmarks"></select><button id="restoreBookmark">Restore</button><button id="renameBookmark">Rename selected</button><button id="deleteBookmark">Delete</button>';$('#run').parentElement.parentElement.append(box);let ps=[],selectedPresetId='';const key=()=>`posenerf-studio-bookmarks:${$('#run').value}`;function pose(){bridge.camera.updateMatrixWorld(true);return bridge.camera.matrixWorld.elements.slice()}function valid(a){return Array.isArray(a)&&a.length===16&&a.every(Number.isFinite)}function loadBookmarks(){let parsed=[];try{let raw=JSON.parse(localStorage.getItem(key())||'[]');parsed=Array.isArray(raw)?raw.filter(v=>v&&typeof v.name==='string'&&v.name.trim()&&valid(v.pose)):[]}catch{}let s=$('#bookmarks');s.innerHTML='';parsed.forEach((v,i)=>s.add(new Option(v.name,i)));return parsed}function setPose(p){let m=new Matrix4().fromArray(p);bridge.camera.matrix.copy(m);bridge.camera.matrix.decompose(bridge.camera.position,bridge.camera.quaternion,bridge.camera.scale);bridge.camera.updateMatrixWorld(true);bridge.controls.target.copy(bridge.camera.position).add(new Vector3(0,0,-1).applyQuaternion(bridge.camera.quaternion));bridge.presetLocked=true}function mark(id){selectedPresetId=id}async function fill(){let response=await fetch('/api/presets?name='+encodeURIComponent($('#run').value));ps=response.ok?await response.json():[];for(let [id,kind] of [['trainPreset','training'],['valPreset','validation']]){let s=$('#'+id);s.innerHTML='';ps.filter(p=>p.kind===kind).forEach(p=>s.add(new Option(p.frame,p.id)))}selectedPresetId=$('#valPreset').value||$('#trainPreset').value||'';loadBookmarks()}$('#trainPreset').onchange=e=>mark(e.target.value);$('#valPreset').onchange=e=>mark(e.target.value);$('#jumpPreset').onclick=()=>{let p=ps.find(x=>x.id===selectedPresetId);if(!p)return;let cv=new Matrix4().fromArray(p.pose.flat()).transpose();setPose(cv.multiply(new Matrix4().makeScale(1,-1,-1)).toArray())};function sceneBounds(){let attr=bridge.pred&&bridge.pred.geometry&&bridge.pred.geometry.attributes.position;if(!attr||!attr.array.length)return null;let lo=new Vector3(Infinity,Infinity,Infinity),hi=new Vector3(-Infinity,-Infinity,-Infinity),a=attr.array;for(let i=0;i<a.length;i+=3){lo.min(new Vector3(a[i],a[i+1],a[i+2]));hi.max(new Vector3(a[i],a[i+1],a[i+2]))}return {center:lo.clone().add(hi).multiplyScalar(.5),size:hi.clone().sub(lo).length()}}$('#frameScene').onclick=()=>{let b=sceneBounds();if(!b)return;bridge.presetLocked=false;bridge.controls.target.copy(b.center);bridge.camera.position.copy(b.center).add(new Vector3(b.size,b.size,b.size));bridge.controls.update();bridge.camera.updateMatrixWorld(true)};$('#centerScene').onclick=()=>{let b=sceneBounds();if(b){bridge.presetLocked=false;bridge.controls.target.copy(b.center);bridge.controls.update();bridge.camera.updateMatrixWorld(true)}};$('#saveBookmark').onclick=()=>{let n=$('#bookmarkName').value.trim();if(!n)return;let a=loadBookmarks();a.push({name:n,pose:pose()});localStorage.setItem(key(),JSON.stringify(a));loadBookmarks()};$('#restoreBookmark').onclick=()=>{let a=loadBookmarks(),x=a[Number($('#bookmarks').value)];if(x)setPose(x.pose)};$('#renameBookmark').onclick=()=>{let n=$('#bookmarkName').value.trim(),a=loadBookmarks(),i=Number($('#bookmarks').value);if(!n||!a[i])return;a[i].name=n;localStorage.setItem(key(),JSON.stringify(a));loadBookmarks();$('#bookmarks').value=String(i)};$('#deleteBookmark').onclick=()=>{let a=loadBookmarks();a.splice(Number($('#bookmarks').value),1);localStorage.setItem(key(),JSON.stringify(a));loadBookmarks()};$('#run').addEventListener('change',()=>setTimeout(fill,100));fill()})()</script>'''
STUDIO_LOCALIZATION_PATCH = r'''<script>(async()=>{const $=s=>document.querySelector(s),box=document.createElement('section');box.innerHTML='<hr><b>Localization Inspector</b><label>Example<select id="locExample"></select></label><label><input id="locGT" type="checkbox"> GT poses (evaluation only)</label><pre id="locMeta" class="pose-matrix"></pre><img id="locQuery"><img id="locMatch">';document.querySelector('.panel').append(box);let s=$('#locExample'),a=await fetch('/api/localization/examples').then(r=>r.json());a.forEach(x=>s.add(new Option(x.label,x.id)));async function load(){let x=await fetch('/api/localization?example='+s.value+'&evaluation='+$('#locGT').checked).then(r=>r.json());$('#locQuery').src=x.query_image||'';$('#locMatch').src=x.matches_image||'';$('#locMeta').textContent=JSON.stringify(x.metrics||x,null,2)}s.onchange=load;$('#locGT').onchange=load;if(s.value)load()})()</script>'''


def three_to_posenerf_c2w(matrix: list[list[float]]) -> np.ndarray:
    """Convert Three camera C2W (x-right, y-up, -z-forward) to OpenCV C2W."""
    # Three.Matrix4.toArray() is column-major; the browser sends four
    # consecutive columns to avoid an ambiguous flattened payload.
    camera = np.asarray(matrix, dtype=np.float32).T
    if camera.shape != (4, 4) or not np.isfinite(camera).all() or not np.allclose(camera[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("pose must be a finite 4x4 homogeneous camera-to-world matrix")
    rotation = camera[:3, :3] @ CV_TO_THREE
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or np.linalg.det(rotation) <= 0.0:
        raise ValueError("pose rotation must be a proper orthonormal transform")
    output = camera.copy(); output[:3, :3] = rotation
    return output


def _apply_config(value, source: dict):
    for item in fields(value):
        if item.name not in source:
            continue
        current = getattr(value, item.name)
        incoming = source[item.name]
        if hasattr(current, "__dataclass_fields__") and isinstance(incoming, dict):
            _apply_config(current, incoming)
        else:
            setattr(value, item.name, incoming)


def config_from_manifest(run: Path, quality: str) -> ExperimentConfig:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    config = ExperimentConfig(); _apply_config(config, manifest["config"])
    width, height, coarse, fine = QUALITY[quality]
    if width * 3 != height * 4 or 640 % width or 480 % height:
        raise ValueError("unsupported render resolution")
    config.scene.downsample = 640 // width
    # Benchmarked E3 FP32 inference: 2,048 rays improves warm latency over
    # 512 while remaining comfortably inside the 6 GB GPU budget.  This is a
    # chunking-only change, so deterministic render values are unchanged.
    config.model.render_chunk = min(config.model.render_chunk, 2048)
    config.train.num_coarse_samples, config.train.num_fine_samples = coarse, fine
    return config


def png_data_uri(array: np.ndarray, mode: str) -> str:
    image = Image.fromarray(array, mode=mode)
    buffer = io.BytesIO(); image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class RenderWorker:
    """One-job worker with a single checkpoint cache; safe beside training."""
    def __init__(self, root: Path):
        self.root, self.device, self.lock = root, get_device(), threading.Lock()
        self.cached_key = None; self.model: HierarchicalNeRF | None = None

    def render(self, name: str, pose: list[list[float]], quality: str, checkpoint_id: str | None = None) -> dict:
        if quality not in QUALITY:
            raise ValueError("quality must be fast, balanced, or quality")
        run = _safe_run(self.root, name)
        if run is None:
            raise ValueError("unknown experiment")
        choice = resolve_checkpoint(run, checkpoint_id, self.device)
        checkpoint = choice["path"]
        c2w = three_to_posenerf_c2w(pose)
        with self.lock, torch.inference_mode():
            config = config_from_manifest(run, quality)
            key = (checkpoint.resolve(), tuple(config.model.__dict__.values()))
            if key != self.cached_key:
                payload = safe_torch_load(checkpoint, self.device)
                model = HierarchicalNeRF(config).to(self.device)
                model.load_state_dict(payload["model_state_dict"], strict=True); model.eval()
                self.model, self.cached_key = model, key
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device); torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            rgb, depth = render_image(self.model, torch.from_numpy(c2w).to(self.device), get_ray_directions(config, self.device), config,
                                      chunk=config.model.render_chunk, num_coarse_samples=config.train.num_coarse_samples, num_fine_samples=config.train.num_fine_samples)
            if self.device.type == "cuda": torch.cuda.synchronize(self.device)
            elapsed = (time.perf_counter() - started) * 1000
            rgb_np = (rgb.detach().cpu().numpy() * 255).round().astype(np.uint8)
            depth_np = depth.detach().cpu().numpy()
            finite = np.isfinite(depth_np); lo, hi = (np.percentile(depth_np[finite], [2, 98]) if finite.any() else (0, 1))
            depth_png = (np.clip((depth_np - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            return {"checkpoint": str(checkpoint.relative_to(self.root.resolve())), "checkpoint_id": choice["id"], "checkpoint_step": choice["step"], "checkpoint_kind": choice["kind"], "quality": quality, "width": config.scene.width, "height": config.scene.height, "camera_pose": c2w.tolist(),
                    "rgb": png_data_uri(rgb_np, "RGB"), "depth": png_data_uri(depth_png, "L"), "latency_ms": elapsed,
                    "cuda_peak_mb": torch.cuda.max_memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else 0.0}


PAGE = r'''<!doctype html><meta charset="utf-8"><title>PoseNeRF Studio</title><style>html,body{margin:0;height:100%;background:#0c101b;color:#e8edf8;font:13px system-ui}.studio{height:100%;display:grid;grid-template-columns:260px minmax(0,1fr) 280px;grid-template-rows:minmax(0,1fr) 180px}.panel{background:#121a2a;border:1px solid #273958;padding:14px;overflow:auto}.view{position:relative;min-width:0}.view canvas{display:block;width:100%;height:100%}select,button,input{width:100%;margin:5px 0;padding:8px;background:#1c2a43;color:#eef;border:1px solid #435b85;border-radius:6px}button{cursor:pointer;background:#3569bd}.bottom{grid-column:1/4}.render{position:absolute;right:14px;bottom:14px;width:320px;background:#101828dd;padding:10px;border:1px solid #405882;border-radius:8px}.render img{width:100%;image-rendering:auto}.badge{color:#91e7b2;font-weight:bold}.muted{color:#9aabc6}pre{white-space:pre-wrap;word-break:break-all}.pose-matrix{max-height:115px;overflow:auto;font-size:10px}@media(max-width:900px){.studio{grid-template-columns:1fr;grid-template-rows:auto 60vh auto auto}.bottom{grid-column:auto}.view{grid-row:2}}</style><div class="studio"><aside class="panel"><h2>Scene Explorer</h2><label>Experiment<select id="run"></select></label><label>Checkpoint step<select id="step"></select></label><hr><b>Layers</b><label><input id="obs" type="checkbox" checked>Observed RGB-D reference</label><label><input id="pred" type="checkbox" checked>NeRF-predicted geometry</label><label><input id="cams" type="checkbox" checked>Camera frustums / trajectory</label><p class="muted">World coordinates. Geometry proxies are not meshes.</p></aside><main class="view"><div id="viewport"></div><div class="render"><b>Neural render</b><label>Quality<select id="quality"><option value="fast">Fast · 80×60 · 16+16</option><option value="balanced">Balanced · 160×120 · 48+48</option><option value="quality">Quality · 320×240 · 48+48</option></select></label><button id="render">Render current camera</button><span id="renderStatus" class="muted">Idle — one GPU job maximum.</span><pre id="renderMeta" class="pose-matrix"></pre><img id="rgb"><img id="depth"></div></main><aside class="panel"><h2>Inspector</h2><p id="identity"></p><label>Navigation<select id="nav"><option value="orbit">Orbit (drag / wheel)</option><option value="fly">Fly (WASD, Q/E)</option></select></label><label>Move speed<input id="speed" type="number" value="0.10" step="0.05" min="0.01" max="2"></label><button id="reset">Reset view</button><p id="pose" class="muted"></p><pre id="poseMatrix" class="pose-matrix"></pre><hr><b>Provenance</b><p id="provenance" class="muted"></p></aside><section class="panel bottom"><b id="status" class="badge">Loading</b><canvas id="timeline" width="1200" height="120"></canvas><details><summary>Raw telemetry</summary><pre id="raw"></pre></details></section></div><script type="module">import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js';import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js';const $=s=>document.querySelector(s);let scene,camera,renderer,controls,obs,pred,cams,active,fly=false,keys={};const cv=new THREE.Matrix4().makeScale(1,-1,-1);async function J(u){let r=await fetch(u);return r.ok?await r.json():null}function setup(){scene=new THREE.Scene();scene.background=new THREE.Color(0x0c101b);camera=new THREE.PerspectiveCamera(55,1,.01,20);camera.position.set(1,1,2);renderer=new THREE.WebGLRenderer({antialias:true});$('#viewport').append(renderer.domElement);controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,0,1);scene.add(new THREE.GridHelper(6,24,0x46648c,0x26384f),new THREE.AxesHelper(.5));resize();addEventListener('resize',resize);renderer.domElement.addEventListener('dblclick',()=>controls.target.set(0,0,0));addEventListener('keydown',e=>keys[e.key.toLowerCase()]=true);addEventListener('keyup',e=>keys[e.key.toLowerCase()]=false);tick()}function resize(){let r=$('#viewport').getBoundingClientRect();camera.aspect=r.width/r.height;camera.updateProjectionMatrix();renderer.setSize(r.width,r.height)}function points(data,colorDefault){let a=[],c=[];for(let p of data.points||[]){a.push(...p.position);let q=new THREE.Color(p.color||colorDefault);c.push(q.r,q.g,q.b)}let g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(a,3));g.setAttribute('color',new THREE.Float32BufferAttribute(c,3));return new THREE.Points(g,new THREE.PointsMaterial({size:.018,vertexColors:true}))}function frustums(data){let g=new THREE.Group();for(let f of data.frustums||[]){let v=[];for(let i=1;i<5;i++)v.push(...f[0],...f[i]);for(let i=1;i<5;i++)v.push(...f[i],...f[i%4+1]);let x=new THREE.BufferGeometry();x.setAttribute('position',new THREE.Float32BufferAttribute(v,3));g.add(new THREE.LineSegments(x,new THREE.LineBasicMaterial({color:0xffb94e})))}return g}function clean(o){if(o)scene.remove(o)}async function runs(){let a=await J('/api/runs'),s=$('#run');s.innerHTML='';for(let r of a)s.add(new Option(r.name+' · '+r.state,r.name));await loadRun()}async function loadRun(){let x=await J('/api/run?name='+$('#run').value),s=$('#step');s.innerHTML='';for(let p of x.snapshots)s.add(new Option('Step '+p.step,p.step));await load()}async function load(){let x=await J('/api/run?name='+$('#run').value+'&step='+$('#step').value);active=x;$('#identity').textContent=$('#run').value+' / step '+x.selected_step;$('#status').textContent=x.status.state==='running'?'LIVE TRAINING':'HISTORICAL REPLAY';$('#raw').textContent=JSON.stringify(x.events,null,2);$('#provenance').textContent='NeRF: '+((await J('/'+x.nerf_scene))?.provenance?.source||'missing');let O=await J('/'+x.observed_scene),N=await J('/'+x.nerf_scene);clean(obs);clean(pred);clean(cams);obs=points(O,'#527dbb');pred=points(N,'#91d66c');cams=frustums(N);scene.add(obs,pred,cams);drawTimeline(x.history||[]);layers()}function layers(){if(obs)obs.visible=$('#obs').checked;if(pred)pred.visible=$('#pred').checked;if(cams)cams.visible=$('#cams').checked}function drawTimeline(es){let c=$('#timeline'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);let a=es.filter(x=>Number.isFinite(x.loss)),p=es.filter(x=>Number.isFinite(x.validation_psnr));for(let [set,col,key] of [[a,'#94baff','loss'],[p,'#94e5a6','validation_psnr']]){if(set.length<2)continue;let lo=Math.min(...set.map(x=>x[key])),hi=Math.max(...set.map(x=>x[key]));g.strokeStyle=col;g.beginPath();set.forEach((x,i)=>{let X=i*c.width/(set.length-1),Y=110-95*(x[key]-lo)/Math.max(hi-lo,1e-6);i?g.lineTo(X,Y):g.moveTo(X,Y)});g.stroke()}}function pose(){camera.updateMatrixWorld();let m=camera.matrixWorld.clone().multiply(cv);return m.toArray().reduce((a,v,i)=>{a[Math.floor(i/4)].push(v);return a},[[],[],[],[]])}async function render(){let b=$('#render');b.disabled=true;$('#renderStatus').textContent='Rendering…';try{let r=await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run:$('#run').value,quality:$('#quality').value,pose:pose()})}),x=await r.json();if(!r.ok)throw Error(x.error);$('#rgb').src=x.rgb;$('#depth').src=x.depth;$('#renderMeta').textContent=JSON.stringify({checkpoint_id:x.checkpoint_id,checkpoint_step:x.checkpoint_step,checkpoint_kind:x.checkpoint_kind,width:x.width,height:x.height,quality:x.quality,camera_pose:x.camera_pose});$('#renderStatus').textContent=x.checkpoint+' · '+x.latency_ms.toFixed(0)+' ms · peak '+x.cuda_peak_mb.toFixed(0)+' MB'}catch(e){$('#renderStatus').textContent='Render failed: '+e.message;$('#renderMeta').textContent=''}finally{b.disabled=false}}function tick(){if(fly){let s=+$('#speed').value*(keys.shift?.2:1);if(keys.w)camera.translateZ(-s);if(keys.s)camera.translateZ(s);if(keys.a)camera.translateX(-s);if(keys.d)camera.translateX(s);if(keys.q)camera.translateY(-s);if(keys.e)camera.translateY(s)}controls.enabled=!fly;controls.update();$('#pose').textContent='C2W translation: '+camera.position.toArray().map(v=>v.toFixed(3)).join(', ');$('#poseMatrix').textContent=JSON.stringify(pose());renderer.render(scene,camera);requestAnimationFrame(tick)}$('#run').onchange=loadRun;$('#step').onchange=load;for(let x of['obs','pred','cams'])$('#'+x).onchange=layers;$('#nav').onchange=e=>fly=e.target.value==='fly';$('#reset').onclick=()=>{camera.position.set(1,1,2);controls.target.set(0,0,1)};$('#render').onclick=render;setup();runs();</script>'''


PAGE = PAGE.replace(
    "renderer.domElement.addEventListener('dblclick',()=>controls.target.set(0,0,0));",
    "renderer.domElement.addEventListener('dblclick',()=>controls.target.set(0,0,0));renderer.domElement.addEventListener('pointerdown',()=>{if(window.PoseNeRFStudio)window.PoseNeRFStudio.presetLocked=false});renderer.domElement.addEventListener('wheel',()=>{if(window.PoseNeRFStudio)window.PoseNeRFStudio.presetLocked=false},{passive:true});",
).replace(
    "controls.enabled=!fly;controls.update();",
    "controls.enabled=!fly;if(!window.PoseNeRFStudio?.presetLocked)controls.update();",
).replace(
    "$('#nav').onchange=e=>fly=e.target.value==='fly';$('#reset').onclick=()=>{camera.position.set(1,1,2);controls.target.set(0,0,1)};",
    "$('#nav').onchange=e=>{fly=e.target.value==='fly';controls.mouseButtons.LEFT=e.target.value==='pan'?THREE.MOUSE.PAN:THREE.MOUSE.ROTATE;if(window.PoseNeRFStudio)window.PoseNeRFStudio.presetLocked=false};$('#reset').onclick=()=>{if(window.PoseNeRFStudio)window.PoseNeRFStudio.presetLocked=false;camera.position.set(1,1,2);controls.target.set(0,0,1)};",
).replace(
    "let m=camera.matrixWorld.clone().multiply(cv);return",
    "let m=camera.matrixWorld.clone();return",
).replace(
    "pred=points(N,'#91d66c');cams=frustums(N);scene.add(obs,pred,cams);",
    "pred=points(N,'#91d66c');if(window.PoseNeRFStudio)window.PoseNeRFStudio.pred=pred;cams=frustums(N);scene.add(obs,pred,cams);",
).replace(
    '<p id="pose" class="muted"></p><pre id="poseMatrix"',
    '<p id="pose" class="muted"></p><p id="orbitTarget" class="muted"></p><pre id="poseMatrix"',
).replace(
    "$('#pose').textContent='C2W translation: '+camera.position.toArray().map(v=>v.toFixed(3)).join(', ');$('#poseMatrix')",
    "$('#pose').textContent='C2W translation: '+camera.position.toArray().map(v=>v.toFixed(3)).join(', ');$('#orbitTarget').textContent='Orbit target: '+controls.target.toArray().map(v=>v.toFixed(3)).join(', ')+' · distance: '+camera.position.distanceTo(controls.target).toFixed(3);$('#poseMatrix')",
).replace(
    '<option value="orbit">Orbit (drag / wheel)</option><option value="fly">Fly (WASD, Q/E)</option>',
    '<option value="orbit">Orbit (drag / wheel)</option><option value="pan">Pan (drag)</option><option value="fly">Fly (WASD, Q/E)</option>',
)


def serve(root: Path, host: str, port: int) -> None:
    worker = RenderWorker(root)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs): super().__init__(*args, directory=str(root), **kwargs)
        def reply(self, payload: object, status=HTTPStatus.OK):
            body = json.dumps(payload).encode(); self.send_response(status); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            u=urlparse(self.path)
            if u.path=="/api/localization/examples":
                demo=root / "github_evidence" / "live_demo" / "live_demo_report.json"
                self.reply([{"id":"live_demo","label":"E3 validation live demo"}] if demo.is_file() else []); return
            if u.path=="/api/localization":
                q=parse_qs(u.query); example=q.get("example",[""])[0]
                if example!="live_demo": self.reply({"error":"unknown localization example"},HTTPStatus.NOT_FOUND); return
                base=root / "github_evidence" / "live_demo"; report=base / "live_demo_report.json"
                if not report.is_file(): self.reply({"error":"live demo artifacts unavailable"},HTTPStatus.NOT_FOUND); return
                try:
                    value=json.loads(report.read_text(encoding="utf-8")); metrics=value["rgb_only_pnp"].copy(); metrics.update({"query_frame_id":value["query_frame_id"],"retrieval_frame_id":value["retrieval_frame_id"],"checkpoint":value["checkpoint"]})
                    if q.get("evaluation",["false"])[0]=="true": metrics["poses_evaluation_only"]=value.get("poses_evaluation_only",{})
                    self.reply({"query_image":"/github_evidence/live_demo/query_reference.png","matches_image":"/github_evidence/live_demo/e3_nerf_depth_pnp_matches.png","metrics":metrics})
                except (OSError,KeyError,json.JSONDecodeError) as exc: self.reply({"error":str(exc)},HTTPStatus.BAD_REQUEST)
                return
            if u.path=="/api/identity":
                self.reply({"build_id": BUILD_ID, "source": "posenerf/studio.py", "quality_profiles": QUALITY})
                return
            if u.path=="/api/runs": self.reply(_runs(root)); return
            if u.path=="/api/checkpoints":
                run=_safe_run(root,parse_qs(u.query).get("name",[""])[0])
                if run is None: self.reply({"error":"unknown run"},HTTPStatus.NOT_FOUND); return
                choices=discover_checkpoints(run, torch.device("cpu"))
                default=next((x["id"] for x in choices if x["kind"]=="best"), next((x["id"] for x in choices if x["kind"]=="final"), None))
                self.reply([{"id":x["id"],"kind":x["kind"],"step":x["step"],"architecture":f"{x['config'].get('model',{}).get('layers','?')}×{x['config'].get('model',{}).get('hidden','?')}","default":x["id"]==default} for x in choices]); return
            if u.path=="/api/presets":
                run=_safe_run(root,parse_qs(u.query).get("name",[""])[0])
                if run is None: self.reply({"error":"unknown run"},HTTPStatus.NOT_FOUND); return
                try: self.reply(presets(run, root.parent / "data" / "chess"))
                except (OSError,ValueError,json.JSONDecodeError) as exc: self.reply({"error":str(exc)},HTTPStatus.BAD_REQUEST)
                return
            if u.path=="/api/run":
                q=parse_qs(u.query); x=_payload(root,q.get("name",[""])[0],q.get("step",[None])[0]); self.reply(x or {"error":"unknown run"},HTTPStatus.OK if x else HTTPStatus.NOT_FOUND); return
            if u.path in ("/","/index.html"):
                self.send_response(200); self.send_header("Content-Type","text/html"); self.send_header("Cache-Control","no-store"); self.end_headers()
                bridge = f"console.info('[PoseNeRF] bridge registration started');setup();window.PoseNeRFStudio={{camera:camera,controls:controls,scene:scene,getCameraPose:()=>camera.matrixWorld.toArray(),getNavigationState:()=>({{camera:!!camera,controls:!!controls,scene:!!scene}})}};document.documentElement.dataset.posenerfBridge='ready';document.documentElement.dataset.posenerfBuild='{BUILD_ID}';console.info('[PoseNeRF] bridge registered',typeof window.PoseNeRFStudio);window.dispatchEvent(new Event('posenerf-navigation-ready'));runs();"
                html=PAGE.replace("<title>", IMPORT_MAP + "<title>").replace("setup();runs();", bridge)
                self.wfile.write((html + STUDIO_PATCH + STUDIO_NAV_PATCH + STUDIO_LOCALIZATION_PATCH).encode()); return
            super().do_GET()
        def do_POST(self):
            if urlparse(self.path).path!="/api/render": self.send_error(404); return
            try:
                size=int(self.headers.get("Content-Length","0"));
                if size<=0 or size>8192: raise ValueError("invalid request size")
                request=json.loads(self.rfile.read(size)); self.reply(worker.render(request["run"],request["pose"],request.get("quality","fast"),request.get("checkpoint_id")))
            except (KeyError, ValueError, json.JSONDecodeError) as exc: self.reply({"error":str(exc)},HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc: self.reply({"error":f"render worker error: {exc}"},HTTPStatus.SERVICE_UNAVAILABLE)
    print(f"PoseNeRF Studio: http://{host}:{port}"); ThreadingHTTPServer((host,port),Handler).serve_forever()


def main():
    p=argparse.ArgumentParser();p.add_argument("--outputs-dir",type=Path,default=Path("workspace/outputs"));p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8768);a=p.parse_args();serve(a.outputs_dir,a.host,a.port)
if __name__=="__main__": main()
