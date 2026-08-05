"""
rebuild_html.py — Regenerate the const D={...} data blob in the interactive HTML.

Reads the analysis outputs (embeddings, labels, centers, frames) and rebuilds
the JSON data blob that the HTML viewer uses. Patches it directly into the HTML file.
"""
import numpy as np
import json, re, sys

HTML_FILE = "UR5_REAL_interactive.html"

def load_data():
    embs = np.load("real_embs.npy")        # (N_windows, D)
    labels = np.load("real_labels.npy")     # (N_windows,)
    centers = np.load("real_centers.npy")   # (N_windows,)
    frames = np.load("real_frames_full.npy")  # (N_total_frames, 64, 64)
    return embs, labels, centers, frames


def compute_pca_projection(embs):
    """PCA → 2D for embedding-space visualization."""
    ec = embs - embs.mean(0)
    U, S, Vt = np.linalg.svd(ec, full_matrices=False)
    proj = ec @ Vt[:2].T
    return [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in proj]


def compute_change_signal(embs):
    """Cosine distance between consecutive windows → phase boundary signal."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    normed = embs / norms
    change = [0.0]
    for i in range(1, len(embs)):
        d = float(1.0 - np.dot(normed[i], normed[i-1]))
        change.append(max(0.0, d))
    mx = max(change) if max(change) > 0 else 1.0
    change = [round(float(c / mx), 4) for c in change]
    return change


def subsample_windows(labels, centers, proj, change, frames, step=2):
    """
    Subsample every `step` windows for display, ensuring the final window is included.
    For each displayed window, we include the full-res frame at that center.
    """
    indices = list(range(0, len(labels), step))
    if indices[-1] != len(labels) - 1:
        indices.append(len(labels) - 1)
        
    sub_labels = [int(labels[i]) for i in indices]
    sub_centers = [int(centers[i]) for i in indices]
    sub_proj = [proj[i] for i in indices]
    sub_change = [change[i] for i in indices]

    # Extract frames at each center position
    n_total = len(frames)
    fw, fh = 30, 53  # display resolution (scaled down for HTML embedding)
    from PIL import Image
    sub_frames = []
    for i in indices:
        c = int(centers[i])
        if c >= n_total:
            c = n_total - 1
        f = frames[c]  # (64, 64) float32
        # Resize to fw×fh for HTML
        img = Image.fromarray((f * 255).astype(np.uint8), mode='L')
        img = img.resize((fw, fh), Image.LANCZOS)
        sub_frames.append(list(np.array(img).flatten().tolist()))

    return sub_labels, sub_centers, sub_proj, sub_change, sub_frames, fw, fh


def assign_phase_names(labels, change):
    """Explicitly map phase labels to semantic action names."""
    known_names = {
        "0": "IDLE / STILL",
        "1": "MOVE DOWN",
        "2": "MOVING",
        "3": "ITEM DROP / RELEASE",
        "4": "MOVE UP / LIFT"
    }
    known_cols = {
        "0": "#2E75B6",
        "1": "#BF9500",
        "2": "#C03030",
        "3": "#8B5CF6",
        "4": "#E07830"
    }

    unique_labels = sorted(set(labels))
    phase_names = {str(lbl): known_names.get(str(lbl), f"PHASE_{lbl}") for lbl in unique_labels}
    phase_col = {str(lbl): known_cols.get(str(lbl), "#780010") for lbl in unique_labels}

    return phase_names, phase_col


def build_data_blob(labels, centers, proj, change, frames_flat, fw, fh, phase_names, phase_col):
    """Build the D={...} JSON object."""
    return {
        "labels": labels,
        "centers": centers,
        "change": change,
        "proj": proj,
        "frames": frames_flat,
        "fw": fw,
        "fh": fh,
        "phase_names": phase_names,
        "phase_col": phase_col,
    }


def patch_html(html_path, data_blob):
    """Write complete HTML document with stacked full-width Panels 3 & 4 and fixed change signal."""
    json_str = json.dumps(data_blob, separators=(',', ': '))

    html_template = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>V-JEPA — UR5 Self-Supervised Action Discovery</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;transition:background 0.25s,color 0.25s,border-color 0.25s}
body{background:#fff;color:#262626;padding:14px}.wrap{max-width:1180px;margin:0 auto}
h1{font-size:18px;color:#780010;margin-bottom:2px}.sub{font-size:12px;color:#595959;margin-bottom:14px}
.nav{display:flex;gap:8px;margin-bottom:14px}
.nav a{font-size:12px;font-weight:700;padding:6px 14px;border-radius:6px;text-decoration:none;transition:background 0.2s,color 0.2s}
.nav a.act{background:#780010;color:#fff}.nav a.pas{background:#e2e2e2;color:#262626}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.panel{border:1px solid #e2e2e2;border-radius:10px;padding:12px;background:#fafafa}
.fullpanel{border:1px solid #e2e2e2;border-radius:10px;padding:14px;background:#fafafa;margin-top:14px;width:100%}
.ptitle{font-size:12.5px;font-weight:700;margin-bottom:8px}
.b1{color:#2E75B6}.b2{color:#375623}.b3{color:#2E75B6}.b4{color:#BF9500}
canvas{display:block;background:#fff;border-radius:6px;max-width:100%}
.controls{display:flex;align-items:center;gap:12px;margin:14px 0 6px}
button{font-size:13px;padding:7px 16px;border-radius:8px;border:1px solid #780010;background:#780010;color:#fff;cursor:pointer;font-weight:600}
button.sec{background:#fff;color:#780010}input[type=range]{flex:1}
.chip{display:inline-block;padding:2px 10px;border-radius:10px;color:#fff;font-size:11px;font-weight:700}
.note{font-size:11px;color:#595959;line-height:1.5;margin-top:6px}.readnow{font-size:13px;font-weight:700}
#legend{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:14px 0 6px;padding:10px 14px;background:#fafafa;border:1px solid #e2e2e2;border-radius:8px;font-size:12px;}

/* Dark Mode Theme */
body.dark-mode{background:#0f172a;color:#f8fafc}
body.dark-mode h1{color:#f43f5e}
body.dark-mode .sub{color:#94a3b8}
body.dark-mode .nav a.act{background:#e11d48;color:#fff}
body.dark-mode .nav a.pas{background:#1e293b;color:#94a3b8;border:1px solid #334155}
body.dark-mode .panel,body.dark-mode .fullpanel,body.dark-mode #legend{background:#1e293b;border-color:#334155;color:#f8fafc}
body.dark-mode canvas{background:#0f172a}
body.dark-mode .note{color:#94a3b8}
body.dark-mode button{background:#e11d48;border-color:#e11d48;color:#fff}
body.dark-mode button.sec{background:#1e293b;color:#f43f5e;border-color:#f43f5e}
body.dark-mode .b1{color:#60a5fa}body.dark-mode .b2{color:#4ade80}body.dark-mode .b3{color:#38bdf8}body.dark-mode .b4{color:#facc15}
body.dark-mode #legend-title{color:#f43f5e !important}
body.dark-mode #legend span{color:#f8fafc !important}
</style></head><body><div class="wrap">
<div class="nav">
  <a class="act" href="index.html">Pure V-JEPA Baseline</a>
  <a class="pas" href="UR5_VL_JEPA_interactive.html">VL-JEPA Zero-Shot</a>
</div>
<h1>V-JEPA on a REAL UR5 video — phases discovered with no labels</h1>
<div class="sub">The encoder was trained self-supervised on this exact video. The phases below were found by clustering its embeddings — nobody labelled them.</div>
<div class="grid">
 <div class="panel"><div class="ptitle b1">1 · Real UR5 video <span id="chip" class="chip"></span></div>
   <div style="position:relative;width:100%;max-width:360px;margin:0 auto">
    <video id="realvid" muted playsinline preload="auto" style="width:100%;border-radius:6px;display:block;background:#000" src="YTDown.com_Shorts_UR5-demonstration_Media_RPhDaaa79vg_001_1080p.mp4"></video>
   </div>
   <canvas id="vid" width="180" height="318" style="display:none"></canvas>
   <div class="note">Actual footage. The colour = the phase the model assigned to this moment.</div></div>
 <div class="panel" style="display:flex;flex-direction:column;gap:12px">
   <div>
     <div class="ptitle b2">2 · Embedding space — JEPA's real output</div>
     <canvas id="emb" width="540" height="200" style="width:100%"></canvas>
     <div class="note">Each window of the video → a point. The <b>clusters</b> are the phases the model discovered on its own.</div>
   </div>
   <div style="border-top:1px solid #e2e2e2;padding-top:10px" id="b3-box">
     <div class="ptitle b3">3 · Phase-boundary signal (no labels)</div>
     <canvas id="drift" width="540" height="140" style="width:100%"></canvas>
     <div class="note">Spikes mark where the embedding crosses between clusters — i.e. a phase transition.</div>
   </div>
 </div>
</div>
<div class="fullpanel"><div class="ptitle b4">4 · Discovered phase timeline</div>
   <canvas id="tl" width="1150" height="140" style="width:100%"></canvas>
   <div class="note">The whole video segmented into phases. Contiguous blocks = the model found coherent stages.</div></div>
<div id="legend"></div>
<div class="controls">
 <button id="play">▶ Play</button>
 <button class="sec" id="reset">Reset</button>
 <button class="sec" id="theme-toggle">🌙 Dark Mode</button>
 <input type="range" id="scrub" min="0" value="0"><span class="readnow" id="info"></span>
</div>
</div><script>
const D=%%DATA_BLOB%%;
const N=D.labels.length, PN=D.phase_names, PC=D.phase_col, VID_FPS=30;
let k=0, playing=false, timer=null;
const realvid=document.getElementById('realvid'), scrub=document.getElementById('scrub');
const info=document.getElementById('info'), chip=document.getElementById('chip');
const themeToggle=document.getElementById('theme-toggle');
scrub.max = N-1;

const emb=document.getElementById('emb').getContext('2d');
const drift=document.getElementById('drift').getContext('2d');
const tl=document.getElementById('tl').getContext('2d');

let xs=D.proj.map(p=>p[0]), ys=D.proj.map(p=>p[1]);
let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);

function isDark(){ return document.body.classList.contains('dark-mode'); }

function drawVid(){
 const targetTime = D.centers[k] / VID_FPS;
 if(Math.abs(realvid.currentTime - targetTime) > 0.03) realvid.currentTime = targetTime;
 const p=D.labels[k]; chip.textContent=PN[p]; chip.style.background=PC[p];
}

function drawEmb(){
 emb.fillStyle = isDark() ? '#0f172a' : '#ffffff';
 emb.fillRect(0,0,540,200); const pad=25,W=540-2*pad,H=200-2*pad;
 const tx=x=>pad+(x-xmin)/(xmax-xmin+1e-9)*W, ty=y=>pad+(1-(y-ymin)/(ymax-ymin+1e-9))*H;
 for(let i=0;i<=k;i++){emb.beginPath();emb.arc(tx(D.proj[i][0]),ty(D.proj[i][1]),4.5,0,7);
  emb.fillStyle=PC[D.labels[i]];emb.globalAlpha=0.7;emb.fill();emb.globalAlpha=1;}
 emb.beginPath();emb.arc(tx(D.proj[k][0]),ty(D.proj[k][1]),9,0,7);emb.lineWidth=2.5;
 emb.strokeStyle = isDark() ? '#f43f5e' : '#780010'; emb.stroke();
}

function drawDrift(){
 const pad=30, W=540-2*pad, H=140-2*pad;
 drift.fillStyle = isDark() ? '#0f172a' : '#ffffff';
 drift.fillRect(0,0,540,140);
 const tx=i=>pad+i/(N-1)*W, ty=v=>pad+(1-v)*H;
 drift.strokeStyle = isDark() ? '#38bdf8' : '#2E75B6'; drift.lineWidth=1.8;drift.beginPath();
 for(let i=0;i<=k;i++){const X=tx(i),Y=ty(D.change[i]);i?drift.lineTo(X,Y):drift.moveTo(X,Y);}drift.stroke();
 for(let i=1;i<=k;i++){
  if(D.labels[i]!=D.labels[i-1]){
   drift.strokeStyle = isDark() ? 'rgba(244,63,94,0.7)' : 'rgba(120,0,16,0.65)'; drift.setLineDash([4,4]);
   drift.beginPath();drift.moveTo(tx(i),pad);drift.lineTo(tx(i),pad+H);drift.stroke();drift.setLineDash([]);
  }
 }
 drift.fillStyle = isDark() ? '#94a3b8' : '#595959'; drift.font='10px sans-serif';drift.fillText('window #',pad+W/2-16,140-4);
}

function drawTL(){
 const pad=20, W=1150-2*pad, H=140, y=30, h=60;
 tl.fillStyle = isDark() ? '#0f172a' : '#ffffff';
 tl.fillRect(0,0,1150,H);
 let prev=D.labels[0],s0=0;
 for(let i=1;i<=k+1;i++){
  const cur=(i<=k)?D.labels[i]:-1;
  if(i>k||cur!=prev){
   const x0=pad+s0/(N-1)*W, x1=pad+i/(N-1)*W;
   tl.fillStyle=PC[prev];tl.globalAlpha=0.75;
   tl.fillRect(x0,y,x1-x0,h);tl.globalAlpha=1;
   if(x1-x0>30){
    tl.fillStyle='#fff';tl.font='bold 11px sans-serif';tl.textAlign='center';
    tl.fillText(PN[prev],(x0+x1)/2,y+h/2+4);
   }
   s0=i;prev=cur;
  }
 }
 const cx=pad+k/(N-1)*W; tl.strokeStyle = isDark() ? '#f43f5e' : '#780010'; tl.lineWidth=2.5;tl.beginPath();tl.moveTo(cx,y-10);tl.lineTo(cx,y+h+10);tl.stroke();
 tl.textAlign='left';
}

function drawLegend(){
 const leg=document.getElementById('legend'); if(!leg)return;
 leg.innerHTML=`<span id="legend-title" style="font-weight:700;color:${isDark()?'#f43f5e':'#780010'};margin-right:8px;">Phase Legend:</span>`+
  Object.keys(PN).map(key=>`<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:11.5px;font-weight:600;"><span style="width:12px;height:12px;border-radius:3px;background:${PC[key]};display:inline-block;"></span><span style="color:${isDark()?'#f8fafc':'#262626'};">${PN[key]}</span></span>`).join('');
}

function updateThemeUI(){
 if(isDark()){
  themeToggle.textContent = '☀️ Light Mode';
 } else {
  themeToggle.textContent = '🌙 Dark Mode';
 }
 drawLegend();
 render();
}

themeToggle.onclick=()=>{
 document.body.classList.toggle('dark-mode');
 localStorage.setItem('theme', isDark() ? 'dark' : 'light');
 updateThemeUI();
};

if(localStorage.getItem('theme')==='dark'){
 document.body.classList.add('dark-mode');
}

function render(){drawVid();drawEmb();drawDrift();drawTL();info.textContent=`window ${k+1}/${N} · frame ${D.centers[k]}`;scrub.value=k;}
function step(){k++;if(k>=N){k=N-1;pause();return;}render();}
function play(){if(playing)return;playing=true;document.getElementById('play').textContent='❚❚ Pause';timer=setInterval(step,110);}
function pause(){playing=false;document.getElementById('play').textContent='▶ Play';clearInterval(timer);}
document.getElementById('play').onclick=()=>playing?pause():play();
document.getElementById('reset').onclick=()=>{pause();k=0;render();};
scrub.oninput=e=>{pause();k=+e.target.value;render();};
updateThemeUI();
</script></body></html>"""

    html_content = html_template.replace("%%DATA_BLOB%%", json_str)
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"Patched {html_path} with new data blob ({len(json_str)} bytes) and full-width stacked layout.")


if __name__ == "__main__":
    print("Loading analysis data...")
    embs, labels, centers, frames = load_data()
    print(f"  embeddings: {embs.shape}")
    print(f"  labels: {labels.shape}, unique: {sorted(set(labels))}")
    print(f"  centers: {centers.shape}")
    print(f"  frames: {frames.shape}")

    print("Computing PCA projection...")
    proj = compute_pca_projection(embs)

    print("Computing change signal...")
    change = compute_change_signal(embs)

    print("Subsampling for display...")
    sub_labels, sub_centers, sub_proj, sub_change, sub_frames, fw, fh = \
        subsample_windows(labels, centers, proj, change, frames, step=2)
    print(f"  display windows: {len(sub_labels)}")

    print("Assigning phase names...")
    phase_names, phase_col = assign_phase_names(sub_labels, sub_change)
    print(f"  phases: {phase_names}")
    print(f"  colors: {phase_col}")

    print("Building data blob...")
    blob = build_data_blob(sub_labels, sub_centers, sub_proj, sub_change,
                           sub_frames, fw, fh, phase_names, phase_col)

    print("Patching HTML...")
    patch_html(HTML_FILE, blob)
    patch_html("index.html", blob)
    print("DONE — HTML updated with real V-JEPA analysis of the actual UR5 video.")
