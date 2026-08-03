"""
=============================================================================
 rebuild_vl_html.py  —  Generates UR5_VL_JEPA_interactive.html
 (VL-JEPA: Chen et al. 2025, arXiv:2512.10942)
=============================================================================
"""
import json, sys, numpy as np

HTML_FILE = "UR5_VL_JEPA_interactive.html"

def compute_pca_joint(vis_embs, prompt_embs):
    """Compute joint 2D PCA projection for both visual embeddings and prompt embeddings."""
    combined = np.vstack([vis_embs, prompt_embs])
    ec = combined - combined.mean(0)
    U, S, Vt = np.linalg.svd(ec, full_matrices=False)
    proj = ec @ Vt[:2].T
    
    n_vis = len(vis_embs)
    vis_proj = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in proj[:n_vis]]
    prompt_proj = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in proj[n_vis:]]
    return vis_proj, prompt_proj


def main():
    print("Loading VL-JEPA data...")
    vl_embs = np.load("vl_embs.npy")               # (667, 128)
    vl_labels = np.load("vl_labels.npy")           # (667,)
    vl_sims = np.load("vl_sims.npy")               # (667, 5)
    prompt_embs = np.load("vl_prompt_embs.npy")   # (5, 128)
    vjepa_labels = np.load("real_labels.npy")      # (667,) baseline
    centers = np.load("real_centers.npy")          # (667,)

    print("Computing PCA for joint vision-language space...")
    vis_proj, prompt_proj = compute_pca_joint(vl_embs, prompt_embs)

    # Subsample for display
    step = 2
    indices = list(range(0, len(vl_labels), step))
    if indices[-1] != len(vl_labels) - 1:
        indices.append(len(vl_labels) - 1)

    sub_vl_labels = [int(vl_labels[i]) for i in indices]
    sub_vjepa_labels = [int(vjepa_labels[i]) for i in indices]
    sub_centers = [int(centers[i]) for i in indices]
    sub_vis_proj = [vis_proj[i] for i in indices]
    sub_sims = [[round(float(x), 4) for x in vl_sims[i]] for i in indices]

    prompts = [
        "IDLE / REST",
        "MOVE DOWN",
        "MOVING",
        "ITEM DROP / RELEASE",
        "MOVE UP / LIFT"
    ]
    colors = [
        "#2E75B6",
        "#BF9500",
        "#C03030",
        "#8B5CF6",
        "#E07830"
    ]

    blob = {
        "vl_labels": sub_vl_labels,
        "vjepa_labels": sub_vjepa_labels,
        "centers": sub_centers,
        "vis_proj": sub_vis_proj,
        "prompt_proj": prompt_proj,
        "sims": sub_sims,
        "prompts": prompts,
        "colors": colors
    }

    json_str = json.dumps(blob, separators=(',', ': '))

    html_template = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>VL-JEPA — Vision-Language Joint Embedding Architecture</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
body{background:#fff;color:#262626;padding:14px}.wrap{max-width:1180px;margin:0 auto}
h1{font-size:18px;color:#780010;margin-bottom:2px}.sub{font-size:12px;color:#595959;margin-bottom:14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.panel{border:1px solid #e2e2e2;border-radius:10px;padding:12px;background:#fafafa}
.fullpanel{border:1px solid #e2e2e2;border-radius:10px;padding:14px;background:#fafafa;margin-top:14px;width:100%}
.ptitle{font-size:12.5px;font-weight:700;margin-bottom:8px}
.b1{color:#2E75B6}.b2{color:#375623}.b3{color:#2E75B6}.b4{color:#BF9500}
canvas{display:block;background:#fff;border-radius:6px;max-width:100%}
.controls{display:flex;align-items:center;gap:12px;margin:14px 0 6px}
button{font-size:13px;padding:7px 16px;border-radius:8px;border:1px solid #780010;background:#780010;color:#fff;cursor:pointer}
button.sec{background:#fff;color:#780010}input[type=range]{flex:1}
.chip{display:inline-block;padding:3px 10px;border-radius:10px;color:#fff;font-size:11px;font-weight:700}
.note{font-size:11px;color:#595959;line-height:1.5;margin-top:6px}.readnow{font-size:13px;font-weight:700}
#legend{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:14px 0 6px;padding:10px 14px;background:#fafafa;border:1px solid #e2e2e2;border-radius:8px;font-size:12px;}
.badge{background:#780010;color:#fff;font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;margin-left:6px;}
</style></head><body><div class="wrap">
<h1>VL-JEPA — Vision-Language Joint Embedding & Zero-Shot Action Identification <span class="badge">Chen et al. 2025</span></h1>
<div class="sub">Joint Vision-Language Predictive Architecture trained on UR5 video clips. Predicts spatiotemporal tubelet targets while aligning visual representations directly with natural language prompt embeddings.</div>
<div class="grid">
 <div class="panel"><div class="ptitle b1">1 · Real UR5 video <span id="chip" class="chip"></span></div>
   <div style="position:relative;width:100%;max-width:360px;margin:0 auto">
    <video id="realvid" muted playsinline preload="auto" style="width:100%;border-radius:6px;display:block;background:#000" src="YTDown.com_Shorts_UR5-demonstration_Media_RPhDaaa79vg_001_1080p.mp4"></video>
   </div>
   <div class="note">Actual UR5 footage. Top badge displays zero-shot VL-JEPA text prompt alignment confidence.</div></div>
 <div class="panel" style="display:flex;flex-direction:column;gap:12px">
   <div>
     <div class="ptitle b2">2 · Joint Vision-Language Embedding Space (Visual Windows + Language Anchors)</div>
     <canvas id="emb" width="540" height="200" style="width:100%"></canvas>
     <div class="note">Visual window embeddings (dots) aligned to natural language prompt landmark anchors (starred diamonds).</div>
   </div>
   <div style="border-top:1px solid #e2e2e2;padding-top:10px">
     <div class="ptitle b3">3 · Zero-Shot Vision-Language Prompt Similarity Probability</div>
     <canvas id="drift" width="540" height="140" style="width:100%"></canvas>
     <div class="note">Curves display dynamic softmax probability P(Prompt | Visual Window) across time for each action prompt.</div>
   </div>
 </div>
</div>
<div class="fullpanel">
  <div class="ptitle b4">4 · Timeline Comparison: Pure Unsupervised V-JEPA vs Zero-Shot VL-JEPA</div>
  <canvas id="tl" width="1150" height="160" style="width:100%"></canvas>
  <div class="note">Top bar: Pure V-JEPA self-supervised K-Means clustering baseline. Bottom bar: VL-JEPA zero-shot language prompt alignment.</div>
</div>
<div id="legend"></div>
<div class="controls"><button id="play">▶ Play</button><button class="sec" id="reset">Reset</button>
<input type="range" id="scrub" min="0" value="0"><span class="readnow" id="info"></span></div>
</div><script>
const D=%%DATA_BLOB%%;
const N=D.vl_labels.length, PN=D.prompts, PC=D.colors, VID_FPS=30;
let k=0, playing=false, timer=null;
const realvid=document.getElementById('realvid'), scrub=document.getElementById('scrub');
const info=document.getElementById('info'), chip=document.getElementById('chip');
scrub.max = N-1;

const emb=document.getElementById('emb').getContext('2d');
const drift=document.getElementById('drift').getContext('2d');
const tl=document.getElementById('tl').getContext('2d');

let all_xs = D.vis_proj.map(p=>p[0]).concat(D.prompt_proj.map(p=>p[0]));
let all_ys = D.vis_proj.map(p=>p[1]).concat(D.prompt_proj.map(p=>p[1]));
let xmin=Math.min(...all_xs), xmax=Math.max(...all_xs), ymin=Math.min(...all_ys), ymax=Math.max(...all_ys);

function drawVid(){
 const targetTime = D.centers[k] / VID_FPS;
 if(Math.abs(realvid.currentTime - targetTime) > 0.03) realvid.currentTime = targetTime;
 const p=D.vl_labels[k]; const score = (D.sims[k][p]*100).toFixed(1);
 chip.textContent = `VL-JEPA: ${PN[p]} (${score}%)`; chip.style.background = PC[p];
}

function drawEmb(){
 emb.clearRect(0,0,540,200); const pad=28, W=540-2*pad, H=200-2*pad;
 const tx=x=>pad+(x-xmin)/(xmax-xmin+1e-9)*W, ty=y=>pad+(1-(y-ymin)/(ymax-ymin+1e-9))*H;
 for(let i=0;i<=k;i++){
  emb.beginPath();emb.arc(tx(D.vis_proj[i][0]),ty(D.vis_proj[i][1]),4.5,0,7);
  emb.fillStyle=PC[D.vl_labels[i]];emb.globalAlpha=0.7;emb.fill();emb.globalAlpha=1;
 }
 // Current visual cursor
 emb.beginPath();emb.arc(tx(D.vis_proj[k][0]),ty(D.vis_proj[k][1]),9,0,7);emb.lineWidth=2.5;emb.strokeStyle='#780010';emb.stroke();

 // Draw Prompt Landmark Anchors (Diamond Star shapes)
 for(let m=0;m<D.prompt_proj.length;m++){
  const px=tx(D.prompt_proj[m][0]), py=ty(D.prompt_proj[m][1]);
  emb.fillStyle=PC[m]; emb.strokeStyle='#000'; emb.lineWidth=1.5;
  emb.beginPath(); emb.moveTo(px, py-7); emb.lineTo(px+6, py); emb.lineTo(px, py+7); emb.lineTo(px-6, py); emb.closePath();
  emb.fill(); emb.stroke();
  emb.fillStyle='#111'; emb.font='bold 9px sans-serif'; emb.fillText(`T${m}`, px+8, py+3);
 }
}

function drawDrift(){
 const pad=30, W=540-2*pad, H=140-2*pad;
 drift.clearRect(0,0,540,140);
 const tx=i=>pad+i/(N-1)*W, ty=v=>pad+(1-v)*H;
 
 // Draw probability curve for each language prompt
 for(let m=0;m<5;m++){
  drift.strokeStyle=PC[m]; drift.lineWidth=1.8; drift.beginPath();
  for(let i=0;i<=k;i++){
   const X=tx(i), Y=ty(D.sims[i][m]);
   i?drift.lineTo(X,Y):drift.moveTo(X,Y);
  }
  drift.stroke();
 }
 drift.fillStyle='#595959';drift.font='10px sans-serif';drift.fillText('window #',pad+W/2-16,140-4);
}

function drawTL(){
 const pad=20, W=1150-2*pad, H=160;
 tl.clearRect(0,0,1150,H);
 
 // Header labels
 tl.fillStyle='#595959'; tl.font='bold 10px sans-serif';
 tl.fillText("Pure V-JEPA Unsupervised (K-Means)", pad, 18);
 tl.fillText("VL-JEPA Zero-Shot Vision-Language", pad, 95);

 // 1. Top Bar: V-JEPA Unsupervised
 let y1=24, h1=40;
 let prev1=D.vjepa_labels[0], s1=0;
 for(let i=1;i<=k+1;i++){
  const cur=(i<=k)?D.vjepa_labels[i]:-1;
  if(i>k||cur!=prev1){
   const x0=pad+s1/(N-1)*W, x1=pad+i/(N-1)*W;
   tl.fillStyle=PC[prev1]; tl.globalAlpha=0.75;
   tl.fillRect(x0,y1,x1-x0,h1); tl.globalAlpha=1;
   if(x1-x0>30){
    tl.fillStyle='#fff'; tl.font='bold 10px sans-serif'; tl.textAlign='center';
    tl.fillText(PN[prev1], (x0+x1)/2, y1+h1/2+3);
   }
   s1=i; prev1=cur;
  }
 }

 // 2. Bottom Bar: VL-JEPA Zero-Shot Language Alignment
 let y2=102, h2=40;
 let prev2=D.vl_labels[0], s2=0;
 for(let i=1;i<=k+1;i++){
  const cur=(i<=k)?D.vl_labels[i]:-1;
  if(i>k||cur!=prev2){
   const x0=pad+s2/(N-1)*W, x1=pad+i/(N-1)*W;
   tl.fillStyle=PC[prev2]; tl.globalAlpha=0.85;
   tl.fillRect(x0,y2,x1-x0,h2); tl.globalAlpha=1;
   if(x1-x0>30){
    tl.fillStyle='#fff'; tl.font='bold 10px sans-serif'; tl.textAlign='center';
    tl.fillText(PN[prev2], (x0+x1)/2, y2+h2/2+3);
   }
   s2=i; prev2=cur;
  }
 }

 // Cursor line
 const cx=pad+k/(N-1)*W; tl.strokeStyle='#780010'; tl.lineWidth=2.5; tl.beginPath(); tl.moveTo(cx,15); tl.lineTo(cx,H-10); tl.stroke();
 tl.textAlign='left';
}

function drawLegend(){
 const leg=document.getElementById('legend'); if(!leg)return;
 leg.innerHTML='<span style="font-weight:700;color:#780010;margin-right:8px;">VL-JEPA Action Prompts:</span>'+
  D.prompts.map((p, key)=>`<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:11.5px;font-weight:600;"><span style="width:12px;height:12px;border-radius:3px;background:${PC[key]};display:inline-block;"></span><span style="color:#262626;">T${key}: ${p}</span></span>`).join('');
}

function render(){drawVid();drawEmb();drawDrift();drawTL();info.textContent=`window ${k+1}/${N} · frame ${D.centers[k]}`;scrub.value=k;}
function step(){k++;if(k>=N){k=N-1;pause();return;}render();}
function play(){if(playing)return;playing=true;document.getElementById('play').textContent='❚❚ Pause';timer=setInterval(step,110);}
function pause(){playing=false;document.getElementById('play').textContent='▶ Play';clearInterval(timer);}
document.getElementById('play').onclick=()=>playing?pause():play();
document.getElementById('reset').onclick=()=>{pause();k=0;render();};
scrub.oninput=e=>{pause();k=+e.target.value;render();};
drawLegend();
render();
</script></body></html>"""

    html_content = html_template.replace("%%DATA_BLOB%%", json_str)
    with open(HTML_FILE, "w") as f:
        f.write(html_content)

    print(f"Generated {HTML_FILE} ({len(json_str)} bytes) with VL-JEPA zero-shot visualization.")

if __name__ == "__main__":
    main()
