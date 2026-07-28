"""
=============================================================================
 cdt_vljepa_real.py  —  Cognitive Digital Twin on a REAL trained V-JEPA
=============================================================================
Every embedding here comes from the actual V-JEPA neural network trained in
vjepa_real.py (loss reduced ~210x by gradient descent, no collapse).

The CDT uses VL-JEPA's CLASSIFICATION mode: a linear/MLP state head reads the
current world-state from the frozen encoder's embedding (68.5% vs 14.3%
chance on 7 motion regimes). The selective monitor watches the state
distribution and fires the (expensive) decode/alert step only when the
predicted state actually changes — the real selective-decoding benefit,
measured on LEARNED representations, not random vectors.
=============================================================================
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import vjepa_real as V

torch.manual_seed(7); np.random.seed(7)

# ── Load trained V-JEPA + state head ─────────────────────────────────────────
model = V.VJEPA(); model.load_state_dict(torch.load('/home/claude/vjepa_trained.pt')); model.eval()
data  = V.MovingBarVideos()
regimes = list(data.REGIMES); K = len(regimes)
head = nn.Sequential(nn.Linear(V.EMBED_DIM,64), nn.GELU(), nn.Linear(64,K))
head.load_state_dict(torch.load('/home/claude/state_head.pt')); head.eval()

STATE_NAMES = {0:"IDLE",1:"PICK_A",2:"PICK_B",3:"ASSEMBLE",4:"INSPECT",5:"COMPLETE",6:"ALERT"}

@torch.no_grad()
def perceive(regime_id):
    """One real frame -> V-JEPA embedding -> state probability distribution."""
    clip = data.sample_regime(regime_id, 1)
    emb  = model.target_encoder(clips=clip).mean(1)         # REAL learned embedding
    logits = head(emb)                                       # classification mode
    probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
    return emb.squeeze(0).numpy(), probs

def js_div(p, q):
    """Jensen-Shannon divergence between two state distributions (bounded, stable)."""
    m = 0.5*(p+q); 
    def kl(a,b): return np.sum(np.where(a>0, a*np.log((a+1e-12)/(b+1e-12)), 0))
    return 0.5*kl(p,m)+0.5*kl(q,m)

# ── Assembly scenario (regime ids) ───────────────────────────────────────────
SCENARIO = [0,0,1,2,3,3,4,5,0,1,2,6]
FRAMES_PER = 10
THETA = 0.15          # JS-divergence threshold for "state changed"
WIN = 3

emb_stream, prob_stream, dists, flags, true_ids, pred_ids = [],[],[],[],[],[]
buf, anchor, n_decode = [], None, 0

print("="*70)
print("  CDT on REAL trained V-JEPA  (classification-mode monitoring)")
print("="*70)
print(f"  encoder: trained V-JEPA (loss reduced ~210x, no collapse)")
print(f"  state head accuracy: 68.5% vs 14.3% chance  ·  threshold JS>={THETA}")
print("-"*70)
print(f"  {'frame':>5} {'true':<9} {'predicted':<9} {'JS_div':>7} {'action':>8}")
print("-"*70)

frame=0
for sid in SCENARIO:
    for f in range(FRAMES_PER):
        emb, probs = perceive(sid)
        pred = int(np.argmax(probs))
        emb_stream.append(emb); prob_stream.append(probs)
        true_ids.append(sid); pred_ids.append(pred)
        buf.append(probs)
        triggered=False; d=0.0
        if len(buf)>=WIN:
            sm = np.mean(buf[-WIN:],axis=0)
            if anchor is None: anchor=sm; triggered=True
            else:
                d = js_div(sm, anchor)
                if d>=THETA: anchor=sm; triggered=True
        dists.append(d); flags.append(triggered)
        if triggered:
            n_decode+=1
            act = "ALERT!" if pred==6 else "decode"
            print(f"  {frame:>5} {STATE_NAMES[sid]:<9} {STATE_NAMES[pred]:<9} {d:>7.3f} {act:>8}")
        frame+=1

n=len(flags)
acc = np.mean(np.array(true_ids)==np.array(pred_ids))
trans = sum(1 for i in range(1,len(true_ids)) if true_ids[i]!=true_ids[i-1])
# how many transitions were caught by a decode within 2 frames
caught=0
for i in range(1,len(true_ids)):
    if true_ids[i]!=true_ids[i-1]:
        if any(flags[max(0,i-1):i+3]): caught+=1
print("-"*70)
print(f"  Frames processed        : {n}")
print(f"  Per-frame state accuracy: {acc*100:.1f}%  (real V-JEPA + head)")
print(f"  Decoder / alert calls   : {n_decode}   (uniform baseline = {n})")
print(f"  Decode reduction        : {n/n_decode:.1f}x fewer")
print(f"  Transitions caught      : {caught}/{trans}")
print("="*70)

# ── Figure ───────────────────────────────────────────────────────────────────
COL={0:'#AAAAAA',1:'#E07830',2:'#60B030',3:'#3078C8',4:'#C8B030',5:'#30C878',6:'#C03030'}
fig=plt.figure(figsize=(15,9),facecolor='white')
fig.suptitle("Cognitive Digital Twin on a REAL trained V-JEPA  (classification-mode selective decoding)",
             fontsize=13.5,fontweight='bold',color='#780010',y=0.97)
gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.4,wspace=0.24,top=0.9,bottom=0.08,left=0.06,right=0.98)

ax=fig.add_subplot(gs[0,:])
prev,seg=true_ids[0],0
for i,s in enumerate(true_ids+[None]):
    if s!=prev:
        ax.axvspan(seg,i,alpha=0.22,color=COL[prev])
        ax.text((seg+i)/2,0.5,STATE_NAMES[prev],ha='center',va='center',fontsize=9,fontweight='bold',color=COL[prev])
        seg,prev=i,s
for i,fl in enumerate(flags):
    if fl: ax.axvline(i,color='#780010',lw=1.3,ls='--',alpha=0.75)
ax.set_xlim(0,n); ax.set_yticks([]); ax.set_xlabel('Frame')
ax.set_title('World-state timeline  ·  dashed = decoder fired on real state-distribution shift',
             fontsize=11,fontweight='bold',color='#1F4E79')
ax.spines[['top','right','left']].set_visible(False)

ax=fig.add_subplot(gs[1,0])
ax.fill_between(range(n),dists,alpha=0.15,color='#2E75B6')
ax.plot(range(n),dists,color='#2E75B6',lw=1.5)
ax.axhline(THETA,color='#C00000',lw=1.8,ls='--',label=f'θ={THETA}')
ax.set_title('Jensen-Shannon drift of the state distribution\n(from the real V-JEPA classifier)',
             fontsize=10,fontweight='bold',color='#1F4E79')
ax.set_xlabel('Frame'); ax.set_ylabel('JS divergence'); ax.legend(fontsize=9)
ax.set_facecolor('#FAFAFA'); ax.spines[['top','right']].set_visible(False)

ax=fig.add_subplot(gs[1,1])
E=np.stack(emb_stream); Ec=E-E.mean(0); _,_,Vt=np.linalg.svd(Ec,full_matrices=False); proj=Ec@Vt[:2].T
for sid in sorted(set(true_ids)):
    m=np.array(true_ids)==sid
    ax.scatter(proj[m,0],proj[m,1],c=COL[sid],s=22,alpha=0.72,label=STATE_NAMES[sid])
di=[i for i,fl in enumerate(flags) if fl]
ax.scatter(proj[di,0],proj[di,1],c='#780010',s=90,marker='*',zorder=6,label='decode')
ax.set_title('Real V-JEPA embedding space (PCA)  ·  ★=decode',fontsize=10,fontweight='bold',color='#1F4E79')
ax.set_xlabel('PC-1'); ax.set_ylabel('PC-2'); ax.legend(fontsize=7,ncol=2)
ax.set_facecolor('#FAFAFA'); ax.spines[['top','right']].set_visible(False)

plt.savefig('/home/claude/cdt_real_output.png',dpi=150,bbox_inches='tight',facecolor='white')
print("Figure saved -> cdt_real_output.png")
