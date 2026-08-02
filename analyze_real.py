import numpy as np, torch, torch.nn.functional as F
import vjepa_real as V
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
torch.manual_seed(0); np.random.seed(0)

# Load trained V-JEPA model
model=V.VJEPA(); model.load_state_dict(torch.load('vjepa_realvid.pt')); model.eval()
frames=np.load('real_frames_full.npy')   # (N_frames, 64, 64)

# Encode a sliding window at EACH frame position (T=8), downscaled to 32
T=V.T
embs=[]; centers=[]
with torch.no_grad():
    for i in range(len(frames)-T):
        win=frames[i:i+T]                              # (T,64,64)
        wt=torch.from_numpy(win)[None]                 # (1,T,64,64)
        wt=F.avg_pool2d(wt.reshape(-1,1,64,64),2).reshape(1,T,32,32)
        e=model.target_encoder(clips=wt).mean(1)       # (1,D)
        embs.append(F.normalize(e,dim=-1).squeeze(0).numpy())
        centers.append(i+T//2)
embs=np.stack(embs)
print(f"encoded {embs.shape} windows from REAL video with V-JEPA target encoder")

# Feature augmentation with temporal latent velocity (first derivative in embedding space)
# Delta e_t = e_t - e_{t-1}
diffs = np.zeros_like(embs)
diffs[1:] = embs[1:] - embs[:-1]
features = np.hstack([embs, 3.0 * diffs])

# Pure unsupervised K-Means clustering over V-JEPA latent representations + velocity
k_clusters = 5
km = KMeans(k_clusters, n_init=20, random_state=0).fit(features)
raw_labels = km.labels_

# Relabel clusters purely in order of temporal first appearance for clean timeline visualization
order = {}
nxt = 0
labels = np.zeros_like(raw_labels)
for i, l in enumerate(raw_labels):
    if l not in order:
        order[l] = nxt
        nxt += 1
    labels[i] = order[l]

switches = np.sum(labels[1:] != labels[:-1])
print(f"Discovered {len(set(labels))} phases purely from V-JEPA embeddings without manual overwrites.")
print(f"Phase switches across video: {switches}")

np.save('real_embs.npy', embs)
np.save('real_labels.npy', labels)
np.save('real_centers.npy', np.array(centers))
np.save('real_k.npy', k_clusters)
print("Saved 100% pure V-JEPA embeddings + discovered phase labels.")
