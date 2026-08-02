import numpy as np, torch, time
import vjepa_real as V
torch.manual_seed(0); np.random.seed(0)

clips_all=np.load('real_clips.npy')   # (N,T,64,64)
# V-JEPA expects H=W=32 in the base config -> downscale clips to 32
import torch.nn.functional as F
c=torch.from_numpy(clips_all)                       # (N,T,64,64)
c=F.avg_pool2d(c.reshape(-1,1,64,64),2).reshape(c.shape[0],V.T,32,32).numpy()
np.save('real_clips32.npy',c)
POOL=c.shape[0]
print(f"training pool: {c.shape}")

# Dataset over REAL clips (pure self-supervised V-JEPA)
class RealVid:
    REGIMES={0:(0,)}; n_orient=1
    def sample(self,batch):
        idx=np.random.randint(0,POOL,batch)
        return torch.from_numpy(c[idx])
    def sample_regime(self,rid,batch=1):
        idx=np.random.randint(0,POOL,batch)
        return torch.from_numpy(c[idx])
V.MovingBarVideos=RealVid
V._one_clip=lambda pid=0,n=1: torch.from_numpy(c[np.random.randint(POOL)])

t=time.time()
model,data,hist=V.train(steps=600,batch=32,lr=1.5e-3,log_every=100)
torch.save(model.state_dict(),'vjepa_realvid.pt')
np.save('vjepa_realvid_history.npy',hist,allow_pickle=True)
print(f"DONE {time.time()-t:.0f}s loss {hist['loss'][0]:.4f}->{hist['loss'][-1]:.5f} ({hist['loss'][0]/hist['loss'][-1]:.0f}x) rank {hist['rank'][0]:.0f}->{hist['rank'][-1]:.0f}")
