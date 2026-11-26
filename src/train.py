"""GRU modelini egitir ve model.pt olarak kaydeder (API bunu yukler)."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from generate_data import EVENTS
torch.manual_seed(0); np.random.seed(0)
class GRU(nn.Module):
    def __init__(s,d=64):
        super().__init__(); s.e=nn.Embedding(len(EVENTS),d); s.n=nn.Linear(4,d); s.g=nn.GRU(d,d,batch_first=True); s.o=nn.Linear(d,8)
    def forward(s,e,n): return s.o(s.g(s.e(e)+s.n(n))[0][:,-1])
def tensors(df):
    ev=df[[f"ev_{t}" for t in range(20)]].values
    gap=np.log1p(df[[f"gap_{t}" for t in range(20)]].values); amt=np.log1p(df[[f"amt_{t}" for t in range(20)]].values); hr=df[[f"hr_{t}" for t in range(20)]].values
    num=np.stack([gap,amt,np.sin(hr/24*6.283),np.cos(hr/24*6.283)],-1).astype("float32")
    return torch.tensor(ev,dtype=torch.long),torch.tensor(num)
if __name__=="__main__":
    df=pd.read_csv("data/synthetic.csv"); E,N=tensors(df); Y=torch.tensor(df.label.values,dtype=torch.long)
    m=GRU(); opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=1e-2)
    for _ in range(40):
        p=torch.randperm(len(Y))
        for i in range(0,len(p),64):
            b=p[i:i+64]; opt.zero_grad(); nn.functional.cross_entropy(m(E[b],N[b]),Y[b]).backward(); opt.step()
    torch.save(m.state_dict(),"data/model.pt"); print("model.pt kaydedildi")
