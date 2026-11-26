import numpy as np, pandas as pd, torch, torch.nn as nn, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, top_k_accuracy_score
from generate_data import EVENTS
torch.manual_seed(0); np.random.seed(0)
df=pd.read_csv("data/synthetic.csv"); y=df.label.values
ev=df[[f"ev_{t}" for t in range(20)]].values
gap=np.log1p(df[[f"gap_{t}" for t in range(20)]].values); amt=np.log1p(df[[f"amt_{t}" for t in range(20)]].values)
hr=df[[f"hr_{t}" for t in range(20)]].values
num=np.stack([gap,amt,np.sin(hr/24*6.283),np.cos(hr/24*6.283)],-1).astype("float32")
idx=np.arange(len(y)); tr,te=train_test_split(idx,test_size=0.2,stratify=y,random_state=0)
T=lambda a,d=None: torch.tensor(a,dtype=d)
class Enc(nn.Module):
    def __init__(s,d=64):
        super().__init__(); s.e=nn.Embedding(len(EVENTS),d); s.n=nn.Linear(4,d)
    def forward(s,e,n): return s.e(e)+s.n(n)
class GRU(nn.Module):
    def __init__(s,d=64):
        super().__init__(); s.enc=Enc(d); s.g=nn.GRU(d,d,batch_first=True,num_layers=1); s.o=nn.Linear(d,8)
    def forward(s,e,n): return s.o(s.g(s.enc(e,n))[0][:,-1])
class Trf(nn.Module):
    def __init__(s,d=64):
        super().__init__(); s.enc=Enc(d); s.pos=nn.Parameter(torch.randn(20,d)*.02)
        s.t=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,128,0.1,batch_first=True),2); s.o=nn.Linear(d,8)
    def forward(s,e,n): return s.o(s.t(s.enc(e,n)+s.pos).mean(1))
def run(M,name,ep=40):
    m=M(); opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=1e-2)
    E,N,Y=T(ev,torch.long),T(num),T(y,torch.long); best=None
    for e in range(ep):
        m.train(); p=torch.tensor(np.random.permutation(tr))
        for i in range(0,len(p),64):
            b=p[i:i+64]; opt.zero_grad(); nn.functional.cross_entropy(m(E[b],N[b]),Y[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad(): P=torch.softmax(m(E[te],N[te]),-1).numpy()
    print(name,"Top-1",round(top_k_accuracy_score(y[te],P,k=1),3),"Top-3",round(top_k_accuracy_score(y[te],P,k=3),3),"Macro-F1",round(f1_score(y[te],P.argmax(1),average='macro'),3))
run(GRU,"GRU"); run(Trf,"Transformer")
