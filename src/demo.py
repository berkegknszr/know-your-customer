"""Uctan uca demo: musteri sec -> son 20 islem -> Top-3 neden -> sira numarasi -> yerel LLM 3 satir ozet."""
import sys, json, urllib.request, numpy as np, pandas as pd, torch, torch.nn as nn
from generate_data import EVENTS, CLASSES
torch.manual_seed(0); np.random.seed(0)
df=pd.read_csv("data/synthetic.csv"); y=df.label.values
ev=df[[f"ev_{t}" for t in range(20)]].values
gap=np.log1p(df[[f"gap_{t}" for t in range(20)]].values); amt=np.log1p(df[[f"amt_{t}" for t in range(20)]].values); hr=df[[f"hr_{t}" for t in range(20)]].values
num=np.stack([gap,amt,np.sin(hr/24*6.283),np.cos(hr/24*6.283)],-1).astype("float32")
test=np.arange(len(y))[-400:]; train=np.arange(len(y))[:-400]
class GRU(nn.Module):
    def __init__(s,d=64):
        super().__init__(); s.e=nn.Embedding(len(EVENTS),d); s.n=nn.Linear(4,d); s.g=nn.GRU(d,d,batch_first=True); s.o=nn.Linear(d,8)
    def forward(s,e,n): return s.o(s.g(s.e(e)+s.n(n))[0][:,-1])
m=GRU(); opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=1e-2)
E,N,Y=torch.tensor(ev,dtype=torch.long),torch.tensor(num),torch.tensor(y,dtype=torch.long)
for _ in range(40):
    p=torch.tensor(np.random.permutation(train))
    for i in range(0,len(p),64):
        b=p[i:i+64]; opt.zero_grad(); nn.functional.cross_entropy(m(E[b],N[b]),Y[b]).backward(); opt.step()
m.eval()
with torch.no_grad(): P=torch.softmax(m(E[test],N[test]),-1).numpy()
top3=(np.argsort(-P,1)[:,:3]==y[test][:,None]).any(1).mean(); top1=(P.argmax(1)==y[test]).mean()
print(f"[Model] GRU  Top-1={top1:.3f}  Top-3={top3:.3f}  (400 test musterisi)\n")
LABEL={"failed_transfer":"basarisiz havale","blocked_card":"bloke kart","loan_inquiry":"kredi bilgisi","account_opening":"hesap acma",
"card_application":"kart basvurusu","fx_exchange":"doviz islemi","dispute_charge":"islem itirazi","app_access_help":"uygulama erisim sorunu"}
k=int(sys.argv[1]) if len(sys.argv)>1 else 7
i=test[k]; order=np.argsort(-P[k])[:3]
print(f"[Gelen musteri #{df.customer_id[i]}]  son 6 islem: "+" > ".join(EVENTS[e] for e in ev[i][-6:]))
print("[Tahmin Top-3] "+" | ".join(f"{LABEL[CLASSES[c]]} %{P[k][c]*100:.0f}" for c in order))
ticket=f"B-{42+k:03d}"; desk={0:"Gise 3",1:"Gise 1",2:"Gise 5",3:"Gise 4",4:"Gise 4",5:"Gise 2",6:"Gise 1",7:"Gise 6"}[order[0]]
print(f"[Sira] SMS: Numaraniz {ticket}, {desk}.  (Gercek neden: {LABEL[CLASSES[y[i]]]})")
prompt=("Banka temsilcisi icin 3 maddelik Turkce on ozet yaz. Musteri son islemleri: "+", ".join(EVENTS[e] for e in ev[i][-8:])+
        ". Tahmini ziyaret nedenleri: "+", ".join(LABEL[CLASSES[c]] for c in order)+". Kisa ve oneri niteliginde ol, kesin hukum verme. /no_think")
try:
    r=urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate",json.dumps({"model":"qwen3:8b","prompt":prompt,"stream":False}).encode(),{"Content-Type":"application/json"}),timeout=90)
    out=json.loads(r.read())["response"].split("</think>")[-1].strip()
except Exception as e: out=f"(LLM yok: {e})"
print("[Temsilci ozeti - Qwen3 yerel]\n"+out)
