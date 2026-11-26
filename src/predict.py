"""Musteri -> Top-3 neden -> SMS (simule) -> yerel LLM ozeti. Veritabani: SQLite."""
import json, sqlite3, urllib.request, numpy as np, pandas as pd, torch
from generate_data import EVENTS, CLASSES
from train import GRU, tensors
TR={"failed_transfer":"basarisiz havale","blocked_card":"bloke kart","loan_inquiry":"kredi bilgisi","account_opening":"hesap acma",
"card_application":"kart basvurusu","fx_exchange":"doviz islemi","dispute_charge":"islem itirazi","app_access_help":"uygulama erisim sorunu"}
DESK=["Gise 3","Gise 1","Gise 5","Gise 4","Gise 4","Gise 2","Gise 1","Gise 6"]
df=pd.read_csv("data/synthetic.csv")
db=sqlite3.connect("data/kyc.db",check_same_thread=False)
df[["customer_id"]+[f"ev_{t}" for t in range(20)]].to_sql("transactions",db,if_exists="replace",index=False)
model=GRU(); model.load_state_dict(torch.load("data/model.pt")); model.eval()
E,N=tensors(df)
def ozet(events,top3):
    p=("Banka temsilcisi icin 3 maddelik kisa Turkce on ozet yaz. Musteri son islemleri: "+", ".join(events)+". Tahmini nedenler (olasilik sirasiyla): "+", ".join(top3)+". Ilk madde en olasi nedene odaklansin; olasiligi %10 altindaki nedenlerden bahsetme. Kesin hukum verme. /no_think")
    try:
        r=urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate",json.dumps({"model":"qwen3:8b","prompt":p,"stream":False,"keep_alive":"2h"}).encode(),{"Content-Type":"application/json"}),timeout=90)
        return json.loads(r.read())["response"].split("</think>")[-1].strip()
    except Exception as e: return f"(LLM kullanilamiyor: {e})"
def predict(customer_id:int, llm:bool=True):
    row=db.execute("select * from transactions where customer_id=?",(customer_id,)).fetchone()
    if row is None: raise ValueError("musteri yok")
    with torch.no_grad(): P=torch.softmax(model(E[customer_id:customer_id+1],N[customer_id:customer_id+1]),-1)[0].numpy()
    order=np.argsort(-P)[:3]; evs=[EVENTS[i] for i in row[1:]][-8:]; top3=[f"{TR[CLASSES[c]]} (%{int(P[c]*100)})" for c in order]
    return {"customer_id":customer_id,"last_events":evs,
            "top3":[{"neden":TR[CLASSES[c]],"olasilik":round(float(P[c]),2)} for c in order],
            "sms":f"Sirano B-{customer_id%900+100}, {DESK[order[0]]}. (simule SMS)",
            "ozet":ozet(evs,top3) if llm else None}
