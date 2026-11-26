"""Sentetik veri: 2000 musteri x son 20 mobil islem -> 8 ziyaret nedeni."""
import numpy as np, pandas as pd, argparse, os

CLASSES = ["failed_transfer","blocked_card","loan_inquiry","account_opening",
           "card_application","fx_exchange","dispute_charge","app_access_help"]
EVENTS = ["login","balance_view","transfer_ok","transfer_fail","card_payment","card_declined",
          "card_blocked","loan_calc_view","loan_page_view","fx_rate_view","fx_trade","bill_pay",
          "otp_fail","password_reset","atm_withdraw","atm_fail","statement_view","chargeback_view",
          "product_page_view","support_chat"]
E = {e:i for i,e in enumerate(EVENTS)}

# Her sinif icin ipucu olaylari (olay, olasilik agirligi)
CLUES = {
 0:[("transfer_fail",.55),("support_chat",.15),("login",.1),("balance_view",.2)],
 1:[("card_declined",.35),("card_blocked",.35),("atm_fail",.15),("support_chat",.15)],
 2:[("loan_calc_view",.45),("loan_page_view",.35),("balance_view",.1),("statement_view",.1)],
 3:[("product_page_view",.5),("login",.1),("support_chat",.2),("statement_view",.2)],
 4:[("product_page_view",.4),("card_payment",.25),("loan_page_view",.1),("balance_view",.25)],
 5:[("fx_rate_view",.55),("fx_trade",.2),("transfer_ok",.1),("balance_view",.15)],
 6:[("chargeback_view",.35),("card_payment",.3),("statement_view",.25),("support_chat",.1)],
 7:[("otp_fail",.4),("password_reset",.35),("login",.15),("support_chat",.1)],
}
PRIOR = np.array([.18,.14,.13,.10,.10,.10,.13,.12])

# Sirali motifler: ayni olay kumesi, farkli sira -> farkli ziyaret nedeni.
# (3 olay arka arkaya; torba-of-olay ozellikleri bu ayrimi goremez)
MOTIF = {
 0:("transfer_fail","support_chat","login"),   5:("login","support_chat","transfer_fail"),
 1:("card_declined","atm_fail","support_chat"), 6:("support_chat","atm_fail","card_declined"),
 2:("loan_calc_view","statement_view","product_page_view"), 4:("product_page_view","statement_view","loan_calc_view"),
 3:("login","otp_fail","password_reset"),       7:("password_reset","otp_fail","login"),
}
def gen(n, seed, noise=0.35, clue_len=(2,6), order=0.7):
    rng = np.random.default_rng(seed)
    base_w = np.ones(len(EVENTS)); base_w[[E["login"],E["balance_view"],E["card_payment"],E["bill_pay"],E["transfer_ok"]]] = 6
    base_w /= base_w.sum()
    rows=[]
    for cid in range(n):
        y = rng.choice(8,p=PRIOR)
        seq = list(rng.choice(len(EVENTS),size=20,p=base_w))
        has_clue = rng.random() > 0.12                      # %12 musteri ipucusuz
        if has_clue:
            k = rng.integers(*clue_len)
            names=[c for c,_ in CLUES[y]]; w=np.array([w for _,w in CLUES[y]])
            # ipuclari son 1-3 gune (dizinin sonuna dogru) yigilir
            pos = rng.choice(np.arange(20-max(k+4,8),20),size=k,replace=False)
            for p in pos:
                if rng.random() > noise*0.5:               # ipucu bazen kayip
                    seq[p]=E[rng.choice(names,p=w/w.sum())]
            # karistirici: baska sinifin ipucu da serpistir
            if rng.random() < noise:
                oy = rng.choice([c for c in range(8) if c!=y])
                for p in rng.choice(np.arange(0,20),size=rng.integers(1,3),replace=False):
                    seq[p]=E[CLUES[oy][0][0]]
        if rng.random() < order:                         # sirali motifi son 8 olay icine yerlestir
            m=MOTIF[y]; st=rng.integers(12,18)
            for j,e in enumerate(m): seq[st+j]=E[e]
        gaps = np.round(rng.exponential(18,size=20)+0.2,1)            # saat
        amt  = np.round(np.exp(rng.normal(5.2,1.2,size=20)),2)        # TL
        hour = rng.integers(0,24,size=20)
        r={"customer_id":cid,"label":y}
        for t in range(20):
            r[f"ev_{t}"]=seq[t]; r[f"gap_{t}"]=gaps[t]; r[f"amt_{t}"]=amt[t]; r[f"hr_{t}"]=hour[t]
        rows.append(r)
    return pd.DataFrame(rows)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=2000); ap.add_argument("--noise",type=float,default=0.35); ap.add_argument("--order",type=float,default=0.7)
    a=ap.parse_args(); os.makedirs("data",exist_ok=True)
    df=gen(a.n,seed=42,noise=a.noise,order=a.order); df.to_csv("data/synthetic.csv",index=False)
    print(df.shape, df.label.map(dict(enumerate(CLASSES))).value_counts().to_dict())
