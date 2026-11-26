import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, top_k_accuracy_score
from generate_data import EVENTS, CLASSES
df=pd.read_csv("data/synthetic.csv"); y=df.label.values
ev=df[[f"ev_{t}" for t in range(20)]].values
F=pd.DataFrame({"gap_mean":df[[f"gap_{t}" for t in range(20)]].mean(axis=1),"amt_mean":df[[f"amt_{t}" for t in range(20)]].mean(axis=1)})
for w,name in [(20,"all"),(6,"last6")]:
    for i,e in enumerate(EVENTS): F[f"{name}_{e}"]=(ev[:,-w:]==i).sum(axis=1)
for t in range(1,6): F[f"lastev_{t}"]=ev[:,-t]
Xtr,Xte,ytr,yte=train_test_split(F,y,test_size=0.2,stratify=y,random_state=0)
m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=15,verbose=-1).fit(Xtr,ytr)
P=m.predict_proba(Xte)
print("Top-1",round(top_k_accuracy_score(yte,P,k=1),3),"Top-3",round(top_k_accuracy_score(yte,P,k=3),3),"Macro-F1",round(f1_score(yte,P.argmax(1),average='macro'),3))
