import json
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
from .rl_agent import sigmoid
def metrics(y,p):
    y,p=np.asarray(y),np.asarray(p); pred=(p>.5).astype(int)
    return {"accuracy":round(accuracy_score(y,pred),4),"precision":round(precision_score(y,pred,zero_division=0),4),"recall":round(recall_score(y,pred,zero_division=0),4),"f1":round(f1_score(y,pred,zero_division=0),4),"roc_auc":round(roc_auc_score(y,p),4) if len(set(y))>1 else None}
def compare(model,test_df):
    feats,_,p_llm,_,_=model.signals(test_df["text"],test_df["source"]); y=test_df["label"].values
    return {"LLM only":metrics(y,p_llm),"KG only":metrics(y,sigmoid(feats[:,1])),"Static fusion (equal weights)":metrics(y,sigmoid(feats.sum(axis=1))),"InfoTrustNet (RL fusion)":metrics(y,model.predict_proba(test_df["text"],test_df["source"]))}
def print_table(results):
    cols=["accuracy","precision","recall","f1","roc_auc"]; print(f"{'Model':34s}"+"".join(f"{c:>11s}" for c in cols))
    for name,m in results.items(): print(f"{name:34s}"+"".join(f"{m[c]:>11}" for c in cols))
def save_json(results,path):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(results,indent=2))
