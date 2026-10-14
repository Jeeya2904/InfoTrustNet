"""InfoTrustNet CLI: train, evaluate, predict, explain or run demo."""
import argparse
import json
import pandas as pd
from infotrustnet import Config,InfoTrustNet
from infotrustnet.config import ARTIFACT_DIR
from infotrustnet.data import load_dataset,split_dataset
from infotrustnet.evaluation import compare,print_table,save_json
def cmd_train(args):
    cfg=Config(llm_backend=args.backend,seed=args.seed); df=load_dataset(args.data,n=cfg.n_synthetic,seed=cfg.seed); train,val,test=split_dataset(df,cfg.seed)
    print(f"Data: {len(train)} train / {len(val)} val / {len(test)} test"); model=InfoTrustNet(cfg).fit(train,val); test.to_csv(ARTIFACT_DIR/"test.csv",index=False)
    print(f"RL mean reward: first epoch {model.agent.history[0]:.3f} -> last {model.agent.history[-1]:.3f}"); print("Saved model to",model.save()); results=compare(model,test); print_table(results); save_json(results,ARTIFACT_DIR/"metrics.json"); return model
def cmd_evaluate(args):
    model=InfoTrustNet.load(); test=pd.read_csv(ARTIFACT_DIR/"test.csv",keep_default_na=False); print_table(compare(model,test))
def cmd_predict(args):
    r=InfoTrustNet.load().explain(args.text,args.source); print(f"{r['label']}  (P(misinformation) = {r['p_misinformation']:.3f}, strategy = {r['strategy']})")
def cmd_explain(args):
    r=InfoTrustNet.load().explain(args.text,args.source); print(r["explanation"])
    if args.json: print(json.dumps(r,indent=2,default=str))
def cmd_demo(args):
    model=InfoTrustNet.load(); examples=[("Paris is the capital of France.","reuters.com"),("Berlin is the capital of France.","viralbuzz24.info"),("Water boils at 50 degrees Celsius at sea level.",""),("BREAKING: vaccines contain microchips that track your location - share before they delete this!","dailytruthleaks.net"),("The city council approved the annual budget on Tuesday after a public hearing.","apnews.com")]
    for text,src in examples:
        r=model.explain(text,src); print("="*78); print(f"CLAIM : {text}\nSOURCE: {src or '(unknown)'}"); print(r["explanation"])
def main():
    ap=argparse.ArgumentParser(description="InfoTrustNet misinformation detector"); sub=ap.add_subparsers(dest="cmd",required=True)
    t=sub.add_parser("train"); t.add_argument("--data",default=None); t.add_argument("--backend",default="tfidf",choices=["tfidf","transformer"]); t.add_argument("--seed",type=int,default=42); t.set_defaults(fn=cmd_train); sub.add_parser("evaluate").set_defaults(fn=cmd_evaluate)
    for name,fn in (("predict",cmd_predict),("explain",cmd_explain)):
        p=sub.add_parser(name); p.add_argument("--text",required=True); p.add_argument("--source",default="")
        if name=="explain": p.add_argument("--json",action="store_true")
        p.set_defaults(fn=fn)
    sub.add_parser("demo").set_defaults(fn=cmd_demo); args=ap.parse_args(); args.fn(args)
if __name__=="__main__": main()
