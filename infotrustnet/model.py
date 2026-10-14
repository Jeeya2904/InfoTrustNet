"""InfoTrustNet combines text, knowledge graph, reinforcement learning and XAI."""
import joblib
import numpy as np
from .config import Config
from .explain import narrative,occlusion_importance
from .knowledge_graph import KnowledgeGraph
from .llm_module import LLMEncoder
from .rl_agent import NAMES,W,RLFusionAgent,sigmoid,state_index
SIGNALS=["LLM","Knowledge Graph","Source credibility"]
def _logit(p):
    p=np.clip(p,1e-4,1-1e-4); return np.log(p/(1-p))
class InfoTrustNet:
    def __init__(self,config=None,kg=None):
        self.cfg=config or Config(); self.kg=kg or KnowledgeGraph(); self.encoder=LLMEncoder(self.cfg.llm_backend,self.cfg.transformer_model,self.cfg.seed); self.agent=RLFusionAgent(self.cfg.rl_lr,self.cfg.rl_baseline_decay,self.cfg.seed)
    def signals(self,texts,sources=None):
        texts=list(texts); sources=list(sources) if sources is not None else [""]*len(texts); p_llm=self.encoder.predict_proba(texts); kg_res=[self.kg.verify(t) for t in texts]; cred=np.array([self.kg.source_credibility(s) for s in sources])
        feats=np.stack([np.clip(_logit(p_llm),-4,4),np.array([-4.*r.score for r in kg_res]),4.*(.5-cred)],axis=1); states=[state_index(r.covered,p,c) for r,p,c in zip(kg_res,p_llm,cred)]
        return feats,states,p_llm,kg_res,cred
    def fit(self,train_df,val_df):
        self.encoder.fit(train_df["text"],train_df["label"]); feats,states,*_=self.signals(val_df["text"],val_df["source"]); self.agent.train(feats,states,val_df["label"].values,epochs=self.cfg.rl_epochs); return self
    def predict_proba(self,texts,sources=None):
        feats,states,*_=self.signals(texts,sources); return np.array([RLFusionAgent.fuse(f,self.agent.act(s)) for f,s in zip(feats,states)])
    def predict(self,texts,sources=None): return (self.predict_proba(texts,sources)>.5).astype(int)
    def explain(self,text,source=""):
        feats,states,p_llm,kg_res,cred=self.signals([text],[source]); f,s,kg=feats[0],states[0],kg_res[0]; a=self.agent.act(s); p=RLFusionAgent.fuse(f,a); contrib={n:float(w*x) for n,w,x in zip(SIGNALS,W[a],f)}; pred={"is_misinformation":p>.5,"p_misinformation":p}; tokens=occlusion_importance(self.predict_proba,text,source,self.cfg.occlusion_top_k)
        return {"label":"MISINFORMATION" if p>.5 else "RELIABLE","p_misinformation":p,"strategy":NAMES[a],"signals":{"llm_p_fake":float(p_llm[0]),"kg_score":kg.score,"source_credibility":float(cred[0])},"contributions":contrib,"kg_support":kg.support,"kg_contradictions":kg.contradict,"kg_paths":kg.paths,"token_importance":tokens,"explanation":narrative(pred,kg,NAMES[a],contrib,tokens)}
    def save(self,path=None):
        path=path or self.cfg.model_path; path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(self,path); return path
    @staticmethod
    def load(path=None): return joblib.load(path or Config().model_path)
