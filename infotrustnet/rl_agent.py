"""REINFORCE fusion agent for model, graph and source signals."""
import numpy as np
STRATEGIES = {"text-focused":np.array([1.5,.5,.5]),"knowledge-focused":np.array([.5,1.5,.5]),"source-focused":np.array([.5,.5,1.5]),"balanced":np.array([1.,1.,1.]),"text+knowledge":np.array([1.2,1.2,.3]),"knowledge-only":np.array([.3,2.,.3])}
NAMES=list(STRATEGIES); W=np.stack([STRATEGIES[n] for n in NAMES]); N_STATES=2*3*3
def sigmoid(z): return 1.0/(1.0+np.exp(-np.clip(z,-30,30)))
def state_index(kg_covered,p_llm,credibility):
    conf=abs(p_llm-.5); c=0 if conf<.15 else (1 if conf<.35 else 2); s=0 if credibility<.35 else (1 if credibility<=.65 else 2)
    return (int(kg_covered)*3+c)*3+s
class RLFusionAgent:
    def __init__(self,lr=.15,baseline_decay=.9,seed=42):
        self.lr,self.decay=lr,baseline_decay; self.rng=np.random.default_rng(seed); self.theta=np.zeros((N_STATES,len(NAMES))); self.baseline=np.zeros(N_STATES); self.history=[]
    def _pi(self,s):
        z=self.theta[s]-self.theta[s].max(); e=np.exp(z); return e/e.sum()
    @staticmethod
    def fuse(features,action): return float(sigmoid(W[action] @ features))
    def act(self,state,greedy=True): return int(np.argmax(self.theta[state])) if greedy else int(self.rng.choice(len(NAMES),p=self._pi(state)))
    def train(self,features,states,labels,epochs=80):
        features,labels=np.asarray(features),np.asarray(labels); n=len(labels)
        for ep in range(epochs):
            total=0.; lr=self.lr/(1+.02*ep)
            for i in self.rng.permutation(n):
                s=states[i]; pi=self._pi(s); a=int(self.rng.choice(len(NAMES),p=pi)); p=self.fuse(features[i],a); r=1.-2.*(p-labels[i])**2; adv=r-self.baseline[s]; grad=-pi; grad[a]+=1.; self.theta[s]+=lr*adv*grad; self.baseline[s]=self.decay*self.baseline[s]+(1-self.decay)*r; total+=r
            self.history.append(total/n)
        return self.history
