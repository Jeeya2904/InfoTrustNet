"""Knowledge-graph module: verifies claims against trusted triples."""
import re
from dataclasses import dataclass, field
import networkx as nx
from .facts import FACTS, SOURCE_CREDIBILITY
def _has(text, phrase):
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, flags=re.I) is not None
@dataclass
class KGResult:
    support: list = field(default_factory=list)
    contradict: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    paths: list = field(default_factory=list)
    @property
    def covered(self): return bool(self.support or self.contradict)
    @property
    def score(self):
        n_s, n_c = len(self.support), len(self.contradict)
        return 0.0 if n_s + n_c == 0 else (n_s - n_c) / (n_s + n_c)
class KnowledgeGraph:
    def __init__(self, facts=None, source_credibility=None):
        self.g = nx.MultiDiGraph(); self.facts = []
        self.sources = {k.lower(): v for k,v in (source_credibility or SOURCE_CREDIBILITY).items()}
        for s,p,o in (facts if facts is not None else FACTS): self.add_fact(s,p,o)
    def add_fact(self,s,p,o):
        self.g.add_edge(s,o,key=p,relation=p); self.facts.append((s,p,o))
    def source_credibility(self,source): return self.sources.get((source or "").strip().lower(),0.5)
    def verify(self,text):
        res=KGResult(); res.entities=[n for n in self.g.nodes if _has(text,n)]; groups={}
        for s,p,o in self.facts: groups.setdefault((s,p),[]).append(o)
        for (s,p),objs in groups.items():
            if not (_has(text,s) and _has(text,p)): continue
            hit=[o for o in objs if _has(text,o)]
            if hit: res.support.append(f"{s} {p} {hit[0]}")
            else: res.contradict.append(f"KG says: {s} {p} {objs[0]}")
        und=self.g.to_undirected(); ents=res.entities
        for i in range(len(ents)):
            for j in range(i+1,len(ents)):
                try: path=nx.shortest_path(und,ents[i],ents[j])
                except nx.NetworkXNoPath: continue
                if len(path)<=3: res.paths.append(" -> ".join(path))
        return res
