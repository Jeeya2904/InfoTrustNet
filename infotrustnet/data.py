"""Synthetic demo dataset and CSV loading utilities."""
import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_DIR
from .facts import EXTRA_FALSE_OBJECTS,FACTS,RELIABLE_SOURCES,UNRELIABLE_SOURCES
TEMPLATES={"capital of":"{s} is the capital of {o}","wrote":"{s} wrote {o}","invented":"{s} invented the {o}","located in":"{s} is located in {o}","boils at":"{s} boils at {o} at sea level","freezes at":"{s} freezes at {o} at sea level"}
PREFIXES=["","","According to the report, ","Officials confirmed that ","Experts say ","A new briefing states that ","It was reported that "]
HYPE=["BREAKING: {x} - share before they delete this!","SHOCKING truth the media is hiding: {x}!!!","You won't believe it: {x}. Doctors hate this trick!","URGENT!!! {x}. Wake up people, they don't want you to know!"]
CONSPIRACY=["drinking lemon water cures all diseases overnight","the moon landing was filmed in a secret studio","5G towers spread viruses to control the population","a secret elite is replacing world leaders with clones","vaccines contain microchips that track your location","scientists admit the earth is flat and hid the proof","this one weird fruit makes billionaires out of ordinary people","government weather machines are causing every hurricane","bank accounts will be frozen tomorrow in a hidden plan","a miracle pill melts fat while you sleep and doctors are furious"]
NEUTRAL_REAL=["The city council approved the annual budget on Tuesday after a public hearing.","The central bank left interest rates unchanged at its latest meeting.","Researchers published a peer-reviewed study on coastal erosion in a scientific journal.","The national football team will play two qualifying matches next month.","Local schools will reopen on Monday following scheduled maintenance work.","The health ministry released its quarterly report on vaccination coverage.","A new bridge connecting the two districts is expected to open next spring.","The university announced a scholarship program for undergraduate students.","Heavy rain is forecast for the northern region over the weekend.","The transport authority introduced a revised bus timetable this week."]
def _render(s,p,o): return TEMPLATES[p].format(s=s,o=o)
def _alt_objects(p): return sorted({o for _,pp,o in FACTS if pp==p})+EXTRA_FALSE_OBJECTS.get(p,[])
def _source(rng,kind):
    r=rng.random()
    if kind=="real": return rng.choice(RELIABLE_SOURCES) if r<.85 else ("" if r<.95 else rng.choice(UNRELIABLE_SOURCES))
    return rng.choice(UNRELIABLE_SOURCES) if r<.75 else ("" if r<.90 else rng.choice(RELIABLE_SOURCES))
def generate_dataset(n=800,seed=42):
    rng=random.Random(seed); rows=[]
    for _ in range(n):
        r=rng.random(); s,p,o=rng.choice(FACTS); false_o=rng.choice([x for x in _alt_objects(p) if x!=o]); prefix=rng.choice(PREFIXES)
        if r<.30: text,label,kind=prefix+_render(s,p,o)+".",0,"real"
        elif r<.52: text,label,kind=prefix+_render(s,p,false_o)+".",1,"fake"
        elif r<.64: text,label,kind=rng.choice(HYPE).format(x=rng.choice(CONSPIRACY)),1,"fake"
        elif r<.72: text,label,kind=rng.choice(HYPE).format(x=_render(s,p,false_o)),1,"fake"
        elif r<.77: text,label,kind="BREAKING: "+_render(s,p,o)+".",0,"real"
        else: text,label,kind=rng.choice(NEUTRAL_REAL),0,"real"
        rows.append({"text":text,"label":label,"source":_source(rng,kind)})
    return pd.DataFrame(rows)
def load_dataset(path=None,n=800,seed=42):
    if path:
        df=pd.read_csv(path)
        if not {"text","label"}<=set(df.columns): raise ValueError("CSV must contain columns: text,label[,source]")
        if "source" not in df.columns: df["source"]=""
        df["source"]=df["source"].fillna("").astype(str)
        return df[["text","label","source"]].dropna(subset=["text","label"]).reset_index(drop=True)
    csv=Path(DATA_DIR)/"claims.csv"; df=generate_dataset(n=n,seed=seed); csv.parent.mkdir(parents=True,exist_ok=True); df.to_csv(csv,index=False); return df
def split_dataset(df,seed=42):
    train,rest=train_test_split(df,test_size=.4,stratify=df["label"],random_state=seed); val,test=train_test_split(rest,test_size=.5,stratify=rest["label"],random_state=seed)
    return train.reset_index(drop=True),val.reset_index(drop=True),test.reset_index(drop=True)
