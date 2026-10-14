import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import pytest
from infotrustnet import Config,InfoTrustNet
from infotrustnet.data import generate_dataset,split_dataset
from infotrustnet.knowledge_graph import KnowledgeGraph
@pytest.fixture(scope="module")
def trained():
    cfg=Config(rl_epochs=30); train,val,test=split_dataset(generate_dataset(500,seed=1),seed=1); return InfoTrustNet(cfg).fit(train,val),test
def test_kg_support_and_contradiction():
    kg=KnowledgeGraph(); assert kg.verify("Paris is the capital of France.").score==1.0; assert kg.verify("Berlin is the capital of France.").score==-1.0; assert kg.verify("Some unrelated sentence.").covered is False
def test_source_credibility():
    kg=KnowledgeGraph(); assert kg.source_credibility("Reuters.com")>.9; assert kg.source_credibility("unknown-site.xyz")==.5
def test_accuracy_on_held_out(trained):
    model,test=trained; acc=(model.predict(test["text"],test["source"])==test["label"].values).mean(); assert acc>.9
def test_explain_structure(trained):
    model,_=trained; r=model.explain("Berlin is the capital of France.","viralbuzz24.info"); assert r["label"]=="MISINFORMATION"; assert r["kg_contradictions"] and r["token_importance"] and r["explanation"]
