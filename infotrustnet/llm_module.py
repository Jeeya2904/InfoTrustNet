"""Text classifier; TF-IDF is offline-friendly, transformer backend is optional."""
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
class LLMEncoder:
    def __init__(self,backend="tfidf",model_name="sentence-transformers/all-MiniLM-L6-v2",seed=42):
        self.backend,self.model_name,self.seed=backend,model_name,seed; self._st=None; self.vectorizer=None; self.clf=LogisticRegression(C=3.,max_iter=2000,random_state=seed)
        if backend=="transformer":
            try: self._load_st()
            except Exception as e: warnings.warn(f"Transformer backend unavailable ({e}); falling back to TF-IDF."); self.backend="tfidf"
    def _load_st(self):
        from sentence_transformers import SentenceTransformer
        self._st=SentenceTransformer(self.model_name)
    def __getstate__(self):
        state=self.__dict__.copy(); state["_st"]=None; return state
    def _embed(self,texts,fit=False):
        texts=list(texts)
        if self.backend=="transformer":
            if self._st is None: self._load_st()
            return self._st.encode(texts,normalize_embeddings=True,show_progress_bar=False)
        if fit:
            self.vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,lowercase=True); return self.vectorizer.fit_transform(texts)
        return self.vectorizer.transform(texts)
    def fit(self,texts,labels): self.clf.fit(self._embed(texts,fit=True),np.asarray(labels)); return self
    def predict_proba(self,texts): return self.clf.predict_proba(self._embed(texts))[:,1]
