from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
@dataclass
class Config:
    seed: int = 42
    n_synthetic: int = 800
    llm_backend: str = "tfidf"
    transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rl_epochs: int = 80
    rl_lr: float = 0.15
    rl_baseline_decay: float = 0.9
    occlusion_top_k: int = 6
    model_path: Path = ARTIFACT_DIR / "infotrustnet.joblib"
