"""InfoTrustNet: KG + LLM + RL + XAI hybrid for misinformation detection."""
from .config import Config
from .model import InfoTrustNet
__all__ = ["Config", "InfoTrustNet"]
__version__ = "1.0.0"
