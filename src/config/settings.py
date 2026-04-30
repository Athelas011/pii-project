import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

_PROJECT_ROOT = Path(__file__).parent.parent.parent

try:            
      from google.colab import userdata
      _colab_openai = userdata.get("OPENAI_API_KEY")
except Exception:                                                                       
      _colab_openai = None
                                                                                          
PENAI_API_KEY: str = _colab_openai or os.getenv("OPENAI_API_KEY", "")

# GCS — required for full BIV-Priv URL exposure demo; optional for text-only retrieval
GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "")
GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

CHROMA_DB_PATH: str = os.getenv(
    "CHROMA_DB_PATH", str(_PROJECT_ROOT / "data" / "chroma_db")
)
IMAGE_EMBEDDINGS_PATH: str = os.getenv(
    "IMAGE_EMBEDDINGS_PATH", str(_PROJECT_ROOT / "data" / "image_embeddings")
)

TOP_K: int = int(os.getenv("TOP_K", "5"))

# Model names — match the models used when embeddings were originally created
TEXT_EMBEDDING_MODEL: str = "text-embedding-3-small"
CLIP_MODEL_NAME: str = "ViT-B-32"
CLIP_PRETRAINED: str = "openai"
LLM_MODEL: str = "gpt-4o"
