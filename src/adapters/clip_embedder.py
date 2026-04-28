"""CLIP text encoder for cross-modal retrieval.

Encodes a text query into a 512-dim vector that lives in the same space as the
stored CLIP image embeddings, enabling text→image retrieval across modalities.
Model is loaded lazily on first call to avoid slow import at startup.
"""

from __future__ import annotations

import torch

from src.config.settings import CLIP_MODEL_NAME, CLIP_PRETRAINED

_model = None
_tokenizer = None


def _load() -> None:
    global _model, _tokenizer
    if _model is not None:
        return
    import open_clip
    _model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    _tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    _model.eval()


def embed_text_for_image_search(text: str) -> list[float]:
    """Return a normalised 512-dim CLIP text embedding suitable for cosine search."""
    _load()
    with torch.no_grad():
        tokens = _tokenizer([text])
        features = _model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].tolist()
