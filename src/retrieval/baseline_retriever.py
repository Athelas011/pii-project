"""Naive baseline retriever — no privacy filtering at any stage.

Queries both Chroma collections and exposes raw GCS URIs for all image hits.
This is intentionally unguarded to demonstrate the leakage baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.adapters.clip_embedder import embed_text_for_image_search
from src.adapters.text_embedder import embed_text
from src.config.settings import TOP_K
from src.memory.chroma_client import ChromaClient
from src.memory.gcs_client import get_gcs_uri, get_vispr_gcs_uri

_chroma: ChromaClient | None = None


def _get_chroma() -> ChromaClient:
    global _chroma
    if _chroma is None:
        _chroma = ChromaClient()
    return _chroma


@dataclass
class TextHit:
    id: str
    document: str
    source_dataset: str
    record_id: str
    is_obfuscated: bool
    distance: float


@dataclass
class ImageHit:
    id: str
    document: str
    source_dataset: str
    record_id: str
    annotation_id: str        # VISPR annotation id; empty for BIV-Priv
    privacy_tags: list[str]
    gcs_uri: str              # gs:// path for direct GCS retrieval
    distance: float


@dataclass
class RetrievalResult:
    query: str
    text_hits: list[TextHit] = field(default_factory=list)
    image_hits: list[ImageHit] = field(default_factory=list)


def retrieve(query: str, top_k: int = TOP_K) -> RetrievalResult:
    """Embed query and fetch top-k from both collections without any filtering."""
    text_emb = embed_text(query)
    image_emb = embed_text_for_image_search(query)

    text_rows = _get_chroma().query_text_collection(text_emb, top_k)
    image_rows = _get_chroma().query_image_collection(image_emb, top_k)

    return RetrievalResult(
        query=query,
        text_hits=[_parse_text_hit(r) for r in text_rows],
        image_hits=[_parse_image_hit(r) for r in image_rows],
    )


def _parse_text_hit(r: dict) -> TextHit:
    meta = r["metadata"]
    return TextHit(
        id=r["id"],
        document=r["document"],
        source_dataset=meta.get("source_dataset", ""),
        record_id=meta.get("record_id", ""),
        is_obfuscated=str(meta.get("is_obfuscated", "False")) == "True",
        distance=r["distance"],
    )


def _parse_image_hit(r: dict) -> ImageHit:
    meta = r["metadata"]
    doc = r["document"]

    tags: list[str] = []
    if "with tags:" in doc:
        tag_str = doc.split("with tags:", 1)[-1].strip()
        tags = [t.strip() for t in tag_str.split(",") if t.strip()]

    source_dataset = meta.get("source_dataset", "")
    record_id = meta.get("record_id", "")
    annotation_id = meta.get("annotation_id", "") or ""
    # Guard against literal "None" stored during ingestion
    if annotation_id == "None":
        annotation_id = ""

    if source_dataset == "VISPR":
        gcs_uri = get_vispr_gcs_uri(annotation_id) if annotation_id else ""
    else:
        gcs_uri = get_gcs_uri(record_id)

    return ImageHit(
        id=r["id"],
        document=doc,
        source_dataset=source_dataset,
        record_id=record_id,
        annotation_id=annotation_id,
        privacy_tags=tags,
        gcs_uri=gcs_uri,
        distance=r["distance"],
    )
