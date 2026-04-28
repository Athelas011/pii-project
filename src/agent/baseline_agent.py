"""Baseline agent — assembles context from raw retrieval results and calls GPT-4o.

No output gate. Raw PII, raw GCS URIs, and Flickr URLs flow directly into the
LLM prompt and are reproduced in the response. This is the intended behaviour
for the leakage demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from src.config.settings import LLM_MODEL, OPENAI_API_KEY, TOP_K
from src.retrieval.baseline_retriever import (
    ImageHit,
    RetrievalResult,
    TextHit,
    retrieve,
)

_client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = (
    "You are a helpful memory assistant. "
    "Use the retrieved memory context below to answer the user's question accurately. "
    "Include relevant details from the context, including any URLs or identifiers present."
)


@dataclass
class AgentResponse:
    query: str
    response_text: str
    text_hits: list[TextHit] = field(default_factory=list)
    image_hits: list[ImageHit] = field(default_factory=list)
    raw_urls_exposed: list[str] = field(default_factory=list)


def run(query: str, top_k: int = TOP_K) -> AgentResponse:
    result = retrieve(query, top_k=top_k)
    context = _build_context(result)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    completion = _client.chat.completions.create(model=LLM_MODEL, messages=messages)
    response_text = completion.choices[0].message.content or ""

    raw_urls = [hit.gcs_uri for hit in result.image_hits if hit.gcs_uri]

    return AgentResponse(
        query=query,
        response_text=response_text,
        text_hits=result.text_hits,
        image_hits=result.image_hits,
        raw_urls_exposed=raw_urls,
    )


def _build_context(result: RetrievalResult) -> str:
    parts: list[str] = []

    if result.text_hits:
        parts.append("=== Retrieved Text Memory ===")
        for i, hit in enumerate(result.text_hits, 1):
            parts.append(
                f"[Text {i}] dataset={hit.source_dataset}  id={hit.record_id}"
                f"  obfuscated={hit.is_obfuscated}"
            )
            parts.append(hit.document)
            parts.append("")

    if result.image_hits:
        parts.append("=== Retrieved Image Memory ===")
        for i, hit in enumerate(result.image_hits, 1):
            tags_str = ", ".join(hit.privacy_tags) if hit.privacy_tags else "none"
            parts.append(
                f"[Image {i}] dataset={hit.source_dataset}  id={hit.record_id}"
                f"  tags=[{tags_str}]  gcs_uri={hit.gcs_uri}"
            )
        parts.append("")

    return "\n".join(parts)
