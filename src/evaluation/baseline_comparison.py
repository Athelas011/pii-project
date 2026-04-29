"""
Baseline comparison utilities for the Privacy-Aware Agent Runtime.

Implements four detection modes for presentation comparison:
  A — text PII filter only (image ignored)
  B — OwlViT visual detection only (no OCR, no text filter)
  C — OwlViT + ephemeral OCR (text input not redacted)
  D — full pipeline: text PII + OwlViT + ephemeral OCR

All functions accept pre-loaded model objects so notebooks can pass models
already in memory.  Results are plain dicts for easy table rendering.

Key argument these functions support:
  Text-only misses ALL visual PII.
  Visual-only misses text PII in the input and text embedded in images.
  Only the full pipeline catches both.

Reference: WebPII (Zhao 2026) shows text-extraction baselines achieve
0.357 mAP@50 vs visual detection at 0.753 — combining both is stronger.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from PIL import Image

PILImage = Any


# ── Public API ────────────────────────────────────────────────────────────────

def run_mode_a_text_only(
    text: str | None,
    pii_filter,
) -> dict:
    """Mode A: NER-based text redaction only; image completely ignored."""
    t0 = time.perf_counter()
    redacted, has_pii = _redact(text, pii_filter)
    return _result("A", "Text-Only", redacted, [], has_pii, False,
                   time.perf_counter() - t0)


def run_mode_b_visual_only(
    image: PILImage | None,
    owl_processor,
    owl_model,
    device: str,
    queries: list[list[str]] | None = None,
) -> dict:
    """Mode B: OwlViT zero-shot detection only; no OCR, no text filter."""
    t0 = time.perf_counter()
    if image is None:
        return _result("B", "Visual-Only (OwlViT)", "", [], False, False, 0.0)
    boxes = _owl_detect(image, owl_processor, owl_model, device, queries)
    boxes = _post_process_boxes(boxes, image)
    return _result("B", "Visual-Only (OwlViT)", "", boxes, False,
                   len(boxes) > 0, time.perf_counter() - t0)


def run_mode_c_visual_ocr(
    image: PILImage | None,
    owl_processor,
    owl_model,
    device: str,
    ocr_reader,
    pii_filter,
    queries: list[list[str]] | None = None,
) -> dict:
    """Mode C: OwlViT + ephemeral OCR; text input not redacted."""
    t0 = time.perf_counter()
    if image is None:
        return _result("C", "Visual + OCR", "", [], False, False, 0.0)
    boxes = _owl_detect(image, owl_processor, owl_model, device, queries)
    boxes += _ocr_boxes(image, ocr_reader, pii_filter)
    boxes = _post_process_boxes(boxes, image)
    return _result("C", "Visual + OCR", "", boxes, False,
                   len(boxes) > 0, time.perf_counter() - t0)


def run_mode_d_full(
    text: str | None,
    image: PILImage | None,
    pii_filter,
    owl_processor,
    owl_model,
    device: str,
    ocr_reader,
    queries: list[list[str]] | None = None,
) -> dict:
    """Mode D: full pipeline — text PII + OwlViT + ephemeral OCR."""
    t0 = time.perf_counter()
    redacted, has_text_pii = _redact(text, pii_filter)
    if image is None:
        return _result("D", "Full Pipeline (Ours)", redacted, [],
                       has_text_pii, False, time.perf_counter() - t0)
    boxes = _owl_detect(image, owl_processor, owl_model, device, queries)
    boxes += _ocr_boxes(image, ocr_reader, pii_filter)
    boxes = _post_process_boxes(boxes, image)
    return _result("D", "Full Pipeline (Ours)", redacted, boxes,
                   has_text_pii, len(boxes) > 0, time.perf_counter() - t0)


def compare_all(
    text: str | None,
    image: PILImage | None,
    pii_filter,
    owl_processor,
    owl_model,
    device: str,
    ocr_reader,
    queries: list[list[str]] | None = None,
) -> list[dict]:
    """Run all four modes and return a list of result dicts."""
    return [
        run_mode_a_text_only(text, pii_filter),
        run_mode_b_visual_only(image, owl_processor, owl_model, device, queries),
        run_mode_c_visual_ocr(image, owl_processor, owl_model, device,
                              ocr_reader, pii_filter, queries),
        run_mode_d_full(text, image, pii_filter, owl_processor, owl_model,
                        device, ocr_reader, queries),
    ]


def compute_utility_score(
    original_image: PILImage | None,
    masked_image: PILImage | None,
    clip_processor,
    clip_model,
    device: str,
) -> float:
    """Cosine similarity between original and masked CLIP image embeddings.

    1.0 = identical semantic content; lower = more utility lost.
    Used to show that MASK-policy inpainting preserves more utility than
    ABSTRACT-policy (full suppression).
    """
    if original_image is None or masked_image is None:
        return float("nan")

    def _embed(img: PILImage) -> torch.Tensor:
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
            emb = clip_model.visual_projection(out.pooler_output)
        return emb / emb.norm(dim=-1, keepdim=True)

    orig = _embed(original_image)
    masked = _embed(masked_image)
    return round((orig * masked).sum().item(), 4)


# ── Private helpers ───────────────────────────────────────────────────────────

def _redact(text: str | None, pii_filter) -> tuple[str, bool]:
    if not text or not str(text).strip():
        return "", False
    text = str(text)
    spans = pii_filter(text)
    if not spans:
        return text, False
    redacted = text
    for s in sorted(spans, key=lambda x: x["start"], reverse=True):
        tag = f"[{s.get('entity_group', s.get('entity', 'PII')).upper()}]"
        redacted = redacted[: s["start"]] + tag + redacted[s["end"]:]
    return redacted, True


def _owl_detect(
    image: PILImage,
    owl_processor,
    owl_model,
    device: str,
    queries: list[list[str]] | None,
) -> list[list[int]]:
    if queries is None:
        queries = [["face", "passport", "drivers license",
                    "credit card", "medical prescription", "laptop screen"]]
    inputs = owl_processor(text=queries, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = owl_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = owl_processor.post_process_grounded_object_detection(
        outputs, threshold=0.1, target_sizes=target_sizes,
    )[0]
    return [[int(v) for v in box.tolist()] for box in results["boxes"]]


def _ocr_boxes(image: PILImage, reader, pii_filter) -> list[list[int]]:
    boxes = []
    for bbox, text, _prob in reader.readtext(np.array(image)):
        if not text or not str(text).strip():
            continue
        _, has_pii = _redact(str(text), pii_filter)
        if has_pii:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
    return boxes


def _post_process_boxes(boxes: list[list[int]], image: PILImage) -> list[list[int]]:
    from src.privacy.box_utils import expand_box, nms_boxes
    w, h = image.size
    expanded = [expand_box(b, w, h) for b in boxes]
    return nms_boxes(expanded)


def _result(
    mode_id: str,
    mode_name: str,
    redacted_text: str,
    boxes: list,
    text_pii_found: bool,
    visual_pii_found: bool,
    elapsed: float,
) -> dict:
    return {
        "mode_id": mode_id,
        "mode_name": mode_name,
        "redacted_text": redacted_text,
        "sensitive_boxes": boxes,
        "text_pii_found": text_pii_found,
        "visual_pii_found": visual_pii_found,
        "n_boxes": len(boxes),
        "elapsed_s": round(elapsed, 3),
    }
