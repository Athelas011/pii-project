"""
Privacy-Aware Agent Runtime — Pipeline Module

All core detection, policy, embedding, and visualization logic lives here.
Notebooks import from this module and stay thin: pip installs, plt.show(),
and demo orchestration only.

Stage-aware privacy control: π(E, s) → a
  E = detected sensitive entities
  s ∈ {SENSING, MEMORY_WRITE, RETRIEVAL, OUTPUT}
  a ∈ {ALLOW, MASK, ABSTRACT, TEXT_ONLY, EMPTY, INVALID_IMAGE}
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Literal

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

# ── Resolution constants ──────────────────────────────────────────────────────
# SD inpainting supports 768px; was 512 — higher means less detail loss on
# large source images (3000 px+) that are downsampled for inpainting.
_INPAINT_SIZE: tuple[int, int] = (768, 768)

# PDF rasterization: 300 DPI gives 2550×3300 px for a letter page — enough
# for crisp OCR and OwlViT detection. Was 200.
_PDF_DPI: int = 300

# Jupyter inline display DPI — default is ~96; 150 gives noticeably crisper
# inline rendering without blowing up cell height.
_DISPLAY_DPI: int = 150

# File save DPI for matplotlib figures exported to outputs/.
_SAVE_DPI: int = 300

# Apply inline DPI at import time so all notebook cells benefit automatically.
plt.rcParams["figure.dpi"] = _DISPLAY_DPI

# ── Module-level model globals (populated by initialize_models()) ─────────────
_device: str = "cpu"
_pii_filter = None
_reader = None
_owl_processor = None
_owl_model = None
_inpaint_pipe = None
_clip_processor = None
_clip_model = None

_SENSITIVE_QUERIES: list[list[str]] = [[
    "face", "passport", "drivers license",
    "credit card", "medical prescription", "laptop screen",
    "ID card", "student card", "barcode",
]]

# Regex patterns for sensitive numbers that NER models frequently miss when
# presented as bare digits without surrounding prose context.
_CARD_RE = re.compile(
    r'\b(?:\d[ \-]?){13,18}\d\b'   # 14–19 digit runs: Visa/MC/Amex/Discover
)
_SSN_RE  = re.compile(r'\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b')
# Alphanumeric IDs: optional 1-2 letter prefix + 8+ digits (student/member cards,
# barcode values, government IDs).  Anchored to word boundary so short tokens
# like room numbers don't fire.
_ALPHANUM_ID_RE = re.compile(r'\b[A-Za-z]{0,2}\d{8,}\b')

# OpenCV Haar cascade paths — frontal + profile cover most head orientations.
_HAAR_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_HAAR_PROFILE = cv2.data.haarcascades + "haarcascade_profileface.xml"

PolicyType = Literal["ALLOW", "MASK", "ABSTRACT", "TEXT_ONLY", "EMPTY", "INVALID_IMAGE"]

# ── Lazy imports (heavy) ──────────────────────────────────────────────────────
# Imported inside initialize_models() so the module can be imported cheaply.

# ── Box utilities (already a clean module) ────────────────────────────────────
from src.privacy.box_utils import (  # noqa: E402
    calculate_box_area,
    draw_boxes_on_image,
    expand_box,
    nms_boxes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────────────────────────

def setup_output_dir(base_path: str | Path | None = None) -> Path:
    """Create and return the outputs/ directory at project root.

    Args:
        base_path: Override root. Defaults to the project root (two levels up
                   from this file's src/ directory).

    Returns:
        Path to outputs/ (created if absent).
    """
    if base_path is None:
        base_path = Path(__file__).resolve().parent.parent / "outputs"
    else:
        base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


# ─────────────────────────────────────────────────────────────────────────────
# Model initialization
# ─────────────────────────────────────────────────────────────────────────────

def initialize_models(
    device: str | None = None,
    load_inpainter: bool = True,
    visual_queries: list[list[str]] | None = None,
) -> None:
    """Load all foundation models into module-level globals.

    Args:
        device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
        load_inpainter: Set False to skip Stable Diffusion (faster for
                        text-only demos). MASK policy will degrade to ABSTRACT.
        visual_queries: Override the default OwlViT query list.
    """
    import easyocr
    from diffusers import AutoPipelineForInpainting
    from transformers import (
        CLIPModel,
        CLIPProcessor,
        OwlViTForObjectDetection,
        OwlViTProcessor,
        pipeline,
    )

    global _device, _pii_filter, _reader, _owl_processor, _owl_model
    global _inpaint_pipe, _clip_processor, _clip_model, _SENSITIVE_QUERIES

    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading local models onto {_device}...")

    if visual_queries is not None:
        _SENSITIVE_QUERIES = visual_queries

    # 1. Text PII filter
    _pii_filter = pipeline(
        task="token-classification",
        model="openai/privacy-filter",
        aggregation_strategy="simple",
        device=0 if _device == "cuda" else -1,
    )

    # 2. Ephemeral OCR reader
    _reader = easyocr.Reader(["en"], gpu=(_device == "cuda"))

    # 3. OwlViT zero-shot visual detector
    _owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    _owl_model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32"
    ).to(_device)

    # 4. Stable Diffusion inpainting (optional)
    if load_inpainter:
        _inpaint_kwargs = (
            {"torch_dtype": torch.float16, "variant": "fp16"}
            if _device == "cuda"
            else {"torch_dtype": torch.float32}
        )
        _inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            **_inpaint_kwargs,
        ).to(_device)
    else:
        _inpaint_pipe = None
        print("  [INFO] Inpainter skipped — MASK policy will degrade to ABSTRACT.")

    # 5. CLIP for safe vectorization
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)

    print("All local models loaded successfully.")


def get_models() -> dict:
    """Return a dict of loaded model references.

    Use this to pass models to baseline_comparison.compare_all() and
    compute_utility_score() without re-initializing.

    Raises:
        RuntimeError: if initialize_models() has not been called yet.
    """
    if _pii_filter is None:
        raise RuntimeError("Call initialize_models() before get_models().")
    return {
        "device": _device,
        "pii_filter": _pii_filter,
        "reader": _reader,
        "owl_processor": _owl_processor,
        "owl_model": _owl_model,
        "inpaint_pipe": _inpaint_pipe,
        "clip_processor": _clip_processor,
        "clip_model": _clip_model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image / PDF loading
# ─────────────────────────────────────────────────────────────────────────────

def load_images_from_path(
    file_path: str | Path | None,
    dpi: int = _PDF_DPI,
) -> list[Image.Image]:
    """Load an image file or PDF and return a list of PIL RGB images.

    Args:
        file_path: Path to .jpg/.png/.pdf, or None/empty → returns [].
        dpi: Resolution for PDF rasterization. Default is _PDF_DPI (300).

    Returns:
        List of PIL RGB images. Empty list if file is missing or path is None.
    """
    if file_path is None or not str(file_path).strip():
        return []

    file_path = str(file_path)
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return []

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        from pdf2image import convert_from_path
        pages = convert_from_path(file_path, dpi=dpi)
        return [page.convert("RGB") for page in pages]

    return [Image.open(file_path).convert("RGB")]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 & 2: Sensing Gate — Text + Image detection
# ─────────────────────────────────────────────────────────────────────────────

def redact_text(
    text: str | None,
    filter_pipe=None,
) -> tuple[str, bool]:
    """Replace PII spans in text with [ENTITY_TYPE] tags.

    Works backwards through spans so earlier character offsets stay valid
    after each substitution.

    Args:
        text: Input string.
        filter_pipe: HuggingFace NER pipeline. Falls back to module-level
                     _pii_filter when None.

    Returns:
        (redacted_text, has_pii)
    """
    pipe = filter_pipe if filter_pipe is not None else _pii_filter
    if text is None:
        return "", False
    text = str(text)
    if not text.strip():
        return text, False

    spans = pipe(text)
    if not spans:
        return text, False

    redacted = text
    for s in sorted(spans, key=lambda x: x["start"], reverse=True):
        tag = f"[{s.get('entity_group', s.get('entity', 'PII')).upper()}]"
        redacted = redacted[: s["start"]] + tag + redacted[s["end"]:]

    return redacted, True


def detect_privacy_risks_from_image(
    image: Image.Image,
    use_ocr: bool = True,
    visual_queries: list[list[str]] | None = None,
    owl_threshold: float = 0.1,
) -> list[list[int]]:
    """Detect sensitive regions in a single PIL image.

    Steps:
      1. OwlViT zero-shot object detection.
      2. Optional ephemeral OCR — bounding boxes only, strings deleted
         immediately (ephemeral OCR principle: text never retained).
      3. expand_box + nms_boxes deduplication.

    Args:
        image: PIL RGB image.
        use_ocr: Whether to run EasyOCR step.
        visual_queries: Override module-level _SENSITIVE_QUERIES.
        owl_threshold: OwlViT confidence threshold.

    Returns:
        Deduplicated list of [xmin, ymin, xmax, ymax] boxes.
    """
    queries = visual_queries if visual_queries is not None else _SENSITIVE_QUERIES
    image_w, image_h = image.size
    if image_w == 0 or image_h == 0:
        return []

    raw_boxes: list[list[int]] = []

    # ── 1. OwlViT visual object detection ─────────────────────────────────────
    inputs = _owl_processor(
        text=queries, images=image, return_tensors="pt"
    ).to(_device)
    with torch.no_grad():
        outputs = _owl_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(_device)
    results = _owl_processor.post_process_grounded_object_detection(
        outputs, threshold=owl_threshold, target_sizes=target_sizes,
    )[0]
    for box in results["boxes"]:
        raw_boxes.append([int(v) for v in box.tolist()])

    # ── 1.5. Dedicated face detection (Haar cascade) ──────────────────────────
    # OwlViT zero-shot is unreliable for faces in real photos; Haar cascades
    # provide deterministic coverage for frontal and profile orientations.
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    for cascade_path in (_HAAR_FRONTAL, _HAAR_PROFILE):
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces):
            for (x, y, w, h) in faces:
                raw_boxes.append([int(x), int(y), int(x + w), int(y + h)])

    # ── 2. Ephemeral OCR text detection ───────────────────────────────────────
    if use_ocr:
        ocr_results = _reader.readtext(np.array(image))
        for bbox, ocr_text, _prob in ocr_results:
            if not ocr_text or not str(ocr_text).strip():
                continue
            _redacted, has_pii = redact_text(ocr_text)
            # Supplement NER with regex: card numbers, SSNs, and alphanumeric IDs
            # are frequently missed by token-classification models when the digits
            # appear without surrounding prose context.
            has_sensitive_pattern = bool(
                _CARD_RE.search(ocr_text)
                or _SSN_RE.search(ocr_text)
                or _ALPHANUM_ID_RE.search(ocr_text)
            )
            if has_pii or has_sensitive_pattern:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                raw_boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
        del ocr_results  # ephemeral — OCR strings leave no trace

    # ── 3. Expand + deduplicate ────────────────────────────────────────────────
    expanded = [expand_box(b, image_w, image_h) for b in raw_boxes]
    return nms_boxes(expanded)


def detect_privacy_risks(
    user_text: str | None = None,
    image_path: str | Path | None = None,
    use_ocr: bool = True,
    pdf_dpi: int = _PDF_DPI,
) -> list[dict]:
    """Detect privacy risks across text and image/PDF inputs.

    Handles all five input combinations:
      - text + image  (with or without OCR)
      - text only
      - image only    (with or without OCR)
      - empty input

    Args:
        user_text: Raw input text (may contain PII).
        image_path: Path to .jpg/.png/.pdf, or None.
        use_ocr: Whether to run ephemeral OCR on images.
        pdf_dpi: DPI for PDF rasterization.

    Returns:
        List of per-page result dicts, each with keys:
          image           : PIL RGB image or None
          redacted_text   : str (PII replaced with [ENTITY_TYPE] tags)
          sensitive_boxes : list of [xmin, ymin, xmax, ymax]
          page_index      : int or None
          source_path     : str or None
    """
    # Text PII detection runs once and is shared across all pages.
    if user_text is not None and str(user_text).strip():
        redacted_user_text, _ = redact_text(user_text)
    else:
        redacted_user_text = ""

    images = load_images_from_path(image_path, dpi=pdf_dpi)

    if not images:
        return [{
            "image": None,
            "redacted_text": redacted_user_text,
            "sensitive_boxes": [],
            "page_index": None,
            "source_path": str(image_path) if image_path else None,
        }]

    return [
        {
            "image": img,
            "redacted_text": redacted_user_text,
            "sensitive_boxes": detect_privacy_risks_from_image(img, use_ocr=use_ocr),
            "page_index": i,
            "source_path": str(image_path) if image_path else None,
        }
        for i, img in enumerate(images)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 & 4: Policy Engine + Memory Write Gate
# ─────────────────────────────────────────────────────────────────────────────

def embed_safe_memory(
    text: str | None = None,
    image: Image.Image | None = None,
) -> dict:
    """CLIP embedding for safe (post-gate) content.

    Avoids get_text_features / get_image_features because in
    transformers 5.x those helpers return BaseModelOutputWithPooling
    rather than a plain tensor. Sub-models are called directly.

    Returns:
        {"text_embeds": Tensor | None, "image_embeds": Tensor | None}
    """
    has_text = text is not None and str(text).strip()
    has_image = image is not None
    out: dict = {"text_embeds": None, "image_embeds": None}

    if not has_text and not has_image:
        return out

    if has_text and has_image:
        inputs = _clip_processor(
            text=[str(text)], images=image, return_tensors="pt", padding=True,
        ).to(_device)
        with torch.no_grad():
            result = _clip_model(**inputs)
        out["text_embeds"] = result.text_embeds
        out["image_embeds"] = result.image_embeds

    elif has_text:
        inputs = _clip_processor(
            text=[str(text)], return_tensors="pt", padding=True,
        ).to(_device)
        with torch.no_grad():
            text_out = _clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            out["text_embeds"] = _clip_model.text_projection(text_out.pooler_output)

    else:
        inputs = _clip_processor(images=image, return_tensors="pt").to(_device)
        with torch.no_grad():
            vision_out = _clip_model.vision_model(
                pixel_values=inputs["pixel_values"]
            )
            out["image_embeds"] = _clip_model.visual_projection(
                vision_out.pooler_output
            )

    return out


def privacy_gate_and_embed(
    image: Image.Image | None = None,
    redacted_text: str = "",
    sensitive_boxes: list[list[int]] | None = None,
    mask_threshold: float = 0.30,
) -> tuple[PolicyType, Image.Image | None, str, dict, str]:
    """Policy engine and memory write gate.

    Determines the privacy action for a single memory event and produces
    a safe CLIP embedding ready for vector DB insertion.

    Policy decisions (based on sensitive_area / total_area):
      EMPTY         — no text or image provided
      TEXT_ONLY     — text present, no image
      INVALID_IMAGE — image with zero area
      ALLOW         — sensitive_ratio == 0
      MASK          — 0 < sensitive_ratio < mask_threshold: paint boxes solid black
      ABSTRACT      — sensitive_ratio >= mask_threshold: block image

    Args:
        image: PIL RGB image or None.
        redacted_text: Already-redacted text string.
        sensitive_boxes: List of [xmin, ymin, xmax, ymax] boxes.
        mask_threshold: Area ratio above which ABSTRACT replaces MASK.

    Returns:
        (policy, final_image, abstract_summary, embeddings, redacted_text)
    """
    if sensitive_boxes is None:
        sensitive_boxes = []
    redacted_text = str(redacted_text) if redacted_text else ""

    has_text = bool(redacted_text.strip())
    has_image = image is not None

    # ── Case 0: nothing provided ───────────────────────────────────────────────
    if not has_text and not has_image:
        return (
            "EMPTY", None,
            "No valid text or image input was provided.",
            embed_safe_memory(), redacted_text,
        )

    # ── Case 1: text only ──────────────────────────────────────────────────────
    if has_text and not has_image:
        return (
            "TEXT_ONLY", None,
            "Text-only memory entry. No image was provided.",
            embed_safe_memory(text=redacted_text), redacted_text,
        )

    # ── Case 2: image present → apply image policy ─────────────────────────────
    image_w, image_h = image.size
    total_area = image_w * image_h

    if total_area == 0:
        memory_text = redacted_text if has_text else "Image has invalid size."
        return (
            "INVALID_IMAGE", None,
            "Image has invalid size and was blocked.",
            embed_safe_memory(text=memory_text), redacted_text,
        )

    sensitive_area = sum(calculate_box_area(b) for b in sensitive_boxes)
    sensitive_ratio = sensitive_area / total_area

    # ── ALLOW ──────────────────────────────────────────────────────────────────
    if sensitive_ratio == 0:
        policy = "ALLOW"
        final_image = image
        summary = "Raw image allowed — no sensitive regions detected."

    # ── MASK ───────────────────────────────────────────────────────────────────
    elif sensitive_ratio < mask_threshold:
        policy = "MASK"
        print(
            f"  [MASK] Sensitive coverage {sensitive_ratio:.1%} — "
            f"painting {len(sensitive_boxes)} region(s) solid black..."
        )
        final_image = image.copy()
        draw = ImageDraw.Draw(final_image)
        for box in sensitive_boxes:
            draw.rectangle(box, fill=(0, 0, 0))
        summary = (
            f"Image masked — {len(sensitive_boxes)} sensitive region(s) "
            f"painted solid black (original resolution preserved)."
        )

    # ── ABSTRACT ───────────────────────────────────────────────────────────────
    else:
        policy = "ABSTRACT"
        print(f"  [ABSTRACT] Sensitive coverage {sensitive_ratio:.1%} — blocking image.")
        final_image = None
        summary = (
            "ABSTRACT SEMANTIC SUMMARY: Image contained a high density of "
            "sensitive personal features and was blocked from memory."
        )

    # ── Memory write: produce safe CLIP embedding ──────────────────────────────
    if final_image is not None:
        embeddings = embed_safe_memory(
            text=redacted_text if has_text else None,
            image=final_image,
        )
    else:
        memory_text = (redacted_text + " " + summary).strip() if has_text else summary
        embeddings = embed_safe_memory(text=memory_text)

    return policy, final_image, summary, embeddings, redacted_text


# ─────────────────────────────────────────────────────────────────────────────
# Output saving utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_pipeline_image(
    image: Image.Image,
    save_path: str | Path,
    label: str = "",
) -> Path:
    """Save a PIL image directly to disk (bypasses matplotlib).

    PIL saves preserve full pixel fidelity — no Agg renderer resampling.
    Use this for original captures, inpainted results, and annotated overlays.

    Args:
        image: PIL Image to save.
        save_path: Destination path (must include filename + extension).
        label: Optional label for the log message.

    Returns:
        Resolved Path of the saved file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
    tag = f" [{label}]" if label else ""
    print(f"  Saved{tag}: {path}")
    return path


def _save_figure(
    fig: matplotlib.figure.Figure,
    save_path: str | Path,
) -> None:
    """Save a matplotlib figure at _SAVE_DPI with tight bounding box."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=_SAVE_DPI, bbox_inches="tight")
    print(f"  Saved figure: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_detection_result(
    result: dict,
    policy: PolicyType | None = None,
    final_image: Image.Image | None = None,
    title: str = "Privacy Gate Result",
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Render original / detected-boxes / after-gate panels.

    Args:
        result: Dict from detect_privacy_risks() — must contain 'image'
                and 'sensitive_boxes'.
        policy: Policy string shown in the after-gate panel title.
        final_image: Post-gate image (ALLOW/MASK output). None for ABSTRACT.
        title: Figure suptitle.
        save_path: If given, saves the figure at _SAVE_DPI AND saves each PIL
                   panel as a separate .png with _original/_detected/_result
                   suffixes, adjacent to the figure file.

    Returns:
        The matplotlib Figure object (call plt.show() in the notebook).
    """
    raw_image = result.get("image")
    boxes = result.get("sensitive_boxes", [])

    if raw_image is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No image (text-only or abstract)",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path is not None:
            _save_figure(fig, save_path)
        return fig

    n_panels = 3 if final_image is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 9))
    if n_panels == 1:
        axes = [axes]

    # Panel 0: original
    axes[0].imshow(raw_image)
    axes[0].set_title("Original  (unsafe)", fontsize=13, color="#c0392b",
                      fontweight="bold", pad=10)
    axes[0].axis("off")

    # Panel 1: detected boxes
    annotated = draw_boxes_on_image(raw_image, boxes)
    axes[1].imshow(annotated)
    axes[1].set_title(f"Detected regions  ({len(boxes)})", fontsize=13,
                      color="#2980b9", fontweight="bold", pad=10)
    axes[1].axis("off")

    # Panel 2: after gate (ALLOW or MASK only)
    if final_image is not None:
        policy_label = f"After gate  [{policy}]" if policy else "After gate"
        axes[2].imshow(final_image)
        axes[2].set_title(policy_label, fontsize=13, color="#27ae60",
                          fontweight="bold", pad=10)
        axes[2].axis("off")

    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()

    # High-res file saves
    if save_path is not None:
        save_path = Path(save_path)
        _save_figure(fig, save_path)
        stem = save_path.stem
        parent = save_path.parent
        save_pipeline_image(raw_image, parent / f"{stem}_original.png", "original")
        save_pipeline_image(annotated, parent / f"{stem}_detected.png", "detected")
        if final_image is not None:
            pol_tag = policy.lower() if policy else "result"
            save_pipeline_image(
                final_image, parent / f"{stem}_result_{pol_tag}.png", pol_tag
            )

    return fig


def visualize_comparison(
    results: list[dict],
    image: Image.Image,
    title: str = "Baseline Comparison",
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Render side-by-side Mode A/B/C/D detection comparison.

    Args:
        results: Output of baseline_comparison.compare_all() — list of 4 dicts.
        image: The PIL image input used in the comparison.
        title: Figure suptitle.
        save_path: If given, saves figure at _SAVE_DPI.

    Returns:
        The matplotlib Figure object (call plt.show() in the notebook).
    """
    fig, axes = plt.subplots(1, 4, figsize=(10 * 4, 10))
    colors = ["#e74c3c", "#f39c12", "#3498db", "#27ae60"]

    for ax, r, color in zip(axes, results, colors):
        vis = draw_boxes_on_image(image, r["sensitive_boxes"]) if r["sensitive_boxes"] else image
        ax.imshow(vis)
        box_label = f"{r['n_boxes']} box(es)" if r["n_boxes"] else "nothing detected"
        ax.set_title(
            f"Mode {r['mode_id']}: {r['mode_name']}\n{box_label}",
            fontsize=12, fontweight="bold", color=color, pad=10,
        )
        ax.axis("off")

    # Highlight the winning mode (D)
    for spine in axes[3].spines.values():
        spine.set_edgecolor("#27ae60")
        spine.set_linewidth(4)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path)

    return fig


def visualize_utility(
    original: Image.Image,
    masked: Image.Image,
    boxes: list[list[int]],
    utility_score: float,
    fname: str = "",
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Render original / detected / masked 3-panel utility visualization.

    Args:
        original: Original (unsafe) PIL image.
        masked: Post-gate (MASK policy) PIL image.
        boxes: Detected sensitive boxes.
        utility_score: CLIP cosine similarity score.
        fname: Filename label for titles.
        save_path: If given, saves figure at _SAVE_DPI AND saves each PIL
                   panel with _original/_detected/_masked suffixes.

    Returns:
        The matplotlib Figure object (call plt.show() in the notebook).
    """
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    annotated = draw_boxes_on_image(original, boxes)

    axes[0].imshow(original)
    axes[0].set_title(
        f"{fname}\nOriginal (unsafe)", fontsize=13, color="#c0392b", fontweight="bold"
    )
    axes[0].axis("off")

    axes[1].imshow(annotated)
    axes[1].set_title(
        f"Detected regions ({len(boxes)})", fontsize=13, color="#2980b9", fontweight="bold"
    )
    axes[1].axis("off")

    axes[2].imshow(masked)
    score_str = f"{utility_score:.3f}" if utility_score == utility_score else "N/A"
    axes[2].set_title(
        f"After MASK gate\nCLIP utility = {score_str}",
        fontsize=13, color="#27ae60", fontweight="bold",
    )
    axes[2].axis("off")

    plt.suptitle(
        "Utility Preservation: Generative Inpainting vs. Full Suppression",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        _save_figure(fig, save_path)
        stem = save_path.stem
        parent = save_path.parent
        save_pipeline_image(original, parent / f"{stem}_original.png", "original")
        save_pipeline_image(annotated, parent / f"{stem}_detected.png", "detected")
        save_pipeline_image(masked, parent / f"{stem}_masked.png", "masked")

    return fig
