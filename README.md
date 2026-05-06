# Privacy-Gated Memory for Multimodal Agents

CS466 Final Project — Yilin Pan, Tyler Hudgins

A privacy gate for multimodal agent memory systems that intercepts sensitive
content in **both text and images** before it is embedded into a vector store.
Once content is encoded into a CLIP vector it cannot be meaningfully extracted
or removed, so the gate must run at the sensing and memory-write stages — not
as a post-hoc output filter.

In a controlled detective-game evaluation, the full pipeline reduces sensitive
attribute recovery by **76%** (0.63 → 0.15) versus an unguarded baseline,
while preserving 100% identity-level task accuracy. Text-only NER filtering
alone reduces recovery by only 6%, confirming that visual masking is required
for any agent that accepts images.

## Pipeline

```
        ┌──────────────── Sensing Gate ─────────────────┐
input → │  NER text redaction                           │
        │  OwlViT zero-shot visual detection            │ → boxes + redacted text
        │  Haar cascade face detection (frontal+profile)│
        │  Ephemeral OCR (boxes kept, strings dropped)  │
        └───────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─── Policy Engine π(E, s) ───┐
                    │  ALLOW         (no PII)     │
                    │  MASK          (< 30% area) │ → solid-black boxes
                    │  ABSTRACT      (≥ 30% area) │ → image blocked
                    │  TEXT_ONLY / EMPTY / …      │
                    └─────────────────────────────┘
                              │
                              ▼
                    CLIP embedding → ChromaDB
```

The 30% area threshold is configurable via `mask_threshold` in
`privacy_gate_and_embed`.

## Repository layout

```
src/
  privacy_pipeline.py         # all detection, policy, embedding, visualization
  agent/baseline_agent.py     # unguarded GPT-4o agent (leakage demo)
  retrieval/                  # ChromaDB + image embedding retrieval
  memory/                     # Chroma + GCS clients
  adapters/                   # CLIP and text embedders
  evaluation/                 # baseline comparison (modes A/B/C/D)
  privacy/box_utils.py        # NMS, box expansion, drawing
  config/settings.py          # env-driven configuration

config/test_inputs.yaml       # editable test cases, queries, prompts
demo_dataset/                 # face / email / prescription / codebase samples

baseline_demo.ipynb           # leakage demo (no gate) — shows raw PII in output
privacy_agent.ipynb           # full gated pipeline — end-to-end walkthrough
evaluation.ipynb              # detective-game evaluation
Data_Preprocessing_CS466.ipynb

REPORT.tex                    # final report
requirements.txt
```

## Setup

Requires Python 3.10+. CUDA is optional but speeds up Stable Diffusion
inpainting and OwlViT considerably.

```powershell
pip install -r requirements.txt
```

Create a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
# Optional — only needed for the GCS / BIV-Priv URL exposure demo
GCS_BUCKET_NAME=
GOOGLE_APPLICATION_CREDENTIALS=
# Optional overrides
CHROMA_DB_PATH=data/chroma_db
TOP_K=5
```

## Models

Loaded by `initialize_models()` in `src/privacy_pipeline.py`:

| Component        | Model                                   |
|------------------|-----------------------------------------|
| Text NER         | `dslim/bert-base-NER`                   |
| OCR (ephemeral)  | EasyOCR (en)                            |
| Visual detection | `google/owlvit-base-patch32`            |
| Face detection   | OpenCV Haar cascades (frontal, profile) |
| Inpainting       | `runwayml/stable-diffusion-inpainting`  |
| Safe embedding   | `openai/clip-vit-base-patch32`          |
| Agent LLM        | `gpt-4o` (via OpenAI API)               |

Pass `load_inpainter=False` to skip Stable Diffusion for faster startup —
`MASK` will degrade to `ABSTRACT` in that mode.

## Quick start

```python
from src.privacy_pipeline import (
    initialize_models, detect_privacy_risks, privacy_gate_and_embed,
)

initialize_models()                       # loads all models
results = detect_privacy_risks(
    user_text="My SSN is 102-93-8564.",
    image_path="demo_dataset/test_prescription.pdf",
)
for r in results:
    policy, final_image, summary, embeds, redacted = privacy_gate_and_embed(
        image=r["image"],
        redacted_text=r["redacted_text"],
        sensitive_boxes=r["sensitive_boxes"],
    )
    print(policy, "—", summary)
```

## Notebooks

Run in order for the demo flow:

1. **`baseline_demo.ipynb`** — agent with no gate. Raw PII, GCS URIs, and
   image URLs flow into GPT-4o and back out in the response.
2. **`privacy_agent.ipynb`** — same agent with the full sensing gate +
   policy engine wired in. Visualizes original / detected boxes / post-gate
   panels for each test case.
3. **`evaluation.ipynb`** — detective-game evaluation. GPT-4o is asked to
   match preprocessed identity documents to one of three suspect profiles
   under three conditions (no gate, text-only filter, full pipeline).

Test inputs, OwlViT queries, and the detective-game prompts are all editable
in `config/test_inputs.yaml` — no notebook changes required.

## Baseline comparison modes

Run via `src/evaluation/baseline_comparison.py`:

| Mode | Components                        | Purpose                              |
|------|-----------------------------------|--------------------------------------|
| A    | NER text only                     | Image completely ignored             |
| B    | OwlViT only                       | No OCR, no text filter               |
| C    | OwlViT + ephemeral OCR            | Text input not redacted              |
| D    | NER + OwlViT + Haar + OCR (full)  | Full pipeline (ours)                 |

## Notes on design

- **Ephemeral OCR.** OCR is used for box localization only; the recognized
  strings are dropped immediately after the box is recorded so PII from
  document images is never retained in the pipeline beyond the sensing stage.
- **Regex supplements NER.** Bare digit runs (cards, SSNs, alphanumeric IDs)
  are frequently missed by token-classification NER without surrounding prose,
  so `_CARD_RE`, `_SSN_RE`, and `_ALPHANUM_ID_RE` patterns run alongside.
- **Haar cascades for faces.** OwlViT zero-shot face detection is unreliable
  on real photos; Haar provides deterministic frontal + profile coverage.
- **Pre-embedding enforcement.** All masking happens on the PIL image before
  it ever reaches `clip_processor`. There is no path for an unmasked region
  to enter the vector store.

## Report

See `REPORT.tex` for the full write-up, evaluation methodology, results
tables, and related work.
