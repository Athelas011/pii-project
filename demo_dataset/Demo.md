### 4.2 The Detection Models — Why We Chose Them

**Model 1: `dslim/bert-base-NER` (Text NER)**

- **What it is:** A BERT-based Named Entity Recognition model fine-tuned on CoNLL-2003, one of the most standard NER benchmarks. It detects PER (person names), LOC (locations), ORG (organizations), and MISC entities.
- **Why we use it:** It's universally supported, requires no custom code, runs on any Python version and any device. It catches the most common PII types in free-form text.
- **How it works:** It tokenizes input text, then for each token predicts a label from the BIO tagging scheme (B-PER = beginning of a person name, I-PER = continuation, O = not an entity). Aggregation mode groups consecutive tokens into spans. Output is a list of entities with start/end character positions and confidence scores.
- **Why it's safe for PII:** Text is processed locally. No data leaves the machine. The model only identifies what is sensitive — it doesn't store or transmit the text. The spans it identifies are used to place redaction tags, and the original text is never retained.
- **Limitation:** It doesn't catch structured identifiers like credit card numbers or SSNs that appear without surrounding context (e.g., just "4532 1234 5678 9012" with no accompanying text). This is addressed by the regex layer below.

**Model 2: EasyOCR (Optical Character Recognition)**

- **What it is:** A deep learning OCR library that uses a CRNN architecture (CNN for visual feature extraction + RNN for sequence reading). It reads text from arbitrary image regions and returns the text string plus a bounding box.
- **Why we use it:** It's accurate across many fonts and image qualities, runs entirely locally, and supports 80+ languages. No cloud API needed.
- **How it works:** Takes a PIL image, runs it through a convolutional network to detect text regions, then reads each region character-by-character using a recurrent sequence model. Returns `(bounding_box, text_string, confidence)` tuples.
- **Ephemeral design (important for safety):** We immediately delete the OCR text strings after checking them — `del ocr_results`. Only the bounding boxes are retained if PII was found. This means the extracted text never persists in memory, only the coordinates of where it appeared.
- **Why it's safe:** The OCR text is treated as ephemeral — it exists only long enough to be checked by the NER model and regex, then deleted. Sensitive text is never stored.

**Model 3: `google/owlvit-base-patch32` (Zero-Shot Visual Object Detection)**

- **What it is:** OWL-ViT (Open-World object detection with Vision Transformers) is a zero-shot object detector from Google. It takes text queries and an image and returns bounding boxes of regions that semantically match the query — without needing to be trained on those specific categories.
- **Why we use it:** Standard object detectors (YOLO, Faster-RCNN) only detect predefined classes. OwlViT can detect arbitrary categories you describe in natural language: "passport", "credit card", "medical prescription", "ID card", "barcode". This makes it flexible for new document types without retraining.
- **How it works:** It encodes the text queries using a CLIP-like text encoder and the image using a ViT (Vision Transformer) image encoder. It then does cross-attention between every image patch and each text query, and predicts a bounding box where the similarity is high. Threshold 0.1 is used — meaning we accept even weak visual matches to maximize recall (we'd rather over-detect than miss a sensitive document).
- **Why it's safe:** Runs entirely locally. The model weights are from Google's public HuggingFace release. No image data is sent externally.
- **Limitation:** Zero-shot detection is less precise than fine-tuned detectors. It may miss objects when image quality is poor or when the object looks different from its text description. This is why we combine it with Haar cascades.

**Model 4: OpenCV Haar Cascades (Face Detection)**

- **What it is:** Classical machine learning detectors (not neural networks) based on Haar features — patterns of light and dark pixel regions. Trained using AdaBoost on thousands of positive (face) and negative (non-face) image patches.
- **Why we use it:** OwlViT is unreliable for detecting human faces in real photographs — it's designed for object-level detection, and "face" as a zero-shot query doesn't always fire. Haar cascades are specifically designed for face detection and are extremely fast, deterministic, and CPU-friendly. They work even on low-spec machines.
- **How it works:** Slides a detection window across the image at multiple scales. At each position, it evaluates a cascade of Haar feature classifiers — early stages reject non-faces quickly, later stages confirm faces. We run both `haarcascade_frontalface_default.xml` (frontal faces) and `haarcascade_profileface.xml` (side-facing).
- **Why it's safe:** Pure OpenCV, entirely local, no model download needed (ships with OpenCV). Deterministic — same input always gives same output.
- **Robustness consideration:** Haar cascades can produce false positives on textured surfaces. We use `minNeighbors=5` (a face must be confirmed by at least 5 overlapping detections at the same location) to reduce false positives. Profile detection catches faces that are turned sideways.

**Model 5: `openai/clip-vit-base-patch32` (Safe Embedding)**

- **What it is:** CLIP (Contrastive Language-Image Pre-Training) is a joint text-image embedding model from OpenAI. It maps both text and images into a shared vector space where semantically similar content is close together.
- **Why we use it:** The vector database stores CLIP embeddings of memory entries. By embedding the *post-gate* (already masked) version of an image, we ensure that what goes into the vector store is the sanitized representation, not the original sensitive content.
- **How it works:** Images are passed through a ViT encoder, text through a transformer. Both produce 512-dimensional vectors. Cosine similarity between vectors indicates semantic relatedness.
- **Why it's safe:** The key insight is that we embed *after* masking. If an ID card gets its text blacked out, the CLIP embedding of that masked image is what enters the vector DB — not the embedding of the original card. So even semantic similarity search cannot reconstruct the original content.

---

### 4.3 Regex Patterns (Supplementing NER)

NER models trained on news text often fail to recognize structured identifiers that appear without surrounding context. We added three regex patterns to catch what NER misses:

- `_CARD_RE`: Matches 14–19 digit sequences with optional spaces or dashes — covers Visa (16 digits), Mastercard (16), Amex (15), Discover (16), Diners (14). Pattern: `\b(?:\d[ \-]?){13,18}\d\b`
- `_SSN_RE`: Matches the canonical US Social Security Number format: `\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b`
- `_ALPHANUM_ID_RE`: Matches alphanumeric IDs like student cards (`A25031006422185`) — up to 2 letter prefix followed by 8+ digits: `\b[A-Za-z]{0,2}\d{8,}\b`

These fire in the OCR path: after EasyOCR extracts a text region, if NER says it's clean but the regex matches, the bounding box is still added to the sensitive list.

---

### 4.4 Policy Engine — How Decisions Are Made

The policy engine computes one number: `sensitive_ratio = sensitive_area / total_image_area`.

- **sensitive_ratio == 0** → **ALLOW**: No sensitive regions detected. Pass the original image through.
- **0 < sensitive_ratio < 0.30** → **MASK**: Some regions are sensitive but the image is mostly clean. Paint those specific regions solid black using `PIL.ImageDraw.rectangle`. The image stays the same resolution, the safe parts are intact.
- **sensitive_ratio ≥ 0.30** → **ABSTRACT**: More than 30% of the image is sensitive. Storing a partially-masked version is risky because context can still reconstruct the private content. Instead, we drop the image entirely and store only a text summary ("Image contained high-density sensitive content and was suppressed."). This is the safest option.

The 0.30 threshold was set conservatively — we prefer to suppress rather than risk leakage.

---

### 4.5 Robustness Considerations

The system was designed to be robust in several ways:

- **Multiple detector redundancy:** If OwlViT misses a face, Haar cascade catches it. If NER misses a card number, the regex catches it. No single point of failure.
- **Low confidence threshold for OwlViT (0.1):** We deliberately accept weak detections. A false positive (masking something harmless) is far cheaper than a false negative (missing something sensitive). The cost of masking a non-sensitive region is minimal; the cost of missing a passport number is high.
- **Ephemeral OCR principle:** Text strings are deleted immediately after checking. Even if the privacy policy has a bug and fails to add a bounding box, the raw OCR text was never persisted.
- **Two-gate design:** Sensing gate catches what it can before storage. Retrieval gate provides a second check before anything reaches the agent. Defense in depth.
- **All models are local:** No cloud API for any privacy decision. Even if an API provider's servers were compromised, no data would be exposed. This is the zero-trust design principle.

## 6. Results Explanation

### What the numbers mean

**Recovered attributes** are the 9 specific sensitive fields we tried to track: Name, ID Number, Address, Bank Card Number, Flight Number, License Plate Number, Diagnosis, Salary, and Relationship. For each field, we compared the agent's answer against the known ground truth.

- **True Positive (TP):** The agent recovered a field AND the value was correct (exact match for IDs/card numbers; fuzzy match for addresses and diagnoses).
- **False Positive (FP):** The agent gave a non-null answer but it was wrong — either a hallucination or a misread value.
- **False Negative (FN):** The field exists in the ground truth but the agent returned null or got it wrong — the information was not recovered.

From these we compute:
- **Precision** = TP / (TP + FP): Of everything the agent claimed to recover, how much was actually correct? High precision means the agent doesn't hallucinate.
- **Recall** = TP / (TP + FN): Of all the sensitive fields that existed, how many did the agent manage to extract? **This is our main privacy metric — lower recall means less leakage.**
- **F1** = harmonic mean of precision and recall: A single score combining both.
- **Identity accuracy**: Did the agent pick the correct candidate? Binary correct/incorrect per case.
- **Attribute recovery rate** = TP / total ground-truth fields: What fraction of all existing sensitive facts were successfully extracted?

---

### What the results say

**Identity Accuracy — all modes = 1.0 (100%)**

Every mode correctly identified which candidate each evidence set belonged to. This is an important nuance. It does NOT mean our filter failed. It means the agent is smart enough to reason from high-level behavioral patterns — the type of medical exam, the general direction of travel, the lifestyle description — and match those to our narrative profiles even when all specific identifiers are blacked out. For case_001 (Amy), the agent's explanation under our_filter said: *"Image 1 shows a normal abdominal ultrasound, matching Candidate A's medical profile. Image 2's flight from Beijing aligns with Candidate A's travel itinerary. Image 3 indicates US nationality."* — it used document type and travel direction, not names or IDs.

This actually demonstrates a deeper privacy finding: even after removing all specific identifiers, the *pattern* of information (this person had a cholesterol check AND flew Frankfurt-Shanghai AND shops for home goods) can still be uniquely identifying if the profiles are distinctive enough. This is what the privacy literature calls **contextual privacy leakage** — a limitation that no single-document-level filter can fully address, and a compelling direction for future work.

**Attribute Recovery Rate: 0.63 → 0.59 → 0.15**

This is the headline result. In raw mode, GPT-4o extracted 63% of all sensitive fields across all three cases. Under presidio_only, it recovered 59% — barely different. This is the key demonstration of why text-only filtering fails: the agent doesn't need us to provide text — it simply reads the text directly from the unmasked images using its own built-in OCR capability. You can redact our text inputs all you want; if the image is unmasked, the information is still there.

Under our_filter, recovery dropped to 15%. Looking at the raw JSON, in case_001 our_filter the agent recovered exactly one field: "Diagnosis: Normal abdominal ultrasound." Every other field — Name, ID Number, Address, Bank Card Number, Flight Number — came back null. The name "Amy" was blacked out. The 16-digit card number was blacked out. The ID number "2015188005046234" was blacked out. The face was covered. This is exactly the behavior we designed for.

**Recall: 0.654 → 0.615 → 0.16**

Recall dropped 75% from raw to our_filter. This is the most direct measure of privacy protection — how much of the sensitive ground truth can the agent actually reconstruct from what it sees? Our filter reduced that from 65% to 16%.

**Precision: 0.944 → 0.941 → 0.667**

Precision stayed high under raw and presidio_only — the agent was accurate about what it claimed to recover. Under our_filter, precision dropped to 0.667. This makes sense: when most real attributes are masked, the agent occasionally tries to infer a value from context (e.g., "Berlin" for Address) and gets a partially-correct or incorrect answer. The FP count was small (0–1 per case), reflecting low hallucination.

**Confidence — all modes = 0.917**

The agent expressed the same confidence level regardless of which preprocessing mode was used. This is an interesting finding that cuts against a naive assumption: people might expect that showing the agent less information would make it less confident. But GPT-4o's calibration seems to reflect its confidence in the identity match (which was always correct) rather than in its attribute recovery. For a presentation, you can frame this as: our filter made the agent significantly less informed — it extracted 75% fewer sensitive facts — while the agent didn't even know it was being filtered, because it was still playing the same detective game with the same quality of reasoning.

**F1: 0.773 → 0.744 → 0.258**

The overall F1 dropped from 0.77 in raw to 0.26 in our_filter — a 66% reduction in the combined precision-recall score. This is the summary stat that best captures the full picture of how much private information extraction capability was neutralized by our pipeline.

**Per-case breakdown — the most interesting detail**

Looking at case_001 (Amy): raw recovered Name + ID + Address + Card + Flight + Diagnosis (5 TP). Under our_filter: only Diagnosis (1 TP, 8 FN). The agent literally could not read Amy's name from the blacked-out ID card. It couldn't see the flight number CA1234 from the blacked-out ticket. It couldn't see the bank card number "1234 5678 1234 5678." The one thing it could still infer was the type of medical examination — because even a masked medical document still looks like a medical document, and the general category ("abdominal ultrasound") was still partially visible in the layout.

This points to a concrete limitation: **document-type leakage**. Even when specific content is masked, the structure and layout of a document (a medical form looks different from a boarding pass) provides information. Addressing this fully would require full image suppression (ABSTRACT policy) for all sensitive document types, at the cost of losing more utility.


- **Stage-awareness:** The same content may be allowed at the reasoning stage (agent sees it briefly) but blocked from durable storage. Privacy decisions are specific to the pipeline stage, not universal.
