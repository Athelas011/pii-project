# Project Improvement Requirements for Claude Code

## Goal

Please review the current project proposal and the existing project implementation, then help finalize the project to a stage that is good enough for tomorrow’s presentation.

The goal is not to rebuild the whole project from scratch. We already have the model and basic pipeline. The task is to improve, organize, and polish the project so that it has a clearer story, stronger comparison, better demo structure, and cleaner file management.

## Important Note: You should think independently as a senior project developer. Reasoning for each expectation I give you. If some requirements seems unnecessary, or will not likely to improve the performance, or improve by a large price e.g. waste much more time/token for detection and masked, send the data to outside platform e.g. llm which disobey the central goal of the project. you should tell me, and do not do them/suggest a subsitution

## After you read the all the expectations, Please read all the files in the folder papers/, see if there's useful thought, design of workflow, benchmark, metrics, etc. that is suitable for our project. notice that we only care about workable contribution from them. For example, we cannot train a new modal at this current moment.
---

## 1. Strengthen the Innovation Story

Right now, the project may look like it lacks enough innovation.

The current model mainly uses an existing OpenAI PII detector together with image masking. This is useful, but by itself it may not appear significantly stronger than simply using those tools separately.
help improve the project framing that it can better reach its expecatation of being a good PII detector. Some idea of 'good': Applicable privacy laws, regulations, and policies
 Restrictions on data collection, storage, and use of PII
 Roles and responsibilities for using and protecting PII
 Appropriate disposal of PII
 Sanctions for misuse of PII
 Recognition of a security or privacy incident involving PII (this one we already doing)
 Retention schedules for PII
 Roles and responsibilities in responding to PII-related incidents and reporting.

It is IMPOSSIBLE and you are not expected to finish all of them. If we can expand its current function to more than 1 or be really good at a single 1, that will be great. 

The improvement should focus on Deisgn-Level, show the value of the architecture and the workflow, especially how the project connects text + image privacy protection, and agent response safety into one system.

---

## 2. Add Clear Comparison With Baselines or Similar Methods

We need a comparison section so that the final version of the work can be shown to beat reasonable baselines or similar approaches.

Please help design and implement a simple but clear comparison structure.

The comparison should show that the final version performs better than simpler alternatives, such as:

* using only a text PII detector;
* using image masking alone;
* using the current components separately rather than as a full multimodal pipeline.

The comparison does not need to be huge, but it should be clean and presentation-friendly.

---

## 3. More about Select Suitable Metrics


Please choose suitable metrics that can support the comparison.

The metrics should help show two things:

First, the final model reduces privacy leakage better than the baseline.

Second, the multimodal architecture is more robust because text-only PII detection can miss sensitive information that appears in images, OCR text, or visual content.

The metrics should be easy to explain during the presentation. They should help support the claim that the full pipeline is safer and more robust than text-only or single-modality approaches.

Remember, We can do quantitative study, qualitative study, that includes robustness, precision, completness, time/token-taken etc. Choose correct metrics and explain why. I do not have the testing set yet, but I could provide you images with pii data, texts with pii data, or a dataset with random combination of both. Tell me what you expect, leave place for that.

---

## 4. Show Multimodal Robustness

One important argument for the project is that a text-only PII detector is not enough.

Please make sure the comparison or demo can show cases where sensitive information appears in images, screenshots, PDFs, or OCR-extracted text. The final multimodal pipeline should handle these cases better than a text-only baseline.

This part is important for the presentation because it directly supports why the project needs to be multimodal.

---

## 5. Improve the Demo

We need a clean and good-looking demo for the presentation. You DO NOT need to make slides -- we have a slide. But you want to create a demo in a way that it include everything (e.g. raw image, sensitive-info boxed, readacted) that should be on the presentation slides. Jupyter notebook could be a good choice, but remember, We need the image size to be enough to be clear. Better both the raw and readacted are clear.  

The demo should not only show that the redacted or masked information no longer contains privacy information. It should also show that the actual agent response/output is:

* natural;
* context-sensitive;
* privacy-safe;
* still useful after private information has been removed or masked.

The demo should make it easy to compare the input, the protected/redacted version, and the final agent response.

---

## 6. Separate Testing Inputs and Sensitive Queries Into Config Files

The testing image/text input paths and the self-defined _sensitive_queries should be moved into separate files.

This is to make the project easier to manage and easier to modify before the presentation.

Please organize these items outside the main code instead of hard-coding them directly in the pipeline or demo script.

The separate files should include:

* testing image paths;
* testing text inputs;
* testing PDF paths if the current project uses PDFs;
* self-defined sensitive queries.

The goal is simple project management: we should be able to add, remove, or edit testing cases and sensitive queries without changing the main code.

---

## 7. Presentation-Ready Final State

After these improvements, the project should be ready for a good presentation.

The final version should make the following story clear:

The project is not only applying a PII detector or masking images. Instead, it builds a multimodal privacy-protection workflow for agent systems. The value of the project is that it protects privacy across different input forms and still allows the agent to produce natural, useful, context-aware responses.

Remember to have a look of the papers! And be selective and reasoning about the useful information & expectations. Do not waste time on things that will not actually improve the performance of the current work to its ideal usage.
