# Storybook PDF-2-JSON

This project extracts text from images (page-by-page) and combines the results into a single book text.

Quick summary of the OCR pipeline (plain language):
- Point the script at a folder of page images (e.g. `data/input_images/sample_images`).
- Each image is sent to Claude for text extraction. The code collects the returned text for each page and joins them in order to produce one combined "book" string.
- The extractor is robust: when the model returns malformed JSON or a descriptive message instead of pure text, the raw response is saved to `logs/json_errors/` for inspection and the script sanitizes/escapes the returned value so the pipeline doesn't crash.
- The final printed output is cleaned so you get readable text (image descriptions are either removed or easily identifiable in the logs).

How to run
1. Provide your Anthropic/OpenAI API key in a local config file so it is not committed to
	the repository.

2. Activate the virtualenv and run the OCR script on a folder of page images:

```bash
source .venv/bin/activate
python agents/ocr/claude_ocr.py
```

# Storybook Moral Theme Extraction Pipeline

This project is an end-to-end research pipeline for extracting, structuring, and evaluating moral themes from children’s books using multimodal LLMs.

It combines OCR, text normalization, ontology-based labeling, and evaluation to study how reliably large language models can generate structured moral annotations from narrative content.

---

## Overview

The pipeline converts raw storybook inputs (PDFs or page images) into structured, labeled datasets suitable for analysis and experimentation.

Core stages:

1. **Ingestion**  
   Input: PDF or folder of page images

2. **OCR (Claude Vision)**  
   Each page is processed independently to extract text

3. **Text Aggregation & Cleaning**  
   Page-level outputs are concatenated into a single book-level document

4. **Moral Theme Labeling (LLM Prompting)**  
   The full text is passed to an LLM to generate structured moral theme labels

5. **Post-processing & Validation**  
   Outputs are normalized into a fixed schema and validated

6. **Evaluation & Analysis**  
   Predictions are compared against human-coded ground truth

---

## Repository Structure

```
storybook-project/
│
├── agents/
│   └── ocr/
│       └── claude_ocr.py        # OCR pipeline using Claude Vision
│
├── data/
│   ├── input_images/            # Raw page images
│   ├── processed_text/          # Extracted book text
│   └── labeled_data/            # Model + human annotations
│
├── logs/
│   └── json_errors/             # Raw malformed LLM responses
│
├── config/
│   └── anthropic_config         # Local API key config (gitignored)
│
├── storybook.py                 # Orchestration script (pipeline entry point)
└── README.md
```

---

## Data Schema

Each book is represented as a structured record:

```
{
  "book_id": str,
  "title": str,
  "raw_text": str,
  "clean_text": str,
  "model_output": {
    "themes": [str],
    "confidence": [float] | null
  },
  "human_labels": [str],
  "metadata": {
    "source": str,
    "num_pages": int
  }
}
```

### Theme Ontology

Themes are drawn from a fixed code set (3-letter abbreviations). Fia read a total of 53 children’s books. These books were sourced directly from the parents of children who completed the first phase of our Moral Circles study. Parents were asked to list three books that they read to their kids between the ages of two and five (see Image 1 below). This question was asked as part of a larger demographic survey for which parents were compensated $14. Two human coders, Karsten and Virginia, read 100 books 3 times, discussing places of disagreement in the scheme and adapting it after every round. In the final round, Karsten and Virginia labeled each book with at least one of the 8 labels. 

```
FRI = Friendship
FAM = Family
JUS = Justice/Fairness
REL = Religion
NAT = The Natural World
ADV = Adventure
PER = Perseverance
NUL = No Theme
...
```

The ontology is intentionally constrained to:
- Enable consistent labeling
- Support precision/recall evaluation
- Reduce prompt ambiguity

---

## OCR Pipeline

The OCR system processes books page-by-page using Claude Vision.

### Key properties

- **Robust to malformed outputs**  
  Non-JSON or noisy responses are logged to `logs/json_errors/`

- **Sanitization layer**  
  Escapes control characters and recovers usable text

- **Deterministic ordering**  
  Pages are processed and concatenated sequentially

### Run OCR

```bash
source .venv/bin/activate
python agents/ocr/claude_ocr.py
```

---

## Labeling Pipeline

After text extraction, the full book is passed to our fine-tuned model for moral theme classification.

### Prompting strategy

- Input: full cleaned text
- Output: structured list of theme codes
- Constraint: must select from ontology

### Failure modes handled

- Free-form explanations instead of structured output
- Hallucinated themes outside ontology
- Partial or empty responses

All raw outputs are preserved for auditability.

---

## Evaluation

Model outputs are compared against human-coded labels.

### Metrics

- **Accuracy**  
  Exact match between predicted and ground truth labels

- **Precision (per theme)**  
  TP / (TP + FP)

- **Recall (per theme)**  
  TP / (TP + FN)

- **F1 Score (per theme)**  
  Harmonic mean of precision and recall

- **Macro / Micro averages**

### Ground truth

- `book_code` or human annotator labels
- Multiple annotators supported for agreement analysis

---

## Experimental Design

### Splits

- Train / validation / test splits at the **book level**
- No page-level leakage across splits

### Variants

- Prompt variations
- Model variants
- OCR quality conditions

### Key questions

- How stable are theme predictions across prompt formulations?
- How does OCR noise affect downstream labeling accuracy?
- How calibrated are model confidence estimates?
- How well do LLMs generalize to unseen books?

---

### Logging

All anomalous outputs are stored:

```
logs/json_errors/json_error_<image>_<timestamp>.txt
```

This enables:
- Debugging failure modes
- Auditing model behavior
- Building better post-processing rules

---

## Setup

Create a local config file for API access:

```
config/anthropic_config

[Anthropic]
api_key = <your-anthropic-api-key>
```

This file is gitignored.

---

## Current Limitations

- OCR errors propagate into labeling
- Theme ontology may be incomplete or overlapping
- Single-pass labeling may miss secondary themes
- Limited dataset size for strong statistical claims

---

## Next Steps

- Add PDF → image conversion (PyMuPDF)
- Save structured outputs to versioned datasets
- Expand ontology and hierarchical theme structure
- Implement confidence calibration and abstention

- The script also normalizes and escapes control characters so joining pages won't raise errors.
