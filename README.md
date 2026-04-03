# Storybook PDF-2-JSON

This project extracts text from images (page-by-page) and combines the results into a single book text.

Quick summary of the OCR pipeline (plain language):
- Point the script at a folder of page images (e.g. `data/input_images/sample_images`).
- Each image is sent to Claude for text extraction. The code collects the returned text for each page and joins them in order to produce one combined "book" string.
- The extractor is robust: when the model returns malformed JSON or a descriptive message instead of pure text, the raw response is saved to `logs/json_errors/` for inspection and the script sanitizes/escapes the returned value so the pipeline doesn't crash.
- The final printed output is cleaned so you get readable text (image descriptions are either removed or easily identifiable in the logs).

How to run
1. Provide your Anthropic API key in a local config file so it is not committed to
	the repository. Create `config/anthropic_config` with the following shape (do
	not paste an actual key into README):

	[Anthropic]
	api_key = <your-anthropic-api-key-here>

	Note: this file is gitignored by default so your secret won't be committed.

2. Activate the virtualenv and run the OCR script on a folder of page images:

```bash
source .venv/bin/activate
python agents/ocr/claude_ocr.py
```

3. (Optional) Convert a PDF into page images using PyMuPDF. The output will be
	written to `data/input_images/round_two_books/<book_title>/` where
	`<book_title>` is lowercase with underscores. Each page uses a 3-digit suffix:
	`<book_title>-001.jpg`, `<book_title>-002.jpg`, etc.

```bash
source .venv/bin/activate
pip install pymupdf
python agents/ocr/claude_ocr.py --pdf /path/to/who_said_coo.pdf --book-title "Who Said Coo?"
```

Example output path and file names:
- `data/input_images/round_two_books/who_said_coo/`
- `who_said_coo-001.jpg`
- `who_said_coo-002.jpg`

Logs and troubleshooting
- If the model returns a non-JSON response or includes extra commentary, the full raw response for that page is written to `logs/json_errors/json_error_<image>_<timestamp>.txt` so you can inspect what happened.
- The script also normalizes and escapes control characters so joining pages won't raise errors.

Next steps:
- Save the combined output to a text file
- Run OCR on a converted folder:

```bash
python agents/ocr/claude_ocr.py --folder data/input_images/round_two_books/who_said_coo
```

## Dry-run mode for pipeline sanity checks

The model pipeline now supports a `--dry-run` flag in `storybook/model/storybook.py`.
This mode runs extraction (PDF text or OCR fallback) and writes CSV output without invoking OpenAI prediction.

```bash
source .venv/bin/activate
python storybook/model/storybook.py --mode image-dir --max-books 1 --dry-run
python storybook/model/storybook.py --mode pdf-dir --max-books 1 --dry-run
```

The output CSV columns include:
- `book_id`
- `source_path`
- `extraction_method` (`pdf`, `ocr`, `error`)
- `pred_code` (`DRY` in dry-run)
- `raw_model_output` (`dry-run` or error text)
- `text_char_count`

## Evaluation module

A new evaluation module is available in `storybook/evaluation/`:

- `dataset.py`: load human-labeled GCOD dataset from `storybook/data/VCODES1.csv` and cleaned text from `storybook/data/storybooks_cleaned.csv`.
- `metrics.py`: per-label precision/recall/F1, macro/micro aggregate, and exact-match accuracy.
- `runner.py`: run predictions in batch with `evaluate_finetuned_model(...)`, save CSV and metrics JSON, and freeze train/test split.

### Freeze test split (deterministic)

The frozen split is created by `create_test_split(...)` in `storybook/evaluation/runner.py`.
- `test_size=20`
- `seed=42`
- saved as `storybook/evaluation/test_split.csv` when you run:

```python
from pathlib import Path
from storybook.evaluation.runner import create_test_split, load_dataset

vc_csv = Path('storybook/data/VCODES1.csv')
cleaned_csv = Path('storybook/data/storybooks_cleaned.csv')
from storybook.evaluation.dataset import load_dataset

data = load_dataset(vc_csv, cleaned_csv)
test_split = create_test_split(data, test_size=20, seed=42, output_path=Path('storybook/evaluation/test_split.csv'))
```

### Run evaluation against fine-tuned model

```bash
python -c "from pathlib import Path; from storybook.evaluation.runner import evaluate_from_data_files; print(evaluate_from_data_files(Path('storybook/data/VCODES1.csv'), Path('storybook/data/storybooks_cleaned.csv'), model_id='ft:gpt-4o-mini-2024-07-18:personal:storybook1:D3j5j130', output_csv=Path('storybook/evaluation/predictions.csv'), test_size=20, seed=42, dry_run=True))"
```

For a real model run (not dry-run), set `dry_run=False`.

### Metric outputs

The runner writes:
- CSV: `storybook/evaluation/predictions.csv` with `book_name`, `true_label`, `pred_label`, `raw_model_output`, `text_length`
- JSON: `storybook/evaluation/predictions.metrics.json` with per-label and macro/micro metrics.

## Test suite

Install pytest in the virtualenv:

```bash
pip install pytest
```

Run tests:

```bash
pytest -q storybook/model/test_storybook.py
```
