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

Logs and troubleshooting
- If the model returns a non-JSON response or includes extra commentary, the full raw response for that page is written to `logs/json_errors/json_error_<image>_<timestamp>.txt` so you can inspect what happened.
- The script also normalizes and escapes control characters so joining pages won't raise errors.

Next steps:
- Save the combined output to a text file
- Add PDF → images conversion (PyMuPDF) to feed full PDFs into the same pipeline.
