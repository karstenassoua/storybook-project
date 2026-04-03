import csv
import importlib.util
import sys
import types
from pathlib import Path

import pytest

# Import the script as a module, since `storybook` is not an installable package in this repo.
storybook_path = Path(__file__).resolve().parent / "storybook.py"
# Ensure the repository root and storybook package folder are in sys.path for imports.
repo_root = Path(__file__).resolve().parents[2]
storybook_root = Path(__file__).resolve().parents[1]
for path in (str(storybook_root), str(repo_root)):
    if path not in sys.path:
        sys.path.insert(0, path)

spec = importlib.util.spec_from_file_location("storybook_model_storybook", storybook_path)
storybook = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(storybook)

run_round_image_batch = storybook.run_round_image_batch
run_round_pdf_batch = storybook.run_round_pdf_batch


def test_run_round_image_batch_dry_run(monkeypatch, tmp_path):
    book_folder = tmp_path / "round_two_books" / "sample_book"
    book_folder.mkdir(parents=True)
    output_csv = tmp_path / "dry_run_image.csv"

    # Monkeypatch OCR function so it does not call external APIs
    monkeypatch.setattr(
        "agents.ocr.claude_ocr.extract_text_from_folder",
        lambda folder: "This is a test text from OCR extraction.",
    )

    called_predict = False

    def fake_predict_book_text(*args, **kwargs):
        nonlocal called_predict
        called_predict = True
        return "UNK", "should not be called"

    monkeypatch.setattr(storybook, "predict_book_text", fake_predict_book_text)

    run_round_image_batch(
        model_id="fake-model",
        images_root=tmp_path / "round_two_books",
        output_csv=output_csv,
        max_books=1,
        dry_run=True,
    )

    assert not called_predict, "predict_book_text should not be called in dry-run mode"

    with output_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["pred_code"] == "DRY"
    assert rows[0]["raw_model_output"] == "dry-run"
    assert rows[0]["extraction_method"] in {"ocr", "pdf"}


def test_run_round_pdf_batch_dry_run(monkeypatch, tmp_path):
    pdf_folder = tmp_path / "pdfs"
    pdf_folder.mkdir(parents=True)
    pdf_path = pdf_folder / "document.pdf"
    pdf_path.write_text("dummy")
    output_csv = tmp_path / "dry_run_pdf.csv"

    # Fake fitz module to avoid external dependency and actual PDF parsing
    class FakePage:
        def get_text(self, mode):
            return "Dummy PDF text"

    class FakeDoc:
        def __iter__(self):
            return iter([FakePage()])

        def close(self):
            return None

    fake_fitz = types.SimpleNamespace(open=lambda p: FakeDoc())
    sys.modules["fitz"] = fake_fitz

    called_predict = False

    def fake_predict_book_text(*args, **kwargs):
        nonlocal called_predict
        called_predict = True
        return "UNK", "should not be called"

    monkeypatch.setattr(storybook, "predict_book_text", fake_predict_book_text)

    run_round_pdf_batch(
        model_id="fake-model",
        pdf_root=pdf_folder,
        output_csv=output_csv,
        max_books=1,
        dry_run=True,
    )

    assert not called_predict, "predict_book_text should not be called in dry-run mode"

    with output_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["pred_code"] == "DRY"
    assert rows[0]["raw_model_output"] == "dry-run"
    assert rows[0]["extraction_method"] == "pdf"
