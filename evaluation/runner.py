"""
DEPRECATED — use qualitative_coding/evaluate_split.py for all evaluation.

This module provides a legacy 20-book frozen test split against VCODES1.csv
and storybooks_cleaned.csv. It is retained for reference only. The canonical
benchmark is evaluate_split.py, which runs a reproducible 60/40 stratified
split on the full 96-example JSONL dataset with bootstrap confidence intervals
and multi-model comparison.
"""
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from storybook.model.storybook import _get_client, predict_book_text
from .dataset import load_dataset
from .metrics import compute_metrics


def create_test_split(
    dataset: List[Dict[str, str]],
    test_size: int = 20,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Create and persist a frozen test split by book_name."""
    random.Random(seed).shuffle(dataset)
    test = dataset[:test_size]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["book_name", "label"])
            writer.writeheader()
            writer.writerows([{"book_name": d["book_name"], "label": d["label"]} for d in test])

    return test


def evaluate_finetuned_model(
    dataset: List[Dict[str, str]],
    model_id: str,
    output_csv: Path,
    max_books: int = 0,
    dry_run: bool = False,
    text_column: str = "book_text",
) -> Dict[str, object]:
    """Run predictions and compute metrics against human labels."""
    client = _get_client() if not dry_run else None
    books = dataset[:max_books] if max_books and max_books > 0 else dataset

    results = []
    for record in books:
        true_label = record["label"]
        book_text = record[text_column]
        book_name = record["book_name"]

        if dry_run:
            pred_label, raw_out = "DRY", "dry-run"
        else:
            pred_label, raw_out = predict_book_text(client, model_id, book_text)

        results.append(
            {
                "book_name": book_name,
                "true_label": true_label,
                "pred_label": pred_label,
                "raw_model_output": raw_out,
                "text_length": len(book_text),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["book_name", "true_label", "pred_label", "raw_model_output", "text_length"],
        )
        writer.writeheader()
        writer.writerows(results)

    # Compute metrics only when not dry run
    if dry_run:
        return {"message": "dry-run mode, no model metrics computed", "num_records": len(results)}

    true = [r["true_label"] for r in results]
    pred = [r["pred_label"] for r in results]
    stats = compute_metrics(true, pred)

    metric_path = output_csv.with_suffix(".metrics.json")
    with metric_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def evaluate_from_data_files(
    vc_csv_path: Path,
    cleaned_csv_path: Path,
    model_id: str,
    output_csv: Path,
    test_size: int = 20,
    seed: int = 42,
    dry_run: bool = False,
) -> Dict[str, object]:
    data = load_dataset(vc_csv_path, cleaned_csv_path)
    test_split = create_test_split(data, test_size=test_size, seed=seed)
    return evaluate_finetuned_model(
        test_split,
        model_id=model_id,
        output_csv=output_csv,
        dry_run=dry_run,
    )
