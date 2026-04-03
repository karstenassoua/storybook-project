import csv
from pathlib import Path
from typing import Dict, List, Tuple


def load_human_labels(vc_csv_path: Path) -> Dict[str, str]:
    """Load human labels from VCODES1.csv (two-row CSV)."""
    vc_csv_path = Path(vc_csv_path)
    with vc_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"VCODES file must have 2 rows: names and labels; got {len(rows)}")
    names = [name.strip() for name in rows[0]]
    codes = [code.strip() for code in rows[1]]
    if len(names) != len(codes):
        raise ValueError("VCODES1.csv label count mismatch: %d names vs %d codes" % (len(names), len(codes)))

    # Keep empty labels as '' so we can detect missing human code quickly.
    return dict(zip(names, codes))


def load_cleaned_texts(cleaned_csv_path: Path) -> List[Tuple[str, str]]:
    """Load cleaned text dataset from storybooks_cleaned.csv."""
    cleaned_csv_path = Path(cleaned_csv_path)
    with cleaned_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(row["book_name"].strip(), row["book_text"].strip()) for row in reader]
    return rows


def _normalize_name(name: str) -> str:
    cleaned = "".join(ch.lower() for ch in name if ch.isalnum())
    return cleaned


def load_dataset(
    vc_csv_path: Path,
    cleaned_csv_path: Path,
) -> List[Dict[str, str]]:
    """Load merged dataset with text and human label where available."""
    labels = load_human_labels(vc_csv_path)
    texts = load_cleaned_texts(cleaned_csv_path)

    # Build a secondary mapping from normalized VCODES names to human labels.
    vc_mapping = {}
    for raw_name, code in labels.items():
        core_name = raw_name.split(",")[0] if "," in raw_name else raw_name
        vc_mapping[_normalize_name(core_name)] = code

    dataset = []
    missing = []
    for book_name, book_text in texts:
        cleaned_name = _normalize_name(book_name)
        label = None

        # Try direct normalized match
        if cleaned_name in vc_mapping:
            label = vc_mapping[cleaned_name]

        # Fallback: substring matches of cleaned into vc name or vice versa
        if label is None:
            for vc_norm, vc_label in vc_mapping.items():
                if cleaned_name and (cleaned_name in vc_norm or vc_norm in cleaned_name):
                    label = vc_label
                    break

        if label and label != "":
            dataset.append({"book_name": book_name, "book_text": book_text, "label": label})
        else:
            missing.append(book_name)

    if missing:
        print(f"Warning: {len(missing)} cleaned books have no VCODE label; they are excluded from evaluation.")
    return dataset
