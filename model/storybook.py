import os
import re
import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Tuple
from openai import OpenAI


def _load_openai_key_from_dotenv() -> None:
    """Load OPENAI_API_KEY from a local .env file if not already set."""
    if os.getenv("OPENAI_API_KEY"):
        return

    # Look for .env in project root and current working directory.
    candidate_paths = [
        Path(__file__).resolve().parents[1] / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in candidate_paths:
        if not env_path.exists():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            if key.strip() == "OPENAI_API_KEY":
                os.environ["OPENAI_API_KEY"] = value.strip().strip('"').strip("'")
                return


PROMPT = """You are a children's book content analyzer. Your task is to read the book text and assign the output one of nine lesson codes based on it's primary moral, educational, or thematic takeaway.
Read the entire Book Text carefully and determine which single lesson code best represents the main message or value that a child reader would take away from the story. Even if multiple themes are present, choose the one that is most central to the narrative.
THE NINE LESSON CODES:
FAM - Love of Family: The Book Text centers relationships between nuclear family members, depicting this love as special and unconditional. The reader learns to value their parents or other family members. Example: A story where parents take care of their child.
FRI - Friendship The Book Text features strong friendships between characters. Friendships may be tested but friends emerge stronger. The focus is on loyalty, reconciliation, or mutual care between friends. The reader learns the importance of having and maintaining friendships with care and reciprocity. Example: Two friends argue but work together to reconcile. Note: If the book primarily emphasizes loyalty, reconciliation, or mutual care between friends, especially if the moral centers on maintaining or valuing friendship, use FRI not JUS.
REL - Religion: The Book Text features overtly religious sentiments, actions, or quotations, often linking religious belief with prosociality. The reader learns the value of religious faith. Example: A protagonist prays for their friends and family.
EMO - Emotions: The Book Text guides readers toward recognizing, expressing, and managing emotions and their effects. Characters experience emotional changes throughout the story. May include characters coping with self-doubt who come to accept themselves. The reader learns that emotions are temporary and manageable. Example: A protagonist is angry about not getting their way but later accepts it and becomes content with what they have.
ADV - Adventure/Trying New Things: The Book Text features characters moving through different locations or settings as part of their journey. The narrative highlights exploration, venturing outside one's comfort zone, and embracing new experiences. Characters are often rewarded for bravery. The reader learns that exploring new things can lead to positive experiences. Example: A protagonist moves to a new school and learns to embrace the environment change. Note: The presence of an unfamiliar situation is necessary. A significant challenge is not required (that would be PER).
PER - Perseverance: The Book Text features characters facing situational challenges such as obstacles, setbacks, or difficult conditions. The reader learns the value of effort and hard work, especially through difficulty. Example: A protagonist improves their grades in school through hard work and studying. Note: The character must apply effort or face difficulty. An unfamiliar situation is not required (that would be ADV).
JUS - Justice and Fairness: The Book Text emphasizes fairness or moral consequences, whether grand scale (heroes vs. villains, good triumphing over evil) or everyday contexts (sharing, kindness, fair treatment). Right actions are rewarded and wrong actions bring penalties. Example: A protagonist shares with a stranger who later is able to give them something in return.
NAT - The Natural World: The Book Text familiarizes readers with and conveys value in animal or plant life. The reader learns that non-human animals and plant life are bearers of value. Example: A protagonist takes great care to preserve a garden. Note: Many books feature anthropomorphized animal protagonists. The mere presence of animal characters is not sufficient. The book must convey value in animals or plants beyond their exhibition of human traits.
NUL - Books Without a Lesson: The Book Text is designed primarily for fun, entertainment, or comfort, such as silly stories or bedtime routines, without an explicit moral, educational takeaway, or skill-building element. Readers read these books for reading's sake. Example: A protagonist plays hide and seek with the reader.
INSTRUCTIONS:
Read the complete book text provided
Identify the primary lesson or takeaway a child reader would receive
Select the single code that best matches this primary lesson
Output only the three-letter code
OUTPUT FORMAT: Return the three-letter code then a 1 sentence rationale as to why your prediction is the primary theme for that book. Example: "NUL - There is no primary lesson for this book that aligns with the theme categories."""


def _parse_code(raw_output: str) -> str:
    text = (raw_output or "").strip().upper()
    match = re.search(r"\b([A-Z]{3})\b", text)
    return match.group(1) if match else "UNK"


def _get_client() -> OpenAI:
    _load_openai_key_from_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your shell or create a .env file at project root "
            "(/Users/kass/Desktop/MML/Storybook Project/.env) with: OPENAI_API_KEY=your-key"
        )
    return OpenAI()


def predict_book_text(client: OpenAI, model_id: str, book_text: str) -> Tuple[str, str]:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": PROMPT},
            {"role": "user", "content": book_text},
        ],
        temperature=0.0,
    )
    raw = (response.choices[0].message.content or "").strip()
    return _parse_code(raw), raw


def iter_image_folders(root_dir: Path) -> Iterable[Path]:
    for path in sorted(root_dir.iterdir()):
        if path.is_dir():
            yield path


def run_round_image_batch(
    model_id: str,
    images_root: Path,
    output_csv: Path,
    max_books: int,
) -> None:
    # Import OCR helper lazily so this script can still run for text-only mode.
    storybook_root = Path(__file__).resolve().parents[1]
    if str(storybook_root) not in sys.path:
        sys.path.insert(0, str(storybook_root))
    from agents.ocr.claude_ocr import extract_text_from_folder

    client = _get_client()
    folders = list(iter_image_folders(images_root))
    if max_books > 0:
        folders = folders[:max_books]

    rows: List[dict] = []
    total = len(folders)
    print(f"Processing {total} books from {images_root}...")

    for i, folder in enumerate(folders, 1):
        book_id = folder.name
        print(f"[{i}/{total}] OCR + predict: {book_id}")
        try:
            book_text = extract_text_from_folder(str(folder))
            pred_code, raw_output = predict_book_text(client, model_id, book_text)
            rows.append(
                {
                    "book_id": book_id,
                    "source_folder": str(folder),
                    "pred_code": pred_code,
                    "raw_model_output": raw_output,
                    "text_char_count": len(book_text),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "book_id": book_id,
                    "source_folder": str(folder),
                    "pred_code": "ERR",
                    "raw_model_output": f"ERROR: {exc}",
                    "text_char_count": 0,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "book_id",
                "source_folder",
                "pred_code",
                "raw_model_output",
                "text_char_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions: {output_csv}")


def run_round_pdf_batch(
    model_id: str,
    pdf_root: Path,
    output_csv: Path,
    max_books: int,
) -> None:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for --mode pdf-dir. Install with pip install pymupdf") from exc

    client = _get_client()
    pdf_files = sorted([p for p in pdf_root.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if max_books > 0:
        pdf_files = pdf_files[:max_books]

    rows: List[dict] = []
    total = len(pdf_files)
    print(f"Processing {total} PDFs from {pdf_root}...")

    for i, pdf_path in enumerate(pdf_files, 1):
        book_id = pdf_path.stem
        print(f"[{i}/{total}] Extract + predict: {book_id}")
        try:
            doc = fitz.open(pdf_path)
            try:
                page_texts = [page.get_text("text") for page in doc]
            finally:
                doc.close()
            book_text = "\n\n".join(page_texts).strip()

            pred_code, raw_output = predict_book_text(client, model_id, book_text)
            rows.append(
                {
                    "book_id": book_id,
                    "source_pdf": str(pdf_path),
                    "pred_code": pred_code,
                    "raw_model_output": raw_output,
                    "text_char_count": len(book_text),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "book_id": book_id,
                    "source_pdf": str(pdf_path),
                    "pred_code": "ERR",
                    "raw_model_output": f"ERROR: {exc}",
                    "text_char_count": 0,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "book_id",
                "source_pdf",
                "pred_code",
                "raw_model_output",
                "text_char_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions: {output_csv}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run finetuned storybook theme inference")
    parser.add_argument(
        "--mode",
        choices=["image-dir", "pdf-dir"],
        default="image-dir",
        help="Use image OCR folders or direct PDF text extraction before finetuned inference.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="ft:gpt-4o-mini-2024-07-18:personal:storybook1:D3j5j130",
        help="Finetuned OpenAI model ID.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=project_root / "storybook" / "data" / "input_images" / "round_two_books",
        help="Directory containing one folder per book with page images.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root
        / "storybook"
        / "data"
        / "output_texts"
        / f"round_two_finetuned_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Path to write batch predictions CSV.",
    )
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=project_root / "p2_books",
        help="Directory containing PDF files when --mode pdf-dir is selected.",
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=0,
        help="Optional limit for number of books to process. 0 means all books.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "pdf-dir":
        run_round_pdf_batch(
            model_id=args.model_id,
            pdf_root=args.pdf_root,
            output_csv=args.output_csv,
            max_books=args.max_books,
        )
    else:
        run_round_image_batch(
            model_id=args.model_id,
            images_root=args.images_root,
            output_csv=args.output_csv,
            max_books=args.max_books,
        )
