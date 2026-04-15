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


SYSTEM_PROMPT = """You are a children's book content analyzer. Your task is to read the book text and assign one of nine lesson codes based on it's primary moral, educational, or thematic takeaway.
Read the entire book text carefully and determine which single lesson code best represents the main message or value that a child reader would take away from the story. Even if multiple themes are present, choose the one that is most central to the narrative.
THE NINE LESSON CODES:
FAM - Love of Family: The book text centers relationships between nuclear family members, depicting this love as special and unconditional. The reader learns to value their parents or other family members. Example: A story where parents take care of their child. Note: If a book depicts family love and warmth — even if family members also model kind or fair behavior — use FAM not JUS. The test is whether the story's emotional core is the family bond itself.
FRI - Friendship: The book text features strong friendships between characters. Friendships may be tested but friends emerge stronger. The focus is on loyalty, reconciliation, or mutual care between friends. The reader learns the importance of having and maintaining friendships with care and reciprocity. Example: Two friends argue but work together to reconcile. Note: If the book primarily emphasizes loyalty, reconciliation, or mutual care between friends, especially if the moral centers on maintaining or valuing friendship, use FRI not JUS. Note: Warm or playful interactions between characters do not qualify as FRI if friendship itself is not the central lesson — a fun story that happens to include friends should be NUL.
REL - Religion: The book text features overtly religious sentiments, actions, or quotations, often linking religious belief with prosociality. The reader learns the value of religious faith. Example: A protagonist prays for their friends and family.
EMO - Emotions: The book text guides readers toward recognizing, expressing, and managing emotions and their effects. Characters experience emotional changes throughout the story. May include characters coping with self-doubt who come to accept themselves. The reader learns that emotions are temporary and manageable. Example: A protagonist is angry about not getting their way but later accepts it and becomes content with what they have. Note: If the primary arc of the story is a character recognizing and regulating their own emotional state — even if their behavior has consequences along the way — use EMO not JUS. The key distinction is whether the reader's main takeaway concerns emotions and internal change, or fairness and moral consequences.
ADV - Adventure/Trying New Things: The book text features characters moving through different locations or settings as part of their journey. The narrative highlights exploration, venturing outside one's comfort zone, and embracing new experiences. Characters are often rewarded for bravery. The reader learns that exploring new things can lead to positive experiences. Example: A protagonist moves to a new school and learns to embrace the environment change. Note: The presence of an unfamiliar situation is necessary. A significant challenge is not required (that would be PER).
PER - Perseverance: The book text features characters facing situational challenges such as obstacles, setbacks, or difficult conditions. The reader learns the value of effort and hard work, especially through difficulty. Example: A protagonist improves their grades in school through hard work and studying. Note: The character must apply effort or face difficulty. An unfamiliar situation is not required (that would be ADV).
JUS - Justice and Fairness: The book text emphasizes fairness or moral consequences, whether grand scale (heroes vs. villains, good triumphing over evil) or everyday contexts (sharing, kindness, fair treatment). Right actions are rewarded and wrong actions bring penalties. Example: A protagonist shares with a stranger who later is able to give them something in return. Note: If the story centers on family love (use FAM) or a character's emotional journey (use EMO), do not use JUS merely because good or bad behavior appears in the story. JUS requires that the moral consequence or fairness principle is itself the central lesson.
NAT - The Natural World: The book text familiarizes readers with and conveys value in animal or plant life. The reader learns that non-human animals and plant life are bearers of value. Example: A protagonist takes great care to preserve a garden. Note: Many books feature anthropomorphized animal protagonists. The mere presence of animal characters is not sufficient. The book must convey value in animals or plants beyond their exhibition of human traits. Note: If an animal character simply has fun adventures or human-like emotions without the narrative conveying appreciation for animal or plant life itself, use the lesson code that best fits the story's actual theme — do not default to NUL simply because the NAT signal is subtle.
NUL - Books Without a Lesson: The book text is designed primarily for fun, entertainment, or comfort, such as silly stories or bedtime routines, without an explicit moral, educational takeaway, or skill-building element. Readers read these books for reading's sake. Example: A protagonist plays hide and seek with the reader. Note: Only use NUL when no lesson code fits. If a lesson is present — even a subtle one — assign the appropriate code.
INSTRUCTIONS:
Read the complete book text provided
Identify the primary lesson or takeaway a child reader would receive
Select the single code that best matches this primary lesson
Refer to the code definitions above to make your labeling
Output only the three-letter code
OUTPUT FORMAT: Return only the three-letter code with no additional text, explanation, or formatting."""

# Prefix that mirrors the user message format used in the fine-tuning training data.
USER_PREFIX = "Please analyze this children's book text and assign the appropriate lesson code:"

VALID_CODES = {"FAM", "FRI", "REL", "EMO", "ADV", "PER", "JUS", "NAT", "NUL"}


def _parse_code(raw_output: str) -> str:
    """Extract a valid 9-code label from model output.

    Iterates over all 3-letter uppercase tokens and returns the first one
    that matches a known label. Falls back to 'UNK' if none match.
    """
    text = (raw_output or "").strip().upper()
    for token in re.findall(r"\b[A-Z]{3}\b", text):
        if token in VALID_CODES:
            return token
    return "UNK"


def _get_client() -> OpenAI:
    _load_openai_key_from_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your shell or add a .env file at the project root "
            "containing: OPENAI_API_KEY=your-key"
        )
    return OpenAI()


def predict_book_text(client: OpenAI, model_id: str, book_text: str) -> Tuple[str, str]:
    """Classify a book text using the fine-tuned model.

    Message format mirrors the fine-tuning training data exactly:
      system  — codebook and classification instructions
      user    — USER_PREFIX + book text
    """
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{USER_PREFIX}\n\n{book_text}"},
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
    dry_run: bool = False,
) -> None:
    # Import OCR helper lazily so this script can still run for text-only mode.
    storybook_root = Path(__file__).resolve().parents[1]
    if str(storybook_root) not in sys.path:
        sys.path.insert(0, str(storybook_root))
    from agents.ocr.claude_ocr import extract_text_from_folder

    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        fitz = None

    client = _get_client()
    folders = list(iter_image_folders(images_root))
    if max_books > 0:
        folders = folders[:max_books]

    rows: List[dict] = []
    total = len(folders)
    print(f"Processing {total} books from {images_root}...")

    for i, folder in enumerate(folders, 1):
        book_id = folder.name
        print(f"[{i}/{total}] Processing: {book_id}")
        book_text = None
        extraction_method = "ocr"
        source_path = str(folder)

        # Try PDF extraction first if PyMuPDF is available
        if fitz:
            pdf_files = list(folder.glob("*.pdf"))
            if pdf_files:
                pdf_path = pdf_files[0]
                try:
                    doc = fitz.open(pdf_path)
                    page_texts = [page.get_text("text") for page in doc]
                    doc.close()
                    candidate_text = "\n\n".join(page_texts).strip()
                    if len(candidate_text) > 1000:  # Threshold for meaningful extractable text
                        book_text = candidate_text
                        extraction_method = "pdf"
                        source_path = str(pdf_path)
                except Exception:
                    pass  # Fall through to OCR

        # If not from PDF, do OCR
        if book_text is None:
            try:
                book_text = extract_text_from_folder(str(folder))
                extraction_method = "ocr"
                source_path = str(folder)
            except Exception as exc:
                rows.append(
                    {
                        "book_id": book_id,
                        "source_path": source_path,
                        "extraction_method": "error",
                        "pred_code": "ERR",
                        "raw_model_output": f"ERROR: {exc}",
                        "text_char_count": 0,
                    }
                )
                continue

        # Predict using the extracted text, unless this is a dry run.
        if dry_run:
            rows.append(
                {
                    "book_id": book_id,
                    "source_path": source_path,
                    "extraction_method": extraction_method,
                    "pred_code": "DRY",
                    "raw_model_output": "dry-run",
                    "text_char_count": len(book_text),
                }
            )
        else:
            try:
                pred_code, raw_output = predict_book_text(client, model_id, book_text)
                rows.append(
                    {
                        "book_id": book_id,
                        "source_path": source_path,
                        "extraction_method": extraction_method,
                        "pred_code": pred_code,
                        "raw_model_output": raw_output,
                        "text_char_count": len(book_text),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "book_id": book_id,
                        "source_path": source_path,
                        "extraction_method": extraction_method,
                        "pred_code": "ERR",
                        "raw_model_output": f"ERROR: {exc}",
                        "text_char_count": len(book_text) if book_text else 0,
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "book_id",
                "source_path",
                "extraction_method",
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
    dry_run: bool = False,
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

            if dry_run:
                rows.append(
                    {
                        "book_id": book_id,
                        "source_path": str(pdf_path),
                        "extraction_method": "pdf",
                        "pred_code": "DRY",
                        "raw_model_output": "dry-run",
                        "text_char_count": len(book_text),
                    }
                )
            else:
                try:
                    pred_code, raw_output = predict_book_text(client, model_id, book_text)
                    rows.append(
                        {
                            "book_id": book_id,
                            "source_path": str(pdf_path),
                            "extraction_method": "pdf",
                            "pred_code": pred_code,
                            "raw_model_output": raw_output,
                            "text_char_count": len(book_text),
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "book_id": book_id,
                            "source_path": str(pdf_path),
                            "extraction_method": "error",
                            "pred_code": "ERR",
                            "raw_model_output": f"ERROR: {exc}",
                            "text_char_count": 0,
                        }
                    )
        except Exception as exc:
            rows.append(
                {
                    "book_id": book_id,
                    "source_path": str(pdf_path),
                    "extraction_method": "error",
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
                "source_path",
                "extraction_method",
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
        default="ft:gpt-4o-mini-2024-07-18:personal:storybook2:DUuqYrWI",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Extract text only and log metadata; skip OpenAI predictions.",
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
            dry_run=args.dry_run,
        )
    else:
        run_round_image_batch(
            model_id=args.model_id,
            images_root=args.images_root,
            output_csv=args.output_csv,
            max_books=args.max_books,
            dry_run=args.dry_run,
        )
