import base64
import json
import os
import time
import re
import argparse
import configparser
from anthropic import Anthropic
from typing import Dict, Any, List
from pathlib import Path


def format_book_title(book_title: str) -> str:
    """Normalize a book title into lowercase underscore format.

    Example: "Who Said Coo?" -> "who_said_coo"
    """
    normalized = book_title.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "untitled_book"


def convert_pdf_to_images(
    pdf_path: str,
    input_images_root: str = "data/input_images",
    round_subfolder: str = "round_two_books",
    book_title: str = "",
    dpi: int = 200,
) -> str:
    """Convert a PDF to page images in data/input_images/round_two_books/<book_title>/.

    Files are named with 3-digit page indices:
    <book_title>-001.jpg, <book_title>-002.jpg, ...
    """
    try:
        import fitz  # type: ignore[import-not-found]  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is not installed. Install it with: pip install pymupdf"
        ) from exc

    source_pdf = Path(pdf_path)
    if not source_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {source_pdf}")

    raw_title = book_title.strip() if book_title else source_pdf.stem
    safe_title = format_book_title(raw_title)

    output_dir = Path(input_images_root) / round_subfolder / safe_title
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(source_pdf)
    try:
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)

        for index, page in enumerate(doc, 1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_name = f"{safe_title}-{index:03d}.jpg"
            image_path = output_dir / image_name
            pix.save(str(image_path))
    finally:
        doc.close()

    return str(output_dir)


def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    """
    Extract text from an image using Claude API.

    Returns a dict with key 'extracted_text' whose value is a string safe for
    joining/printing (control characters escaped). On JSON-parse failures the
    raw API response is saved to logs/json_errors for later inspection.
    """
    # Read API key from config file
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'anthropic_config')
    config.read(config_path)
    api_key = config['Anthropic']['api_key']

    anthropic = Anthropic(api_key=api_key)

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = anthropic.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_image
                        }
                    },
                    {
                        "type": "text",
                        "text": "Extract all text from this image and return it as a JSON object with 'extracted_text' as the key."
                    }
                ]
            }
        ]
    )

    raw_text = response.content[0].text

    def _escape_and_normalize(value: Any) -> str:
        """Convert arbitrary returned value to a single escaped string.

        - Lists are joined with newlines.
        - Dicts are JSON-dumped.
        - Then unicode/control characters are escaped (so strings are safe).
        """
        if isinstance(value, list):
            s = '\n'.join(map(str, value))
        elif isinstance(value, dict):
            s = json.dumps(value, ensure_ascii=False)
        else:
            s = str(value)

        try:
            escaped = s.encode('unicode_escape').decode('ascii')
        except Exception:
            escaped = ''.join(ch if ord(ch) >= 32 else '?' for ch in s)
        return escaped

    # Attempt to parse JSON and extract a meaningful textual value. If
    # parsing fails, log the raw response and fall back to escaping the raw
    # text. This avoids returning raw JSON blobs into the final combined text.
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            # Prefer the explicit extracted_text key if present
            if 'extracted_text' in parsed:
                return {"extracted_text": _escape_and_normalize(parsed['extracted_text'])}

            # Otherwise, collect textual pieces from values
            pieces: List[str] = []
            for v in parsed.values():
                if isinstance(v, (str, int, float)):
                    pieces.append(str(v))
                elif isinstance(v, list):
                    pieces.extend(map(str, v))
                elif isinstance(v, dict):
                    pieces.append(json.dumps(v, ensure_ascii=False))

            if pieces:
                return {"extracted_text": _escape_and_normalize('\n'.join(pieces))}

            # Fallback: stringify whole dict
            return {"extracted_text": _escape_and_normalize(parsed)}

        # If parsed is not a dict (list or string), normalize it
        return {"extracted_text": _escape_and_normalize(parsed)}
    except json.JSONDecodeError:
        # Save raw response for debugging
        try:
            logs_dir = Path(__file__).resolve().parents[2] / 'logs' / 'json_errors'
            logs_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem if image_path else 'unknown'
            timestamp = int(time.time())
            fname = logs_dir / f"json_error_{stem}_{timestamp}.txt"
            with open(fname, 'w', encoding='utf-8') as fh:
                fh.write(raw_text)
            print(f"Saved raw API response to {fname}")
        except Exception as write_e:
            print(f"Failed to write raw API response: {write_e}")

        # If there's an embedded JSON object in the response, try to extract it
        m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if m:
            try:
                inner = json.loads(m.group(0))
                if isinstance(inner, dict) and 'extracted_text' in inner:
                    return {"extracted_text": _escape_and_normalize(inner['extracted_text'])}
                pieces = []
                for v in inner.values():
                    if isinstance(v, (str, int, float)):
                        pieces.append(str(v))
                    elif isinstance(v, list):
                        pieces.extend(map(str, v))
                if pieces:
                    return {"extracted_text": _escape_and_normalize('\n'.join(pieces))}
            except Exception:
                pass

        return {"extracted_text": _escape_and_normalize(raw_text)}


def extract_text_from_folder(folder_path: str) -> str:
    """
    Extract text from all images in a folder in order and combine them into one string.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    if not image_files:
        print(f"No image files found in {folder_path}")
        return ""

    combined_text: List[str] = []

    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, filename)
        print(f"Processing {i}/{len(image_files)}: {filename}")
        try:
            result = extract_text_from_image(image_path)
            extracted_text = result.get('extracted_text', '')

            # Normalize types (lists -> joined string, others -> str)
            if isinstance(extracted_text, list):
                extracted_text = '\n'.join(map(str, extracted_text))
            elif not isinstance(extracted_text, str):
                extracted_text = str(extracted_text)

            combined_text.append(extracted_text)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    return '\n\n'.join(combined_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR and PDF-to-images utilities")
    parser.add_argument(
        "--pdf",
        type=str,
        default="",
        help="Path to a PDF to convert into page images.",
    )
    parser.add_argument(
        "--book-title",
        type=str,
        default="",
        help="Optional book title override used for folder and file names.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="data/input_images/sample_images",
        help="Folder of images to OCR.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF page rendering when using --pdf.",
    )
    args = parser.parse_args()

    if args.pdf:
        generated_folder = convert_pdf_to_images(
            pdf_path=args.pdf,
            book_title=args.book_title,
            dpi=args.dpi,
        )
        print(f"Saved page images to: {generated_folder}")
    else:
        book_text = extract_text_from_folder(args.folder)

        # Remove JSON-like snippets containing an extracted_text key so the final
        # output is clean and human-readable.
        cleaned = re.sub(r"\n?\s*\{[^}]*\"extracted_text\"[^}]*\}\n?", "\n", book_text, flags=re.DOTALL)

        # Decode unicode escapes to make output readable (we escaped control chars earlier)
        try:
            decoded = cleaned.encode('utf-8').decode('unicode_escape')
        except Exception:
            decoded = cleaned

        print("\n" + "="*50)
        print("COMBINED BOOK TEXT:")
        print("="*50)
        print(decoded)
