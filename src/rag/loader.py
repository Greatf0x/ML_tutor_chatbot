from pathlib import Path
import docx


def load_text_from_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)
    if suffix == ".docx":
        return load_docx(file_path)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    raise ValueError("Unsupported file type. Please use PDF, DOCX, or TXT.")


def load_pdf(file_path: str) -> str:
    import pdfplumber
    text_parts = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract with layout=True to preserve word spacing
            text = page.extract_text(layout=True)
            if not text:
                # Fallback for pages where layout mode returns nothing
                text = page.extract_text()
            if text:
                text_parts.append(text)

    return "\n\n".join(text_parts)


def load_docx(file_path: str) -> str:
    document = docx.Document(file_path)
    return "\n".join(
        para.text for para in document.paragraphs if para.text.strip()
    )