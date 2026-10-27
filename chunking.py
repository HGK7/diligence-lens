from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

# Active diligence corpus for Diligence Lens (CUAD subset exports land here).
DOCS_PATH = Path("dataroom")
# Legacy Nextwork seed folder; still accepted if present during migration.
LEGACY_DOCS_PATH = Path("vendor_docs")

# Steady chunking defaults: section-first, then paragraph windows.
# Tuned for ~50-100 contract docs without blowing 4 GB embedding VRAM.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNK_MIN_CHARS = 80


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _split_paragraphs(text: str) -> list[str]:
    blocks = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def _window_chunk(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Character windows that prefer breaking near sentence ends."""
    normalized = _normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= size:
        return [normalized]

    output: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(length, start + size)
        if end < length:
            window = normalized[start:end]
            # Prefer a sentence boundary in the last third of the window.
            search_from = max(0, len(window) - size // 3)
            best = -1
            for marker in (". ", "? ", "! ", ";\n", ".\n"):
                idx = window.rfind(marker, search_from)
                if idx > best:
                    best = idx + len(marker.strip())
            if best >= size // 2:
                end = start + best
        piece = normalized[start:end].strip()
        if len(piece) >= CHUNK_MIN_CHARS:
            output.append(piece)
        if end >= length:
            break
        start = max(0, end - overlap)
    return output


def _chunks(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Steady strategy:
    1. Pack whole paragraphs while under size.
    2. Split oversized paragraphs with sentence-aware windows.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return []

    packed: list[str] = []
    buffer = ""
    for paragraph in _split_paragraphs(normalized):
        if len(paragraph) > size:
            if buffer:
                packed.append(buffer)
                buffer = ""
            packed.extend(_window_chunk(paragraph, size=size, overlap=overlap))
            continue
        candidate = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
        if len(candidate) <= size:
            buffer = candidate
        else:
            if buffer:
                packed.append(buffer)
            buffer = paragraph
    if buffer:
        packed.append(buffer)
    return [item for item in packed if item]


CLAUSE_HEADING = re.compile(
    r"""
    ^\s*
    (?P<number>
        (?:SECTION\s+)?
        \d+(?:\.\d+)*
        |
        ARTICLE\s+[IVXLC]+
    )
    [\s.:\-–—]*
    (?P<title>.*?)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_clause_heading(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 160:
        return None

    match = CLAUSE_HEADING.match(stripped)
    if not match:
        return None

    number = match.group("number").strip()
    title = match.group("title").strip()

    # Bare numbers are often dates, page numbers, or list items.
    if not title and not number.upper().startswith(("SECTION", "ARTICLE")):
        return None

    return number, title


def contract_sections(text: str) -> list[dict]:
    sections: list[dict] = []
    current_number = ""
    current_title = "Document"
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        content = "\n".join(body).strip()
        if content:
            sections.append(
                {
                    "section_number": current_number,
                    "section_title": current_title,
                    "body": content,
                }
            )
        body = []

    for line in text.splitlines():
        heading = parse_clause_heading(line)
        if heading:
            flush()
            current_number, title = heading
            current_title = title or current_number
        else:
            body.append(line)

    flush()
    return sections


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    sections = []
    title = "Document"
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Markdown headings and ALL-CAPS contract section labels.
        is_heading = stripped.startswith("#") or (
            len(stripped) <= 80
            and stripped.isupper()
            and any(ch.isalpha() for ch in stripped)
            and not stripped.startswith("HTTP")
        )
        if is_heading and stripped.startswith("#"):
            if lines:
                sections.append((title, "\n".join(lines).strip()))
            title = stripped.lstrip("#").strip() or "Document"
            lines = []
        elif is_heading and stripped.isupper() and lines:
            sections.append((title, "\n".join(lines).strip()))
            title = stripped.title()
            lines = []
        else:
            lines.append(line)
    if lines:
        sections.append((title, "\n".join(lines).strip()))
    return [(name, body) for name, body in sections if body]


def _record_chunks(
    text: str,
    source: str,
    section: str | None = None,
    *,
    section_number: str = "",
    section_title: str = "",
    page: int | None = None,
) -> list[dict]:
    if section is None:
        section = f"{section_number} {section_title}".strip() or "Document"
    if not section_title and section:
        section_title = section

    records = []
    for chunk_index, chunk in enumerate(_chunks(text), start=1):
        raw_text = chunk
        embedding_text = (
            f"Contract: {source}\n"
            f"Clause: {section_number} {section_title}\n".strip()
            + f"\nContent: {raw_text}"
        )
        records.append(
            {
                "source": source,
                "section": section,
                "section_number": section_number,
                "section_title": section_title,
                "page": page,
                "chunk_index": chunk_index,
                "text": raw_text,
                "embedding_text": embedding_text,
            }
        )
    return records


def records_from_bytes(name: str, data: bytes) -> list[dict]:
    source = Path(name).name
    suffix = Path(source).suffix.lower()
    records = []
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(data))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            records.extend(
                _record_chunks(
                    text, source, f"Page {page_number}", page=page_number
                )
            )
        return records

    text = data.decode("utf-8", errors="replace")
    if suffix == ".md":
        for section, body in _markdown_sections(text):
            records.extend(_record_chunks(body, source, section))
    else:
        clause_sections = contract_sections(text)
        # Prefer clause splits when we found more than a single "Document" blob.
        if len(clause_sections) > 1 or any(
            item["section_number"] for item in clause_sections
        ):
            for item in clause_sections:
                records.extend(
                    _record_chunks(
                        item["body"],
                        source,
                        section_number=item["section_number"],
                        section_title=item["section_title"],
                    )
                )
        else:
            # Fallback: old ALL-CAPS / whole-doc behavior
            sections = _markdown_sections(text)
            if len(sections) > 1:
                for section, body in sections:
                    records.extend(_record_chunks(body, source, section=section))
            else:
                records.extend(_record_chunks(text, source, section="Document"))
    return records


def _iter_corpus_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        # Skip nested dataset archives / raw dumps; only index the active room.
        if "contract-nli" in path.parts or "raw" in path.parts:
            continue
        if path.suffix.lower() == ".json":
            continue
        if path.name.upper() in {"LICENSE", "LICENSE.TXT", "NOTICE", "README.MD"}:
            continue
        if path.suffix.lower() in {".md", ".txt", ".pdf"}:
            files.append(path)
    return files


def load_seed_records() -> list[dict]:
    """Load the active dataroom first; fall back to legacy vendor_docs if empty."""
    records = []
    primary = _iter_corpus_files(DOCS_PATH)
    if primary:
        root = DOCS_PATH
        sources = primary
    else:
        root = LEGACY_DOCS_PATH
        sources = _iter_corpus_files(root)
    for path in sources:
        try:
            label = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            label = path.name
        records.extend(records_from_bytes(label, path.read_bytes()))
    return records


def load_uploaded_records(uploaded_files) -> list[dict]:
    records = []
    for uploaded_file in uploaded_files or []:
        records.extend(records_from_bytes(uploaded_file.name, uploaded_file.getvalue()))
    return records


def corpus_file_count() -> int:
    primary = _iter_corpus_files(DOCS_PATH)
    if primary:
        return len(primary)
    return len(_iter_corpus_files(LEGACY_DOCS_PATH))
