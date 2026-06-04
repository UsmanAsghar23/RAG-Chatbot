import re
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

from pypdf import PdfReader

SourceType = Literal["pdf", "markdown"]

SUPPORTED_EXTENSIONS: dict[str, SourceType] = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
}

_FRONT_MATTER_RE = re.compile(r"^---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


class DocumentParseError(Exception):
    """Raised when a document cannot be parsed."""


class EmptyDocumentError(DocumentParseError):
    """Raised when a document contains no extractable text."""


@dataclass(frozen=True)
class ParsedSegment:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class ParsedDocument:
    segments: list[ParsedSegment]
    source_type: SourceType
    filename: str


def get_source_type(filename: str) -> SourceType:
    extension = _extension(filename)
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(f"Unsupported file type: {extension or '(none)'}")
    return SUPPORTED_EXTENSIONS[extension]


def parse_document(content: bytes, filename: str) -> ParsedDocument:
    source_type = get_source_type(filename)
    if source_type == "pdf":
        segments = parse_pdf(content)
    else:
        segments = parse_markdown(content)

    if not segments or not any(segment.text.strip() for segment in segments):
        raise EmptyDocumentError(f"No extractable text in {filename}")

    return ParsedDocument(
        segments=segments,
        source_type=source_type,
        filename=filename,
    )


def parse_pdf(content: bytes) -> list[ParsedSegment]:
    reader = PdfReader(BytesIO(content))
    segments: list[ParsedSegment] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            segments.append(ParsedSegment(text=text, page=page_number))

    return segments


def parse_markdown(content: bytes) -> list[ParsedSegment]:
    text = content.decode("utf-8")
    text = _strip_front_matter(text).strip()
    if not text:
        return []
    return [ParsedSegment(text=text, page=None)]


def _strip_front_matter(text: str) -> str:
    return _FRONT_MATTER_RE.sub("", text, count=1)


def _extension(filename: str) -> str:
    dot = filename.rfind(".")
    if dot == -1:
        return ""
    return filename[dot:].lower()
