import hashlib
from dataclasses import dataclass

import tiktoken

from app.services.parsers import ParsedSegment, SourceType

EMBEDDING_ENCODING = "cl100k_base"
SEPARATORS = ["\n\n", "\n", " "]


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    chunk_index: int
    doc_id: str
    source_type: SourceType
    page: int | None = None


def chunk_document(
    doc_id: str,
    segments: list[ParsedSegment],
    source_type: SourceType,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextChunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    chunks: list[TextChunk] = []
    chunk_index = 0

    for segment in segments:
        for text in chunk_text(
            segment.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding=encoding,
        ):
            chunks.append(
                TextChunk(
                    chunk_id=_make_chunk_id(doc_id, chunk_index),
                    text=text,
                    chunk_index=chunk_index,
                    doc_id=doc_id,
                    source_type=source_type,
                    page=segment.page,
                )
            )
            chunk_index += 1

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoding: tiktoken.Encoding | None = None,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    encoding = encoding or tiktoken.get_encoding(EMBEDDING_ENCODING)
    splits = _split_text(text, SEPARATORS, chunk_size, encoding)
    return _merge_splits(splits, chunk_size, chunk_overlap, encoding)


def _split_text(
    text: str,
    separators: list[str],
    chunk_size: int,
    encoding: tiktoken.Encoding,
) -> list[str]:
    if _token_length(text, encoding) <= chunk_size:
        return [text] if text else []

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator:
        parts = text.split(separator)
    else:
        parts = list(text)

    splits: list[str] = []
    for index, part in enumerate(parts):
        if not part:
            continue

        piece = part if not separator or index == len(parts) - 1 else part + separator
        if _token_length(piece, encoding) <= chunk_size:
            splits.append(piece)
            continue

        if remaining_separators:
            splits.extend(
                _split_text(piece, remaining_separators, chunk_size, encoding)
            )
        else:
            splits.extend(_split_by_tokens(piece, chunk_size, encoding))

    return splits


def _split_by_tokens(
    text: str,
    chunk_size: int,
    encoding: tiktoken.Encoding,
) -> list[str]:
    tokens = encoding.encode(text)
    parts: list[str] = []
    for start in range(0, len(tokens), chunk_size):
        parts.append(encoding.decode(tokens[start : start + chunk_size]))
    return parts


def _merge_splits(
    splits: list[str],
    chunk_size: int,
    chunk_overlap: int,
    encoding: tiktoken.Encoding,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for split in splits:
        split_tokens = _token_length(split, encoding)
        if split_tokens > chunk_size:
            if current:
                chunks.append("".join(current))
                current = []
                current_tokens = 0
            chunks.extend(_split_by_tokens(split, chunk_size, encoding))
            continue

        if current_tokens + split_tokens > chunk_size and current:
            chunks.append("".join(current))
            current, current_tokens = _start_overlap_chunk(
                current, chunk_overlap, encoding
            )

        current.append(split)
        current_tokens += split_tokens

    if current:
        chunks.append("".join(current))

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _start_overlap_chunk(
    pieces: list[str],
    chunk_overlap: int,
    encoding: tiktoken.Encoding,
) -> tuple[list[str], int]:
    if chunk_overlap <= 0:
        return [], 0

    overlap_pieces: list[str] = []
    overlap_tokens = 0
    for piece in reversed(pieces):
        piece_tokens = _token_length(piece, encoding)
        if overlap_tokens + piece_tokens > chunk_overlap and overlap_pieces:
            break
        overlap_pieces.insert(0, piece)
        overlap_tokens += piece_tokens

    return overlap_pieces, overlap_tokens


def _token_length(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text))


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    digest = hashlib.sha256(f"{doc_id}:{chunk_index}".encode()).hexdigest()
    return digest[:16]
