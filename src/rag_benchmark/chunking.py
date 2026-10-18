from __future__ import annotations

import hashlib
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import ChunkRecord, DocumentRecord

_HEADING = re.compile(
    r"^[ \t]*(?:"
    r"(?:BAB|Bab|BAGIAN|Bagian|PASAL|Pasal|LAMPIRAN|Lampiran)[ \t]+[^\n]{1,80}"
    r"|\d+(?:\.\d+){0,3}[.)]?[ \t]+[A-ZÀ-ÖØ-Þ][^\n]{2,80}"
    r"|[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9&()'\-–]*(?:[ \t]+[A-ZÀ-ÖØ-Þ0-9&()'\-–]+){1,11}"
    r")[ \t]*$",
    re.MULTILINE,
)
_TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_SPACE = re.compile(r"[ \t]+")


def clean_document(text: str) -> str:
    """Normalize extracted text and remove Markdown table rendering artifacts."""
    cleaned: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = _SPACE.sub(" ", raw).strip()
        if line.lstrip().startswith("|"):
            if _TABLE_RULE.match(line):
                continue
            cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
            cells = [cell for index, cell in enumerate(cells) if cell and (index == 0 or cell != cells[index - 1])]
            line = f"| {' | '.join(cells)} |" if cells else ""
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def split_structure(text: str) -> list[tuple[str, str]]:
    headings = [match for match in _HEADING.finditer(text) if not match.group(0).rstrip().endswith((".", ",", ";", ":"))]
    if not headings:
        return [("", text)]
    blocks: list[tuple[str, str]] = []
    preamble = text[: headings[0].start()].strip()
    if preamble:
        blocks.append(("", preamble))
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        blocks.append((match.group(0).strip(), text[match.start() : end].strip()))
    return blocks

# https://github.com/robbypambudi/DiLLeMa/blob/main/apps/rag/nlp/doc_chunking.py

class DocumentChunker:
    """DiLLeMa-style structure-first chunking with canonical parent windows."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        min_chunk_chars: int = 150,
        parent_window_chars: int = 5000,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self.parent_window_chars = parent_window_chars
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ".", " "],
        )

    def chunk_document(self, document: DocumentRecord) -> list[ChunkRecord]:
        body = clean_document(document.text)
        raw_pieces: list[tuple[str, str]] = []
        for heading, block in split_structure(body):
            if not block:
                continue
            for piece in self.splitter.split_text(block):
                piece = piece.strip()
                if piece:
                    raw_pieces.append((heading, piece))
        pieces = self._merge_small(raw_pieces)
        chunks: list[ChunkRecord] = []
        cursor = 0
        for index, (heading, evidence) in enumerate(pieces):
            pattern = r"\s+".join(re.escape(word) for word in evidence.split())
            match = re.compile(pattern).search(body, cursor) if pattern else None
            start, end = (match.start(), match.end()) if match else (0, min(len(body), len(evidence)))
            if match:
                cursor = start + 1
                evidence = body[start:end]
            spare = max(0, self.parent_window_chars - (end - start)) // 2
            window_start = max(0, start - spare)
            window_end = min(len(body), window_start + max(self.parent_window_chars, end - start))
            window = body[window_start:window_end]
            index_text = evidence
            if heading and not evidence.startswith(heading):
                index_text = f"{heading}\n{evidence}"
            chunk_id = hashlib.sha256(f"{document.id}:{index}:{index_text}".encode("utf-8")).hexdigest()
            chunks.append(
                ChunkRecord(
                    id=chunk_id,
                    document_id=document.id,
                    dataset=document.dataset,
                    language=document.language,
                    text=index_text,
                    evidence_text=evidence,
                    parent_window=window,
                    parent_id=document.id,
                    source_path=document.source_path,
                    title=document.title,
                    metadata={
                        **document.metadata,
                        "section": heading,
                        "source_start": start,
                        "source_end": end,
                        "offset_basis": "cleaned_document_unicode",
                        "aliases": document.aliases,
                    },
                )
            )
        return chunks

    def _merge_small(self, pieces: list[tuple[str, str]]) -> list[tuple[str, str]]:
        merged: list[tuple[str, str]] = []
        limit = self.chunk_size + self.min_chunk_chars
        for heading, text in pieces:
            if merged:
                old_heading, old_text = merged[-1]
                if (
                    heading == old_heading
                    and min(len(text), len(old_text)) < self.min_chunk_chars
                    and len(text) + len(old_text) + 1 <= limit
                ):
                    merged[-1] = (old_heading, f"{old_text}\n{text}")
                    continue
            merged.append((heading, text))
        return merged

    def chunk_documents(self, documents: list[DocumentRecord]) -> list[ChunkRecord]:
        return [chunk for document in documents for chunk in self.chunk_document(document)]

