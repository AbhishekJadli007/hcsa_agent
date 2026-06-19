"""
pdf_chunker.py — Section-aware PDF chunker for SOP/policy docs and annual reports.

Strategy:
  1. Extract text page-by-page with pdfplumber (preserves layout).
  2. Detect heading lines (ALL CAPS or numbered pattern) to split sections.
  3. Chunk within sections at CHUNK_SIZE characters with CHUNK_OVERLAP overlap.
  4. Each chunk carries rich metadata: source_file, page_start, section, source_type.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pdfplumber
from loguru import logger

from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC, SRC_SOP, SRC_REPORT


# ─── Heading detection ────────────────────────────────────────────────────────
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:\d+\.){1,3}\s+[A-Z]"     # 1.2.3 Heading
    r"|[A-Z][A-Z\s\-&]{6,}$"      # ALL CAPS heading
    r"|Annex\s+[A-Z]"             # Annex A / Annex B
    r")",
    re.MULTILINE,
)


@dataclass
class Chunk:
    text: str
    source: str          # filename (no path)
    source_type: str     # SRC_SOP | SRC_REPORT
    page_start: int
    section: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "source_type": self.source_type,
            "page_start": self.page_start,
            "section": self.section,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping windows."""
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def chunk_pdf(path: Path, source_type: str = SRC_SOP) -> List[Chunk]:
    """
    Parse a PDF and return a flat list of Chunk objects.
    Works for SOPs (text-heavy) and annual reports (mixed tables/text).
    """
    source_name = path.name
    chunks: List[Chunk] = []
    current_section = "Introduction"
    buffer_text = ""
    buffer_page = 1
    chunk_idx = 0

    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text(x_tolerance=2, y_tolerance=3) or ""

                # Also extract tables as pipe-separated text rows
                tables_text = ""
                for table in page.extract_tables():
                    for row in table:
                        cleaned_row = " | ".join(
                            str(cell).strip() if cell else "" for cell in row
                        )
                        tables_text += cleaned_row + "\n"

                page_text = raw + ("\n" + tables_text if tables_text else "")

                for line in page_text.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue

                    # Detect heading → flush buffer and start new section
                    if _HEADING_RE.match(stripped) and len(stripped) < 120:
                        if buffer_text.strip():
                            for window in _sliding_window(buffer_text, CHUNK_SIZE, CHUNK_OVERLAP):
                                chunks.append(
                                    Chunk(
                                        text=window,
                                        source=source_name,
                                        source_type=source_type,
                                        page_start=buffer_page,
                                        section=current_section,
                                        chunk_index=chunk_idx,
                                    )
                                )
                                chunk_idx += 1
                                if chunk_idx >= MAX_CHUNKS_PER_DOC:
                                    logger.warning(f"{source_name}: hit MAX_CHUNKS_PER_DOC cap")
                                    return chunks
                        current_section = stripped
                        buffer_text = ""
                        buffer_page = page_num
                    else:
                        buffer_text += " " + stripped

            # Flush remaining buffer
            if buffer_text.strip():
                for window in _sliding_window(buffer_text, CHUNK_SIZE, CHUNK_OVERLAP):
                    chunks.append(
                        Chunk(
                            text=window,
                            source=source_name,
                            source_type=source_type,
                            page_start=buffer_page,
                            section=current_section,
                            chunk_index=chunk_idx,
                        )
                    )
                    chunk_idx += 1

    except Exception as exc:
        logger.error(f"Failed to parse {path}: {exc}")

    logger.info(f"[PDF] {source_name} → {len(chunks)} chunks (type={source_type})")
    return chunks


def chunk_directory(directory: Path, source_type: str) -> List[Chunk]:
    """Chunk all PDFs in a directory."""
    all_chunks: List[Chunk] = []
    pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))
    for pdf_path in sorted(pdf_files):
        all_chunks.extend(chunk_pdf(pdf_path, source_type=source_type))
    return all_chunks


# ── Guard added for missing/empty directories ──────────────────────────────
_original_chunk_directory = chunk_directory

def chunk_directory(directory: Path, source_type: str) -> List[Chunk]:
    if not directory.exists():
        logger.warning(f"[PDF] Directory not found, skipping: {directory}")
        return []
    pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))
    if not pdf_files:
        logger.warning(f"[PDF] No PDFs found in {directory}")
        return []
    return _original_chunk_directory(directory, source_type)
