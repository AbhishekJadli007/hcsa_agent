"""
email_parser.py — Parses Email_N.pdf files where each PDF may contain
                  ONE or MULTIPLE email threads (forwarded/reply chains).

Key design decisions:
  - Detects "From:" / "To:" / "Date:" / "Subject:" headers to split emails
    within a single PDF (handles page-break splits by buffering across pages).
  - Each individual email → one chunk (emails are already short).
  - Stores structured metadata: sender, recipients, date_str, subject, 
    email_index_in_thread (1-based), source_file, page_start.
  - Thread-level chunk also stored for "how many emails" counting queries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber
from loguru import logger

from src.core.config import SRC_EMAIL

# ─── Header regex patterns ────────────────────────────────────────────────────
_FROM_RE    = re.compile(r"^From\s*:\s*(.+)$", re.IGNORECASE)
_TO_RE      = re.compile(r"^To\s*:\s*(.+)$", re.IGNORECASE)
_CC_RE      = re.compile(r"^Cc\s*:\s*(.+)$", re.IGNORECASE)
_DATE_RE    = re.compile(r"^(?:Date|Sent)\s*:\s*(.+)$", re.IGNORECASE)
_SUBJECT_RE = re.compile(r"^Subject\s*:\s*(.+)$", re.IGNORECASE)
_SEPARATOR  = re.compile(r"^[-_=]{4,}$")   # ---- or ==== dividers


@dataclass
class EmailChunk:
    text: str                        # full body text of this single email
    source: str                      # filename e.g. "Email_3.pdf"
    source_type: str = SRC_EMAIL
    page_start: int = 1
    sender: str = ""
    recipients: str = ""
    cc: str = ""
    date_str: str = ""
    subject: str = ""
    email_index: int = 1             # position within thread (1-based)
    total_in_thread: int = 1         # filled after all emails parsed
    thread_id: str = ""              # filename stem, for counting queries

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "source_type": self.source_type,
            "page_start": self.page_start,
            "section": f"Email {self.email_index} of {self.total_in_thread}",
            "chunk_index": self.email_index - 1,
            "metadata": {
                "sender": self.sender,
                "recipients": self.recipients,
                "cc": self.cc,
                "date_str": self.date_str,
                "subject": self.subject,
                "email_index": self.email_index,
                "total_in_thread": self.total_in_thread,
                "thread_id": self.thread_id,
            },
        }


def _extract_full_text(path: Path) -> List[tuple[int, str]]:
    """Return list of (page_number, page_text) for every page in the PDF."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            pages.append((i, text))
    return pages


def _split_into_raw_emails(pages: List[tuple[int, str]]) -> List[tuple[int, str]]:
    """
    Detect "From:" boundaries to split a multi-email thread PDF into
    individual raw-text blocks.  Returns list of (page_start, raw_text).
    """
    blocks: List[tuple[int, str]] = []
    current_lines: List[str] = []
    current_page: int = 1
    found_first_from = False

    for page_num, page_text in pages:
        for line in page_text.splitlines():
            stripped = line.strip()

            # A new "From:" line at the beginning signals a new email
            if _FROM_RE.match(stripped):
                if found_first_from and current_lines:
                    # Save previous email block
                    blocks.append((current_page, "\n".join(current_lines)))
                    current_lines = []
                    current_page = page_num
                found_first_from = True

            current_lines.append(line)

    # Flush last block
    if current_lines and found_first_from:
        blocks.append((current_page, "\n".join(current_lines)))

    # If no "From:" found at all, treat whole doc as one email
    if not blocks:
        all_text = "\n".join(t for _, t in pages)
        blocks = [(1, all_text)]

    return blocks


def _parse_headers(raw: str) -> dict:
    """Extract From/To/Cc/Date/Subject from raw email text."""
    headers = {"sender": "", "recipients": "", "cc": "", "date_str": "", "subject": ""}
    for line in raw.splitlines()[:30]:   # headers are always near the top
        stripped = line.strip()
        if m := _FROM_RE.match(stripped):
            headers["sender"] = m.group(1).strip()
        elif m := _TO_RE.match(stripped):
            headers["recipients"] = m.group(1).strip()
        elif m := _CC_RE.match(stripped):
            headers["cc"] = m.group(1).strip()
        elif m := _DATE_RE.match(stripped):
            headers["date_str"] = m.group(1).strip()
        elif m := _SUBJECT_RE.match(stripped):
            headers["subject"] = m.group(1).strip()
    return headers


def parse_email_pdf(path: Path) -> List[EmailChunk]:
    """
    Parse a single Email_N.pdf and return one EmailChunk per message
    found in the thread.
    """
    source_name = path.name
    thread_id   = path.stem   # e.g. "Email_3"

    try:
        pages = _extract_full_text(path)
    except Exception as exc:
        logger.error(f"Cannot open {path}: {exc}")
        return []

    raw_blocks = _split_into_raw_emails(pages)
    emails: List[EmailChunk] = []

    for idx, (page_start, raw) in enumerate(raw_blocks, start=1):
        headers = _parse_headers(raw)
        chunk = EmailChunk(
            text=raw.strip(),
            source=source_name,
            source_type=SRC_EMAIL,
            page_start=page_start,
            email_index=idx,
            thread_id=thread_id,
            **headers,
        )
        emails.append(chunk)

    # Back-fill total_in_thread
    for e in emails:
        e.total_in_thread = len(emails)

    # Also add a thread-level summary chunk for "how many emails in Email_X"
    # so counting queries hit this directly
    if len(emails) > 1:
        summary_text = (
            f"[Thread summary] {source_name} contains {len(emails)} emails. "
            + " | ".join(
                f"Email {e.email_index}: From {e.sender} on {e.date_str} — Subject: {e.subject}"
                for e in emails
            )
        )
        summary_chunk = EmailChunk(
            text=summary_text,
            source=source_name,
            source_type=SRC_EMAIL,
            page_start=1,
            email_index=0,
            total_in_thread=len(emails),
            subject=f"[Thread summary] {len(emails)} emails",
            thread_id=thread_id,
        )
        emails.insert(0, summary_chunk)

    logger.info(f"[Email] {source_name} → {len(emails)} chunks (incl. summary)")
    return emails


def parse_email_directory(directory: Path) -> List[EmailChunk]:
    """Parse ALL pdf files in the Email Repository folder.
    
    Picks up any naming convention:
      Email_1.pdf, Email 1.pdf, email1.pdf, 1.pdf, etc.
    Falls back gracefully if directory doesn't exist.
    """
    all_chunks: List[EmailChunk] = []

    if not directory.exists():
        logger.warning(f"[Email] Directory not found: {directory}")
        return all_chunks

    # Grab every PDF in the folder regardless of naming convention
    pdf_files = sorted(directory.glob("*.pdf")) + sorted(directory.glob("*.PDF"))
    # Deduplicate (glob on case-insensitive FS may double-count)
    seen = set()
    unique_pdfs = []
    for p in pdf_files:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique_pdfs.append(p)

    logger.info(f"[Email] Found {len(unique_pdfs)} email PDFs in {directory}")
    for pdf_path in unique_pdfs:
        all_chunks.extend(parse_email_pdf(pdf_path))
    return all_chunks
