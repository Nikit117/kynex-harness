"""Corpus discovery. Every document gets a stable ID + content fingerprint.

The original file is ALWAYS the source of truth — this module only reads.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Union

SUPPORTED_EXTS = frozenset(
    {
        ".pdf",
        ".docx",
        ".xlsx",
        ".xlsm",
        ".csv",
        ".txt",
        ".md",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".pptx",
    }
)

# Recognized but without a bundled parser: profiled explicitly as
# "unsupported" instead of crashing (see parsers/base.py).
RECOGNIZED_WITHOUT_PARSER = frozenset({".pptx"})


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug[:24] or "doc"


def fingerprint_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class DocumentRef:
    doc_id: str
    path: str
    filename: str
    fingerprint: str
    file_type: str  # lowercase extension, e.g. ".pdf"
    size_bytes: int = 0


@dataclass
class Corpus:
    docs: List[DocumentRef] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.docs)

    def __iter__(self) -> Iterator[DocumentRef]:
        return iter(self.docs)

    def get(self, doc_id: str) -> Optional[DocumentRef]:
        for doc in self.docs:
            if doc.doc_id == doc_id:
                return doc
        return None

    @classmethod
    def load(cls, input_path: Union[str, Path], recursive: bool = True) -> "Corpus":
        root = Path(input_path)
        if root.is_file():
            files = [root]
        elif root.is_dir():
            pattern = "**/*" if recursive else "*"
            files = sorted(
                (
                    p
                    for p in root.glob(pattern)
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
                ),
                key=lambda p: p.name.lower(),
            )
        else:
            raise FileNotFoundError(f"input path does not exist: {input_path}")
        docs = []
        for i, path in enumerate(files, start=1):
            docs.append(
                DocumentRef(
                    doc_id=f"doc_{i:03d}_{_slug(path.stem)}",
                    path=str(path),
                    filename=path.name,
                    fingerprint=fingerprint_file(path),
                    file_type=path.suffix.lower(),
                    size_bytes=path.stat().st_size,
                )
            )
        return cls(docs=docs)
