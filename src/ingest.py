"""Load documents and split them into overlapping, heading-aware chunks."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
EVAL_PATH = ROOT / "data" / "eval_questions.json"


@dataclass
class Chunk:
    text: str
    source: str   # file name
    section: str  # nearest heading ("" if none)

    @property
    def label(self) -> str:
        return f"{self.source} > {self.section}" if self.section else self.source


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8-sig")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text on headings -> [(heading, body)]."""
    parts, heading, buf = [], "", []
    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                parts.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        parts.append((heading, "\n".join(buf).strip()))
    return parts


def chunk_words(text: str, size: int = 120, overlap: int = 25) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    out, step = [], size - overlap
    for i in range(0, len(words), step):
        out.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
    return out


def load_chunks(folder: str | Path = DOCS_DIR, size: int = 120, overlap: int = 25) -> list[Chunk]:
    if not Path(folder).is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder}")
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        for heading, body in split_sections(read_file(path)):
            for piece in chunk_words(body, size, overlap):
                chunks.append(Chunk(piece, path.name, heading))
    return chunks
