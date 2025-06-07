# ingest.py

import os
import pickle
import re
from pathlib import Path
from typing import List, Dict
import ast

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ―― CONFIGURATION ―― #
PROJECT_ROOT  = Path(__file__).parent.resolve()

# ─── where your real code lives ───
CODE_DIRS     = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "yolov5_src",
]

# ─── where to dump everything ───
ARTIFACTS_DIR = PROJECT_ROOT / "data"
CHUNKS_PATH   = ARTIFACTS_DIR / "chunks.pkl"
EMB_PATH      = ARTIFACTS_DIR / "embeddings.npy"
INDEX_PATH    = ARTIFACTS_DIR / "local_index.faiss"
BI_ENCODER    = "microsoft/graphcodebert-base"
DIM           = 768

# Ensure the artifacts directory exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

class Doc:
    """
    Simple container for a code "chunk" plus metadata.
    """
    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata


def extract_chunks_from_file(path: Path) -> List[Doc]:
    """
    Parse a Python file and extract each top‐level function/class definition as a chunk.
    If parsing fails, treat the entire file as one chunk.
    """
    text = path.read_text(encoding='utf-8', errors='ignore')
    try:
        tree = ast.parse(text)
    except SyntaxError:
        meta = {
            "source": str(path.resolve()),
            "rel_path": str(path.relative_to(PROJECT_ROOT)),
            "tags": [path.parent.name, path.stem],
            "doc_summary": "",
            "name_tokens": [],
            "dir_name": path.parent.name,
        }
        return [Doc(text, meta)]

    chunks: List[Doc] = []
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines)

            name = node.name
            docstring = ast.get_docstring(node) or ""
            name_tokens = re.findall(r"[A-Za-z_]\w*", name)
            meta = {
                "source": str(path.resolve()),
                "rel_path": str(path.relative_to(PROJECT_ROOT)),
                "tags": [path.parent.name, path.stem],
                "doc_summary": docstring,
                "name_tokens": name_tokens,
                "dir_name": path.parent.name,
            }
            chunks.append(Doc(chunk_text, meta))
    return chunks


def build_corpus() -> List[Doc]:
    """
    Walk through specified code directories, find all .py files (excluding venv or hidden),
    and extract chunks from each.
    """
    all_chunks: List[Doc] = []
    for code_dir in CODE_DIRS:
        if not code_dir.exists():
            continue
        for pyfile in code_dir.rglob("*.py"):
            if any(part.startswith('.') for part in pyfile.parts) or "venv" in pyfile.parts:
                continue
            all_chunks.extend(extract_chunks_from_file(pyfile))
    return all_chunks


def ingest() -> None:
    """
    Build the corpus of chunks, compute embeddings, and write out FAISS index.
    """
    print("→ Extracting chunks from code corpus…")
    chunks = build_corpus()
    print(f"   → Found {len(chunks)} chunks.")

    print("→ Saving chunks to disk…")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("→ Encoding chunks with CodeBERT…")
    model = SentenceTransformer(BI_ENCODER, device="cpu")
    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)

    print("→ Saving embeddings to disk…")
    np.save(EMB_PATH, embeddings)

    print("→ Building FAISS index…")
    index = faiss.IndexFlatIP(DIM)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    print("✅ Ingestion complete.")


if __name__ == "__main__":
    ingest()

