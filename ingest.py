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
PROJECT_ROOT  = Path(__file__).parent / "yolov5_src"

# ── PUT YOUR OUTPUT ARTIFACTS HERE ──
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CHUNKS_PATH   = ARTIFACTS_DIR / "chunks.pkl"
EMB_PATH      = ARTIFACTS_DIR / "embeddings.npy"
INDEX_PATH    = ARTIFACTS_DIR / "local_index.faiss"

BI_ENCODER    = "BAAI/bge-large-en-v1.5"
DIM           = 1024

# ── POINT THIS AT YOUR CODE FOLDER ──

class Doc:
    """
    Simple container for a code "chunk" plus metadata.
    """
    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata


def extract_chunks_from_file(path: Path) -> List[Doc]:
    """
    Parse a Python file and extract all code elements as chunks:
    - Top-level functions and classes
    - Methods inside classes
    - Nested functions
    - Code outside functions/classes
    """
    text = path.read_text(encoding='utf-8', errors='ignore')
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # fallback: entire file is one chunk
        meta = {
            "source": str(path.resolve()),
            "rel_path": str(path.relative_to(PROJECT_ROOT)),
            "tags": [path.parent.name, path.stem],
            "doc_summary": "",
            "name_tokens": [],
            "dir_name": path.parent.name,
            "type": "file",
            "parent": None
        }
        return [Doc(text, meta)]

    chunks: List[Doc] = []
    lines = text.splitlines()

    def get_node_text(node):
        """Get the source text for a node."""
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
        return "\n".join(lines[start:end])

    def process_node(node, parent_name=None, parent_type=None):
        """Process a node and its children recursively."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Process function
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
                "type": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                "parent": parent_name,
                "parent_type": parent_type
            }
            chunks.append(Doc(get_node_text(node), meta))

            # Process nested functions
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    process_node(child, f"{parent_name}.{name}" if parent_name else name, 
                               "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")

        elif isinstance(node, ast.ClassDef):
            # Process class
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
                "type": "class",
                "parent": parent_name,
                "parent_type": parent_type
            }
            chunks.append(Doc(get_node_text(node), meta))

            # Process methods and nested classes
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    process_node(child, f"{parent_name}.{name}" if parent_name else name, "class")

        # Process any standalone code (outside functions/classes)
        elif not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
            # Only create chunks for non-trivial standalone code
            if len(get_node_text(node).strip()) > 0:
                meta = {
                    "source": str(path.resolve()),
                    "rel_path": str(path.relative_to(PROJECT_ROOT)),
                    "tags": [path.parent.name, path.stem],
                    "doc_summary": "",
                    "name_tokens": [],
                    "dir_name": path.parent.name,
                    "type": "standalone_code",
                    "parent": parent_name,
                    "parent_type": parent_type
                }
                chunks.append(Doc(get_node_text(node), meta))

    # Process all top-level nodes
    for node in tree.body:
        process_node(node)

    return chunks


def build_corpus() -> List[Doc]:
    """
    Walk through the project directory, find all .py files (excluding venv or hidden),
    and extract chunks from each.
    """
    all_chunks: List[Doc] = []
    file_count = 0
    for pyfile in PROJECT_ROOT.rglob("*.py"):
        # Skip virtual environment or hidden directories
        if any(part.startswith('.') for part in pyfile.parts):
            continue
        if "venv" in pyfile.parts:
            continue
        file_count += 1
        file_chunks = extract_chunks_from_file(pyfile)
        print(f"File: {pyfile.relative_to(PROJECT_ROOT)} - Chunks: {len(file_chunks)}")
        all_chunks.extend(file_chunks)
    print(f"Total Python files processed: {file_count}")
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

    print("→ Encoding chunks with BGE-Large…")
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
    
    # Write debug info to a file
    with open("debug_info.txt", "w") as debug_file:
        # Get chunk info
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
            debug_file.write(f"DEBUG INFO:\n")
            debug_file.write(f"Number of chunks: {len(chunks)}\n")
            debug_file.write(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
            
            # Count Python files in PROJECT_ROOT
            py_files = list(PROJECT_ROOT.rglob("*.py"))
            py_files = [f for f in py_files if not any(part.startswith('.') for part in f.parts) and "venv" not in f.parts]
            debug_file.write(f"Number of Python files in {PROJECT_ROOT}: {len(py_files)}\n")
            
            # Write first few chunks for debugging
            debug_file.write("\nFirst 5 chunks:\n")
            for i, chunk in enumerate(chunks[:5]):
                debug_file.write(f"Chunk {i+1}: {chunk.metadata['rel_path']}\n")
