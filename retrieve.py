

# retrieve.py

import os
import re
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# —— CONFIGURATION & HYPERPARAMETERS —— #
CHUNKS_PATH    = "data/chunks.pkl"
EMB_PATH       = "data/embeddings.npy"
INDEX_PATH     = "data/local_index.faiss"
EMBED_MODEL    = "microsoft/graphcodebert-base"
DIM            = 768

# Hybrid-search hyperparams
K_LEX          = 30      # how many BM25-lexical candidates to pass into FAISS
MIN_LEX_MATCH  = 1       # require at least 1 exact token overlap for BM25 preselection
K_CAND         = 50      # default how many FAISS candidates to fetch
K_FINAL        = 5       # default how many final results to return

# ── your ordered_file_keys and KEYWORD_TO_TAG definitions here ──
# (copy exactly from your original retrieve.py)

# —— LOAD artifacts —— #
with open(CHUNKS_PATH, "rb") as f:
    chunks: list["Doc"] = pickle.load(f)

if not os.path.exists(EMB_PATH):
    raise FileNotFoundError("Missing embeddings.npy—run ingest.py first.")
embeddings = np.load(EMB_PATH)

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("Missing local_index.faiss—run ingest.py first.")
index = faiss.read_index(INDEX_PATH)

# Build BM25 index
token_lists = [
    re.findall(r"[A-Za-z_]\w*", c.page_content.lower())
    for c in chunks
]
bm25 = BM25Okapi(token_lists)

# Load embedder
embedder = SentenceTransformer(EMBED_MODEL, local_files_only=True)


def is_identifier_query(query: str) -> bool:
    pattern = r"""
        [A-Za-z]+_[A-Za-z0-9]+         # snake_case
      | [a-z]+[A-Z][A-Za-z]+            # camelCase
      | [A-Za-z_]\w+\(\)             # foo_bar() or doSomething()
      | [A-Za-z_]\w+\.[A-Za-z_]\w+    # Module.Class
    """
    return bool(re.search(pattern, query, re.VERBOSE))


def retrieve(
    query: str,
    k_candidates: int | None = None,
    k_final:      int | None = None
) -> list[dict]:
    """
    Runs hybrid BM25 + FAISS search and returns the top chunks.
    """
    q = query.strip().lower()
    identifier_mode = is_identifier_query(q)
    q_tokens = re.findall(r"[A-Za-z_]\w*", q)

    # Step 1: file-based routing
    domain_idxs = None
    for key, filepath in ordered_file_keys:
        if key in q:
            idxs = [i for i, c in enumerate(chunks) if c.metadata["source"].endswith(filepath)]
            if idxs:
                domain_idxs = np.array(idxs, dtype=np.int64)
            break

    # Step 2: tag-based routing
    if domain_idxs is None:
        for key, tag in KEYWORD_TO_TAG.items():
            if key in q:
                idxs = [i for i, c in enumerate(chunks) if tag in c.metadata.get("tags", [])]
                if idxs:
                    domain_idxs = np.array(idxs, dtype=np.int64)
                break

    if domain_idxs is None or len(domain_idxs) == 0:
        domain_idxs = np.arange(len(chunks), dtype=np.int64)

    # Step 3: BM25 prefilter
    bm25_scores = bm25.get_scores(q_tokens)
    subset = domain_idxs
    lex_overlap = np.zeros(len(subset), dtype=int)
    for pos, idx in enumerate(subset):
        chunk_tokens = set(re.findall(r"[A-Za-z_]\w*", chunks[idx].page_content.lower()))
        lex_overlap[pos] = sum(1 for t in q_tokens if t in chunk_tokens)

    eligible = subset[lex_overlap >= MIN_LEX_MATCH]
    if eligible.size == 0:
        eligible = subset
    scores = bm25_scores[eligible]
    order = np.argsort(scores)[::-1]
    top_lex = eligible[order][:K_LEX] if len(order) > 0 else subset

    # Step 4: semantic search
    q_emb = embedder.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    use_k_cand  = k_candidates if k_candidates is not None else K_CAND
    use_k_final = k_final      if k_final      is not None else K_FINAL

    sub_embs = embeddings[top_lex]
    sub_index = faiss.IndexFlatIP(DIM)
    sub_index.add(sub_embs)

    n_search = min(use_k_cand, top_lex.shape[0])
    sem_scores, ids = sub_index.search(q_emb, n_search)
    cand_idxs = [int(top_lex[i]) for i in ids[0]]

    # Step 5: rerank with lexical bonuses
    rerank = []
    for sc, idx in zip(sem_scores[0], cand_idxs):
        bonus = 0.0
        tokset = set(re.findall(r"[A-Za-z_]\w*", chunks[idx].page_content.lower()))
        if identifier_mode and any(t in tokset for t in q_tokens):
            bonus += 1.0
        elif any(t in tokset for t in q_tokens):
            bonus += 0.3
        if any(t in chunks[idx].metadata.get("tags", []) for t in q_tokens):
            bonus += 0.2
        rerank.append((sc + bonus, idx))

    rerank.sort(key=lambda x: x[0], reverse=True)
    final = rerank[:use_k_final]

    return [
        {
            "source": chunks[i].metadata["source"],
            "score": float(s),
            "tags":  chunks[i].metadata.get("tags", []),
            "snippet": chunks[i].page_content,
        }
        for s, i in final
    ]


# CLI test harness
if __name__ == "__main__":
    q = input("Enter your code query: ")
    hits = retrieve(q)
    if not hits:
        print("No results found.")
    else:
        for i, h in enumerate(hits, 1):
            print(f"\n--- Result #{i} ({h['source']}, score={h['score']:.3f}) ---")
            print("Tags:", h['tags'])
            print(h['snippet'])
