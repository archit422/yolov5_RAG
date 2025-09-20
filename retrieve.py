# retrieve.py

import os
import re
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

CHUNKS_PATH    = "artifacts/chunks.pkl"
EMB_PATH       = "artifacts/embeddings.npy"
INDEX_PATH     = "artifacts/local_index.faiss"
EMBED_MODEL    = "BAAI/bge-large-en-v1.5"  # Better semantic search model
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
DIM            = 1024 

K_LEX          = 100     
MIN_LEX_MATCH  = 2       
K_CAND         = 200    
K_FINAL        = 10      
RERANK_K       = 50    

# Reranking weights
LEX_BONUS_EXACT = 1.0    # Bonus for exact token matches
LEX_BONUS_PARTIAL = 0.3  # Bonus for partial matches
TAG_BONUS = 0.2         # Bonus for tag matches
NAME_BONUS = 0.5        # Bonus for matches in function/class names

ordered_file_keys = [
    ("train", "train.py"),
    ("val", "val.py"),
    ("detect", "detect.py"),
    ("export", "export.py"),
    ("hubconf", "hubconf.py"),
    ("metrics", "utils/metrics.py"),
    ("plots", "utils/plots.py"),
    ("dataloaders", "utils/dataloaders.py"),
    ("autoanchor", "utils/autoanchor.py"),
    ("torch_utils", "utils/torch_utils.py"),
    ("augmentations", "utils/augmentations.py"),
    ("loss", "utils/loss.py"),
    ("general", "utils/general.py"),
    ("downloads", "utils/downloads.py"),
    ("callbacks", "utils/callbacks.py"),
    ("activations", "utils/activations.py"),
    ("autobatch", "utils/autobatch.py"),
    ("triton", "utils/triton.py"),
    ("models", "models/yolo.py"),
    ("common", "models/common.py"),
    ("experimental", "models/experimental.py"),
    ("tf", "models/tf.py"),
    ("segment", "segment/"),
    ("classify", "classify/"),
]

KEYWORD_TO_TAG = {
    "train": "train",
    "val": "val",
    "detect": "detect",
    "export": "export",
    "hubconf": "hubconf",
    "metrics": "metrics",
    "plots": "plots",
    "dataloaders": "dataloaders",
    "autoanchor": "autoanchor",
    "torch_utils": "torch_utils",
    "augmentations": "augmentations",
    "loss": "loss",
    "general": "general",
    "downloads": "downloads",
    "callbacks": "callbacks",
    "activations": "activations",
    "autobatch": "autobatch",
    "triton": "triton",
    "models": "models",
    "common": "common",
    "experimental": "experimental",
    "tf": "tf",
    "segment": "segment",
    "classify": "classify",
}

# Query intent categories
QUERY_INTENTS = {
    'model': ['model', 'network', 'backbone', 'head', 'layer', 'architecture', 'yolo'],
    'training': ['train', 'optimizer', 'loss', 'scheduler', 'epoch', 'batch', 'learning rate'],
    'data': ['dataset', 'dataloader', 'augment', 'transform', 'image', 'label', 'annotation'],
    'metrics': ['metric', 'compute', 'calculate', 'iou', 'ap', 'precision', 'recall', 'f1'],
    'visualization': ['plot', 'draw', 'visualize', 'show', 'display', 'save image']
}

# File purpose metadata
FILE_PURPOSES = {
    'models/yolo.py': ['model', 'architecture'],
    'models/common.py': ['model', 'layer'],
    'models/tf.py': ['model', 'tensorflow'],
    'utils/datasets.py': ['data', 'dataset'],
    'utils/dataloaders.py': ['data', 'dataloader'],
    'utils/augmentations.py': ['data', 'augmentation'],
    'utils/metrics.py': ['metrics', 'computation'],
    'utils/general.py': ['utility', 'helper'],
    'utils/plots.py': ['visualization', 'plotting'],
    'utils/torch_utils.py': ['utility', 'pytorch'],
    'train.py': ['training', 'optimization'],
    'val.py': ['evaluation', 'validation'],
    'detect.py': ['inference', 'detection'],
    'export.py': ['export', 'conversion'],
    'utils/autoanchor.py': ['model', 'anchors'],
    'utils/segment/general.py': ['segmentation', 'mask'],
    'utils/segment/metrics.py': ['segmentation', 'metrics'],
    'utils/segment/dataloaders.py': ['segmentation', 'data'],
    'utils/segment/segment.py': ['segmentation', 'inference'],
    'classify/train.py': ['classification', 'training'],
    'hubconf.py': ['model', 'hub'],
}

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

# Load reranker
reranker = CrossEncoder(RERANK_MODEL, local_files_only=True)

def is_identifier_query(query: str) -> bool:
    pattern = r"""
        [A-Za-z]+_[A-Za-z0-9]+  
      | [a-z]+[A-Z][A-Za-z]+           
      | [A-Za-z_]\w+\(\)          
      | [A-Za-z_]\w+\.[A-Za-z_]\w+   
    """
    return bool(re.search(pattern, query, re.VERBOSE))

def get_name_matches(text: str, query_tokens: set[str]) -> float:
    """Check if query tokens appear in function/class names."""
    name_pattern = r"(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"
    names = re.findall(name_pattern, text)
    return sum(1 for name in names if any(t in name.lower() for t in query_tokens)) * NAME_BONUS

def get_query_intent(query):
    """Classify query intent based on keywords."""
    query = query.lower()
    scores = {intent: 0 for intent in QUERY_INTENTS}
    
    for intent, keywords in QUERY_INTENTS.items():
        for keyword in keywords:
            if keyword in query:
                scores[intent] += 1
                
    # Get top 2 intents
    top_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
    return [intent for intent, score in top_intents if score > 0]

def get_file_relevance(file_path, intents):
    """Score file relevance based on its purpose and query intents."""
    file_name = file_path.split('/')[-2] + '/' + file_path.split('/')[-1]
    purposes = FILE_PURPOSES.get(file_name, [])
    
    # Count matching purposes
    matches = sum(1 for intent in intents if intent in purposes)
    return matches / len(intents) if intents else 0

def retrieve(
    query: str,
    k_candidates: int | None = None,
    k_final:      int | None = None
) -> list[dict]:
    """
    Runs hybrid BM25 + FAISS search with cross-encoder reranking.
    """
    q = query.strip().lower()
    identifier_mode = is_identifier_query(q)
    q_tokens = re.findall(r"[A-Za-z_]\w*", q)

    # file-based routing
    domain_idxs = None
    for key, filepath in ordered_file_keys:
        if key in q:
            idxs = [i for i, c in enumerate(chunks) if c.metadata["source"].endswith(filepath)]
            if idxs:
                domain_idxs = np.array(idxs, dtype=np.int64)
            break

    #tag-based routing
    if domain_idxs is None:
        for key, tag in KEYWORD_TO_TAG.items():
            if key in q:
                idxs = [i for i, c in enumerate(chunks) if tag in c.metadata.get("tags", [])]
                if idxs:
                    domain_idxs = np.array(idxs, dtype=np.int64)
                break

    if domain_idxs is None or len(domain_idxs) == 0:
        domain_idxs = np.arange(len(chunks), dtype=np.int64)

    # BM25 prefilter
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

    # Initial rerank with lexical bonuses
    rerank = []
    for sc, idx in zip(sem_scores[0], cand_idxs):
        bonus = 0.0
        chunk_text = chunks[idx].page_content.lower()
        tokset = set(re.findall(r"[A-Za-z_]\w*", chunk_text))
        
        # Exact token matches
        exact_matches = sum(1 for t in q_tokens if t in tokset)
        bonus += exact_matches * LEX_BONUS_EXACT
        
        # Partial matches
        partial_matches = sum(0.5 for t in q_tokens if any(t in tok for tok in tokset))
        bonus += partial_matches * LEX_BONUS_PARTIAL
        
        # Tag matches
        if any(t in chunks[idx].metadata.get("tags", []) for t in q_tokens):
            bonus += TAG_BONUS
            
        # Name matches
        bonus += get_name_matches(chunks[idx].page_content, set(q_tokens))
        
        rerank.append((sc + bonus, idx))

    # Take top candidates for cross-encoder reranking
    rerank.sort(key=lambda x: x[0], reverse=True)
    top_candidates = rerank[:RERANK_K]
    
    # Cross-encoder reranking
    if len(top_candidates) > 1:
        pairs = [(query, chunks[idx].page_content) for _, idx in top_candidates]
        rerank_scores = reranker.predict(pairs)
        
        # Combine semantic and reranking scores
        final_scores = []
        for (sem_score, idx), rerank_score in zip(top_candidates, rerank_scores):
            final_score = 0.7 * rerank_score + 0.3 * sem_score  # Weighted combination
            final_scores.append((final_score, idx))
    else:
        final_scores = top_candidates

    # Sort by final scores and take top k
    final_scores.sort(key=lambda x: x[0], reverse=True)
    use_k_final = k_final if k_final is not None else K_FINAL
    final = final_scores[:use_k_final]

    return [
        {
            "source": chunks[i].metadata["source"],
            "score": float(s),
            "tags":  chunks[i].metadata.get("tags", []),
            "snippet": chunks[i].page_content,
        }
        for s, i in final
    ]


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
