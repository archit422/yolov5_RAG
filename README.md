# YOLOv5 Code Search

A semantic code search engine for the YOLOv5 codebase using RAG (Retrieval Augmented Generation) techniques.

## Features

- Semantic code search using state-of-the-art embedding models
- Hybrid search combining BM25 and FAISS
- Cross-encoder reranking for improved relevance
- File purpose metadata and tag-based routing
- Evaluation framework for measuring retrieval performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolov5-code-search.git
cd yolov5-code-search
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. First, ingest the YOLOv5 codebase:
```bash
python ingest.py
```

2. Run a search query:
```bash
python retrieve.py "your search query"
```

3. Evaluate the retrieval system:
```bash
python evaluate_retriever.py
```

## Project Structure

- `ingest.py`: Code for processing and indexing the YOLOv5 codebase
- `retrieve.py`: Implementation of the retrieval system
- `evaluate_retriever.py`: Evaluation framework
- `artifacts/`: Directory for storing embeddings and indices
- `yolov5_src/`: The YOLOv5 source code being indexed

## How it Works

1. **Ingestion**:
   - Extracts code chunks from the YOLOv5 codebase
   - Generates embeddings using BGE-Large model
   - Creates FAISS index for fast similarity search
   - Builds BM25 index for lexical search

2. **Retrieval**:
   - Combines semantic and lexical search
   - Uses cross-encoder for reranking
   - Applies file purpose metadata and tag-based routing
   - Provides hybrid scoring with lexical bonuses

3. **Evaluation**:
   - Uses a curated set of test queries
   - Measures accuracy and other metrics
   - Provides detailed performance analysis

## Future Improvements

- Dependency graph analysis for better context understanding
- Code-specific embedding models
- Multi-turn conversation support
- Advanced chunking strategies
- Improved evaluation metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv5 team for the source code
- Sentence Transformers for the embedding models
- FAISS for efficient similarity search 