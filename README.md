# BM25s

A high-performance, production-ready implementation of BM25 (Best Matching 25) ranking algorithm for full-text search and information retrieval.

## Overview

BM25s is a Python library that provides an efficient implementation of the BM25 ranking function, one of the most effective non-learned text retrieval algorithms. BM25 is widely used in search engines, information retrieval systems, and semantic search applications.

## What is BM25?

BM25 (Best Matching 25) is a probabilistic relevance ranking function used to estimate the relevance of documents to a given search query. It combines:

- **Term frequency (TF)**: How often a term appears in a document
- **Inverse document frequency (IDF)**: How rare a term is across all documents
- **Document length normalization**: Adjusting scores based on document length

## Features

- ⚡ **Fast & Efficient**: Optimized for speed with minimal memory overhead
- 📚 **Scalable**: Handles large document collections
- 🔧 **Easy to Use**: Simple API for indexing and searching
- 📊 **Production Ready**: Thoroughly tested and battle-hardened
- 🎯 **Configurable**: Tunable parameters (k1, b values)
- 📖 **Well Documented**: Comprehensive documentation and examples

## Installation

Install BM25s using pip:

```bash
pip install bm25s
```

Or with conda:

```bash
conda install -c conda-forge bm25s
```

## Quick Start

### Basic Usage

```python
import bm25s

# Create a BM25 retriever
retriever = bm25s.BM25()

# Index documents
documents = [
    "Python is a programming language",
    "Java is used for building applications",
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks"
]

corpus_tokens = [doc.split() for doc in documents]
retriever.index(corpus_tokens)

# Search
query = "machine learning"
query_tokens = query.split()
results, scores = retriever.retrieve(query_tokens, k=2)

for i, (result, score) in enumerate(zip(results, scores)):
    print(f"{i+1}. {documents[result]} (score: {score:.4f})")
```

## Configuration

### BM25 Parameters

Customize BM25 behavior with parameters:

```python
retriever = bm25s.BM25(
    k1=1.5,      # Controls term frequency saturation point
    b=0.75,      # Controls how much document length affects relevance
    epsilon=0.25 # Small epsilon value for IDF
)
```

## API Reference

### `BM25()`

Main class for BM25 retrieval.

**Parameters:**
- `k1` (float): Term frequency saturation parameter (default: 1.5)
- `b` (float): Document length normalization parameter (default: 0.75)
- `epsilon` (float): IDF epsilon value (default: 0.25)

**Methods:**
- `index(corpus_tokens)`: Index a tokenized corpus
- `retrieve(query_tokens, k=10)`: Retrieve top-k documents for a query
- `get_scores(query_tokens)`: Get relevance scores for all documents

## Performance

BM25s is optimized for performance:

- Indexing: O(n) where n is the total number of tokens
- Searching: O(m * avg_doc_len) where m is the number of queries
- Memory: Efficient sparse representation of the corpus

## Examples

### Example 1: Simple Search

```python
import bm25s

retriever = bm25s.BM25()
docs = ["cat dog", "dog bird", "cat bird dog"]
retriever.index([doc.split() for doc in docs])

query = "cat"
results, scores = retriever.retrieve(query.split(), k=2)
```

### Example 2: Working with Real Data

```python
import bm25s
import pandas as pd

# Load documents from CSV
df = pd.read_csv('documents.csv')
documents = df['content'].tolist()

# Create and index retriever
retriever = bm25s.BM25()
corpus_tokens = [doc.lower().split() for doc in documents]
retriever.index(corpus_tokens)

# Search
query = "your search query"
results, scores = retriever.retrieve(query.lower().split(), k=10)

for doc_idx, score in zip(results, scores):
    print(f"{documents[doc_idx]}: {score:.4f}")
```

## Comparison with Alternatives

| Feature | BM25s | Elasticsearch | Whoosh |
|---------|-------|---------------|--------|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Size | Lightweight | Heavy | Medium |
| Learning Curve | Easy | Steep | Medium |
| Python Native | ✓ | ✗ | ✓ |

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond
- [BM25 Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)

## Support

For questions and support, please open an issue on GitHub or visit our documentation site.
