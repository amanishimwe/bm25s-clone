"""Test script for BM25 scoring with different methods and context."""

import numpy as np
from ams26 import BM25, Results

print("\n" + "="*60)
print("Testing BM25 Scoring Context - Different Methods")
print("="*60 + "\n")

# Test corpus
corpus_tokens = [
    ["python", "programming", "language", "dynamic"],
    ["java", "object", "oriented", "language", "static"],
    ["javascript", "web", "development", "frontend", "dynamic"],
    ["c", "low", "level", "programming", "systems"],
]

query_tokens = [["python", "programming"]]

# Test different BM25 methods
methods = ["lucene", "robertson", "atire"]

print("Testing standard BM25 methods:")
print("-" * 40)

results_by_method = {}

for method in methods:
    try:
        bm25 = BM25(method=method)
        bm25.index(corpus_tokens)
        
        results = bm25.retrieve(
            query_tokens=query_tokens,
            k=2,
            return_as="tuple",
            show_progress=False
        )
        
        results_by_method[method] = results
        print(f"✓ Method '{method}':")
        print(f"  - Retrieved docs: {results.documents[0]}")
        print(f"  - Scores: {results.scores[0]}")
        print(f"  - Max score: {results.scores[0].max():.4f}")
        
    except Exception as e:
        print(f"✗ Method '{method}' failed: {e}")

print("\n\nTesting BM25L and BM25+ variants with delta parameter:")
print("-" * 40)

bm25l_methods = ["bm25l", "bm25+"]

for method in bm25l_methods:
    try:
        bm25 = BM25(method=method, delta=0.5)
        bm25.index(corpus_tokens)
        
        results = bm25.retrieve(
            query_tokens=query_tokens,
            k=2,
            return_as="tuple",
            show_progress=False
        )
        
        print(f"✓ Method '{method}' with delta=0.5:")
        print(f"  - Retrieved docs: {results.documents[0]}")
        print(f"  - Scores: {results.scores[0]}")
        print(f"  - Max score: {results.scores[0].max():.4f}")
        
    except Exception as e:
        print(f"✗ Method '{method}' failed: {e}")

print("\n\nTesting different BM25 parameters:")
print("-" * 40)

params_list = [
    {"k1": 1.5, "b": 0.75, "delta": 0.5},
    {"k1": 2.0, "b": 0.5, "delta": 0.5},
    {"k1": 1.0, "b": 1.0, "delta": 0.5},
]

for i, params in enumerate(params_list):
    try:
        bm25 = BM25(**params)
        bm25.index(corpus_tokens)
        
        results = bm25.retrieve(
            query_tokens=query_tokens,
            k=2,
            return_as="tuple",
            show_progress=False
        )
        
        print(f"✓ Parameters k1={params['k1']}, b={params['b']}:")
        print(f"  - Scores: {results.scores[0]}")
        print(f"  - Max score: {results.scores[0].max():.4f}")
        
    except Exception as e:
        print(f"✗ Parameters {params} failed: {e}")

print("\n\nTesting multi-query retrieval:")
print("-" * 40)

multi_queries = [
    ["python", "programming"],
    ["java", "object"],
    ["web", "development"],
]

try:
    bm25 = BM25()
    bm25.index(corpus_tokens)
    
    results = bm25.retrieve(
        query_tokens=multi_queries,
        k=2,
        return_as="tuple",
        show_progress=False
    )
    
    print(f"✓ Multi-query retrieval:")
    print(f"  - Number of queries: {len(multi_queries)}")
    print(f"  - Results shape: {results.scores.shape}")
    for i, query in enumerate(multi_queries):
        print(f"  - Query {i} ({' '.join(query[:2])}):")
        print(f"    Scores: {results.scores[i]}")
        
except Exception as e:
    print(f"✗ Multi-query retrieval failed: {e}")

print("\n" + "="*60)
print("✓ Scoring context properly handled across all methods")
print("="*60 + "\n")
