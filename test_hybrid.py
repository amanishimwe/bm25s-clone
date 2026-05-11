"""Test script for Hybrid (BM25 + DPR) Retrieval."""
import numpy as np
from ams26 import Hybrid

print("\n" + "="*60)
print("Testing Hybrid (BM25 + DPR) Retrieval")
print("="*60 + "\n")

corpus_docs = [
    "Python programming language",
    "Java object-oriented language",
    "JavaScript web development"
]
corpus_tokens = [
    ["python", "programming", "language"],
    ["java", "object", "oriented", "language"],
    ["javascript", "web", "development"]
]
corpus_embeddings = np.random.randn(3, 768).astype('float32')  # 3 docs, 768-dim embeddings

hybrid = Hybrid(bm25_weight=0.5, dpr_weight=0.5)
hybrid.index(
    corpus=corpus_tokens,
    corpus_embeddings=corpus_embeddings,
)
print("✓ Hybrid index built successfully")
print(f"✓ BM25 weight: {hybrid.bm25_weight}, DPR weight: {hybrid.dpr_weight}")

query_tokens = [["python", "programming"]]
query_embeddings = np.random.randn(1, 768).astype('float32')

hybrid_results = hybrid.retrieve(
    query_tokens=query_tokens,
    query_embeddings=query_embeddings,
    corpus=corpus_docs,         # pass readable docs for output
    k=2,
    return_as="tuple"
)

print(f"✓ Hybrid retrieved {len(hybrid_results.documents[0])} documents")
print(f"✓ Hybrid scores shape: {hybrid_results.scores.shape}")
print(f"✓ Retrieved docs: {hybrid_results.documents[0]}")
print(f"✓ Hybrid combined scores: {hybrid_results.scores[0]}")

print("\n" + "="*60)
print("✓ Hybrid test passed!")
print("="*60 + "\n")