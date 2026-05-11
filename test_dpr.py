"""Test script for DPR (Dense Passage Retriever)."""
import numpy as np
from ams26 import DPR

print("\n" + "="*60)
print("Testing DPR (Dense Passage Retriever)")
print("="*60 + "\n")

corpus_docs = [
    "Python programming language",
    "Java object-oriented language",
    "JavaScript web development"
]
corpus_embeddings = np.random.randn(3, 768).astype('float32')  # 3 docs, 768-dim embeddings

dpr = DPR()
dpr.index(corpus_embeddings, corpus=corpus_docs)
print("✓ DPR index built successfully")

query_embeddings = np.random.randn(1, 768).astype('float32')
dpr_results = dpr.retrieve(
    queries=None,
    query_embeddings=query_embeddings,
    k=2,
    return_as="tuple"
)

print(f"✓ DPR retrieved {len(dpr_results.documents[0])} documents")
print(f"✓ DPR scores shape: {dpr_results.scores.shape}")
print(f"✓ Retrieved docs: {dpr_results.documents[0]}")
print(f"✓ Retrieved scores: {dpr_results.scores[0]}")

print("\n" + "="*60)
print("✓ DPR test passed!")
print("="*60 + "\n")