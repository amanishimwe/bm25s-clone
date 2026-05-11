"""Test script for BM25 (Sparse Retrieval)."""
from ams26 import BM25
 
print("\n" + "="*60)
print("Testing BM25 (Sparse Retrieval)")
print("="*60 + "\n")
 
corpus_tokens = [
    ["python", "programming", "language"],
    ["java", "object", "oriented", "language"],
    ["javascript", "web", "development"]
]
 
bm25 = BM25()
bm25.index(corpus_tokens)
print("✓ BM25 index built successfully")
 
query_tokens = [["python", "programming"]]
bm25_results = bm25.retrieve(
    query_tokens,
    k=2,
    return_as="tuple"
)
 
print(f"✓ BM25 retrieved {len(bm25_results.documents[0])} documents")
print(f"✓ BM25 scores shape: {bm25_results.scores.shape}")
print(f"✓ BM25 document indices: {bm25_results.documents[0]}")
print(f"✓ BM25 scores: {bm25_results.scores[0]}")
 
print("\n" + "="*60)
print("✓ BM25 test passed!")
print("="*60 + "\n")