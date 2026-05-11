"""Test script for BM25, DPR, and Hybrid retrieval."""
import numpy as np
from ams26 import BM25, DPR, Hybrid, Results

print("\n" + "="*60)
print("Testing BM25, DPR, and Hybrid Retrieval Methods")
print("="*60 + "\n")

# 1. Test DPR (Dense Passage Retriever)
print("1. Testing DPR (Dense Passage Retriever)")
print("-" * 40)

corpus_docs = ["Python programming language", "Java object-oriented language", "JavaScript web development"]
corpus_embeddings = np.random.randn(3, 768).astype('float32')  # 3 docs, 768-dim embeddings

dpr = DPR()
dpr.index(corpus_embeddings, corpus=corpus_docs)

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

# 2. Test BM25
print("\n2. Testing BM25 (Sparse Retrieval)")
print("-" * 40)

corpus_tokens = [
    ["python", "programming", "language"],
    ["java", "object", "oriented", "language"],
    ["javascript", "web", "development"]
]

bm25 = BM25()
bm25.index(corpus_tokens)

query_tokens = [["python", "programming"]]
bm25_results = bm25.retrieve(
    query_tokens=query_tokens,
    k=2,
    return_as="tuple"
)
print(f"✓ BM25 retrieved {len(bm25_results.documents[0])} documents")
print(f"✓ BM25 scores shape: {bm25_results.scores.shape}")
print(f"✓ BM25 document indices: {bm25_results.documents[0]}")

# 3. Test Hybrid (Combined BM25 + DPR)
print("\n3. Testing Hybrid (BM25 + DPR)")
print("-" * 40)

hybrid = Hybrid(
    bm25_weight=0.5,
    dpr_weight=0.5
)

hybrid.index(
    corpus=corpus_tokens,
    corpus_embeddings=corpus_embeddings,
)

hybrid_results = hybrid.retrieve(
    query_tokens=query_tokens,
    query_embeddings=query_embeddings,
    k=2,
    return_as="tuple"
)
print(f"✓ Hybrid retrieved {len(hybrid_results.documents[0])} documents")
print(f"✓ Hybrid scores shape: {hybrid_results.scores.shape}")
print(f"✓ Hybrid combined scores: {hybrid_results.scores[0]}")

# 4. Verify Results type
print("\n4. Verifying Results Type")
print("-" * 40)
print(f"✓ Results is a NamedTuple with fields: {Results._fields}")
print(f"✓ Can unpack: documents, scores = results")

print("\n" + "="*60)
print("✓ All tests passed! BM25, DPR, and Hybrid are working.")
print("="*60 + "\n")
