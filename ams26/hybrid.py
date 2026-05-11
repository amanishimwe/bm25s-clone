"""Hybrid (BM25 + DPR) retrieval."""
import json
from pathlib import Path
from typing import Any, List, Union
import numpy as np

from ._base import __version__, tokenization, json_functions
from .results import Results
from .bm25 import BM25
from .dpr import DPR


class Hybrid:
    """Hybrid retriever combining sparse BM25 and dense DPR with weighted score fusion."""

    def __init__(
        self,
        bm25: BM25 = None,
        dpr: DPR = None,
        bm25_weight: float = 0.5,
        dpr_weight: float = 0.5,
        dtype: str = "float32",
        int_dtype: str = "int32",
    ):
        self.bm25 = bm25
        self.dpr = dpr
        self.dtype = dtype
        self.int_dtype = int_dtype
        self._original_version = __version__

        total = bm25_weight + dpr_weight
        self.bm25_weight = bm25_weight / total if total > 0 else 0.5
        self.dpr_weight = dpr_weight / total if total > 0 else 0.5

    def index(self, corpus, corpus_embeddings: np.ndarray = None,
              bm25_params: dict = None, embedding_fn=None):
        if bm25_params is None:
            bm25_params = {}
        self.bm25 = BM25(**bm25_params)
        self.bm25.index(corpus)
        if corpus_embeddings is not None:
            self.dpr = DPR(corpus_embeddings=corpus_embeddings, corpus=None,
                           embedding_fn=embedding_fn, dtype=self.dtype, int_dtype=self.int_dtype)
            self.dpr.index(corpus_embeddings)

    def _normalize_scores(self, scores: np.ndarray, axis: int = 1) -> np.ndarray:
        s_min = scores.min(axis=axis, keepdims=True)
        s_max = scores.max(axis=axis, keepdims=True)
        s_range = s_max - s_min
        s_range[s_range == 0] = 1.0
        return (scores - s_min) / s_range

    def retrieve(
        self,
        query_tokens: Union[List[List[str]], "tokenization.Tokenized"] = None,
        query_embeddings: np.ndarray = None,
        queries: Union[List[str], np.ndarray] = None,
        corpus: List[Any] = None,
        k: int = 10,
        sorted: bool = True,
        return_as: str = "tuple",
        show_progress: bool = True,
        leave_progress: bool = False,
    ):
        if self.bm25 is None or self.dpr is None:
            raise ValueError("Both BM25 and DPR must be indexed before retrieval.")

        bm25_results = self.bm25.retrieve(
            query_tokens=query_tokens, corpus=corpus, k=k, sorted=sorted,
            return_as="tuple", show_progress=show_progress, leave_progress=leave_progress,
        )
        dpr_results = self.dpr.retrieve(
            queries=queries, query_embeddings=query_embeddings, corpus=corpus,
            k=k, sorted=sorted, return_as="tuple", show_progress=False, leave_progress=leave_progress,
        )

        bm25_norm = self._normalize_scores(bm25_results.scores)
        dpr_norm = self._normalize_scores(dpr_results.scores)

        num_queries = bm25_results.scores.shape[0]
        num_docs = self.bm25.scores["num_docs"]
        combined = np.zeros((num_queries, num_docs), dtype=self.dtype)

        for i in range(num_queries):
            for j, doc_idx in enumerate(bm25_results.documents[i]):
                combined[i, doc_idx] += self.bm25_weight * bm25_norm[i, j]
            for j, doc_idx in enumerate(dpr_results.documents[i]):
                combined[i, doc_idx] += self.dpr_weight * dpr_norm[i, j]

        topk_indices = np.argsort(-combined, axis=1)[:, :k]
        topk_scores = np.take_along_axis(combined, topk_indices, axis=1)

        corpus_obj = corpus if corpus is not None else self.bm25.corpus
        if corpus_obj is None:
            retrieved_docs = topk_indices
        elif isinstance(corpus_obj, np.ndarray) and corpus_obj.ndim == 1:
            retrieved_docs = corpus_obj[topk_indices]
        else:
            retrieved_docs = np.array([[corpus_obj[i] for i in idx] for idx in topk_indices])

        if return_as == "tuple":
            return Results(documents=retrieved_docs, scores=topk_scores)
        elif return_as == "documents":
            return retrieved_docs
        else:
            raise ValueError("`return_as` must be 'tuple' or 'documents'")

    def save(self, save_dir: str, corpus: List[Any] = None,
             bm25_dir: str = "bm25_index", dpr_dir: str = "dpr_index",
             params_name: str = "params.json"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.bm25:
            self.bm25.save(save_dir / bm25_dir, corpus=corpus)
        if self.dpr:
            self.dpr.save(save_dir / dpr_dir, corpus=corpus)
        with open(save_dir / params_name, "w") as f:
            json.dump({"bm25_weight": float(self.bm25_weight), "dpr_weight": float(self.dpr_weight),
                       "dtype": self.dtype, "int_dtype": self.int_dtype, "version": __version__}, f, indent=4)

    @classmethod
    def load(cls, save_dir: str, bm25_dir: str = "bm25_index", dpr_dir: str = "dpr_index",
             params_name: str = "params.json", load_corpus: bool = False,
             embedding_fn=None) -> "Hybrid":
        save_dir = Path(save_dir)
        with open(save_dir / params_name, "r") as f:
            params = json.load(f)
        bm25 = BM25.load(save_dir / bm25_dir, load_corpus=load_corpus)
        dpr = DPR.load(save_dir / dpr_dir, load_corpus=load_corpus,
                       embedding_fn=embedding_fn,
                       dtype=params.get("dtype", "float32"),
                       int_dtype=params.get("int_dtype", "int32"))
        return cls(bm25=bm25, dpr=dpr,
                   bm25_weight=params.get("bm25_weight", 0.5),
                   dpr_weight=params.get("dpr_weight", 0.5),
                   dtype=params.get("dtype", "float32"),
                   int_dtype=params.get("int_dtype", "int32"))