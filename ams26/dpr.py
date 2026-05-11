"""DPR (Dense Passage Retriever) dense retrieval."""
import logging
from pathlib import Path
from typing import Any, List, Union
import numpy as np

from ._base import __version__, json_functions
from .results import Results

logger = logging.getLogger("bm25s")


class DPR:
    """Dense Passage Retriever using cosine similarity over pre-computed embeddings."""

    def __init__(
        self,
        corpus_embeddings: np.ndarray = None,
        corpus: List[Any] = None,
        embedding_fn=None,
        dtype="float32",
        int_dtype="int32",
    ):
        self.corpus_embeddings = corpus_embeddings
        self.corpus = corpus
        self.embedding_fn = embedding_fn
        self.dtype = dtype
        self.int_dtype = int_dtype
        self._original_version = __version__

    def index(self, corpus_embeddings: np.ndarray, corpus: List[Any] = None):
        if not isinstance(corpus_embeddings, np.ndarray):
            raise ValueError("corpus_embeddings must be a numpy array")
        self.corpus_embeddings = corpus_embeddings.astype(self.dtype)
        self.corpus = corpus
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        self.corpus_embeddings_normalized = self.corpus_embeddings / (norms + 1e-10)

    def _retrieve_from_embeddings(self, query_embeddings, k=10, sorted=True):
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_norm = query_embeddings / (query_norms + 1e-10)
        scores = np.dot(query_norm, self.corpus_embeddings_normalized.T)
        indices = np.argsort(-scores, axis=1)[:, :k]
        scores = np.take_along_axis(scores, indices, axis=1)
        return scores, indices

    def retrieve(
        self,
        queries: Union[List[str], np.ndarray],
        query_embeddings: np.ndarray = None,
        corpus: List[Any] = None,
        k: int = 10,
        sorted: bool = True,
        return_as: str = "tuple",
        show_progress: bool = True,
        leave_progress: bool = False,
    ):
        num_docs = len(self.corpus_embeddings) if self.corpus_embeddings is not None else 0
        if k > num_docs:
            raise ValueError(f"k={k} > num_docs={num_docs}")

        if query_embeddings is None:
            if self.embedding_fn is None:
                raise ValueError("Provide query_embeddings or set embedding_fn.")
            query_embeddings = self.embedding_fn(queries)

        query_embeddings = np.asarray(query_embeddings, dtype=self.dtype)
        scores, indices = self._retrieve_from_embeddings(query_embeddings, k=k, sorted=sorted)

        corpus = corpus if corpus is not None else self.corpus
        if corpus is None:
            retrieved_docs = indices
        elif isinstance(corpus, np.ndarray) and corpus.ndim == 1:
            retrieved_docs = corpus[indices]
        else:
            retrieved_docs = np.array([[corpus[i] for i in idx] for idx in indices])

        if return_as == "tuple":
            return Results(documents=retrieved_docs, scores=scores)
        elif return_as == "documents":
            return retrieved_docs
        else:
            raise ValueError("`return_as` must be 'tuple' or 'documents'")

    def save(self, save_dir: str, corpus: List[Any] = None,
             embeddings_name: str = "embeddings.npy", corpus_name: str = "corpus.jsonl"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / embeddings_name, self.corpus_embeddings)
        corpus = corpus if corpus is not None else self.corpus
        if corpus is not None:
            with open(save_dir / corpus_name, "wt", encoding="utf-8") as f:
                for i, doc in enumerate(corpus):
                    if isinstance(doc, str):
                        doc = {"id": i, "text": doc}
                    try:
                        f.write(json_functions.dumps(doc, ensure_ascii=False) + "\n")
                    except Exception as e:
                        logger.warning(f"Error saving doc {i}: {e}")

    @classmethod
    def load(cls, save_dir: str, embeddings_name: str = "embeddings.npy",
             corpus_name: str = "corpus.jsonl", load_corpus: bool = False,
             embedding_fn=None, dtype: str = "float32", int_dtype: str = "int32") -> "DPR":
        save_dir = Path(save_dir)
        embeddings = np.load(save_dir / embeddings_name)
        corpus = None
        if load_corpus:
            corpus_file = save_dir / corpus_name
            if corpus_file.exists():
                with open(corpus_file, "r", encoding="utf-8") as f:
                    corpus = [json_functions.loads(line) for line in f]
        obj = cls(corpus_embeddings=embeddings, corpus=corpus,
                  embedding_fn=embedding_fn, dtype=dtype, int_dtype=int_dtype)
        norms = np.linalg.norm(obj.corpus_embeddings, axis=1, keepdims=True)
        obj.corpus_embeddings_normalized = obj.corpus_embeddings / (norms + 1e-10)
        return obj