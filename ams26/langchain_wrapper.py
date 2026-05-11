"""LangChain retriever wrappers for bm25s.

This module provides lightweight LangChain-compatible retrievers for the
existing BM25, DPR, and Hybrid classes in this repo.

To install:
    pip install langchain

Basic usage examples:

    from bm25s import BM25, DPR, Hybrid
    from bm25s.langchain_wrapper import BM25LangChainRetriever

    # BM25
    bm25 = BM25()
    bm25.index([['hello', 'world'], ['goodbye', 'world']])
    bm25_retriever = BM25LangChainRetriever(bm25, corpus=['hello world', 'goodbye world'])
    docs = bm25_retriever.get_relevant_documents('hello')

    # DPR
    dpr = DPR(embedding_fn=my_embedding_fn)
    dpr.index(corpus_embeddings=my_embeddings, corpus=['hello world', 'goodbye world'])
    dpr_retriever = DPRLangChainRetriever(dpr, corpus=['hello world', 'goodbye world'])
    docs = dpr_retriever.get_relevant_documents('hello')

    # Hybrid
    hybrid = Hybrid(bm25=bm25, dpr=dpr)
    hybrid_retriever = HybridLangChainRetriever(hybrid, corpus=['hello world', 'goodbye world'])
    docs = hybrid_retriever.get_relevant_documents('hello')
"""

import json
from typing import Any, Callable, Iterable, List, Optional, Sequence
from xml.dom.minidom import Document
from typing import TYPE_CHECKING

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:
    raise ImportError(
        "langchain is required to use bm25s.langchain_wrapper. "
        "Install it with `pip install langchain`."
    ) from exc

from ._base import tokenization

if TYPE_CHECKING:
    from . import BM25, DPR, Hybrid


def _make_document(doc: Any) -> Document:
    if isinstance(doc, Document):
        return doc

    if isinstance(doc, str):
        return Document(page_content=doc)

    if isinstance(doc, dict):
        text = doc.get("text") or doc.get("content") or json.dumps(doc, ensure_ascii=False)
        metadata = {k: v for k, v in doc.items() if k != "text" and k != "content"}
        return Document(page_content=text, metadata=metadata)

    return Document(page_content=str(doc))


def _flatten_documents(retrieved: Any) -> List[Document]:
    docs: List[Document] = []
    if retrieved is None:
        return docs
    if isinstance(retrieved, list):
        for item in retrieved:
            if isinstance(item, (list, tuple)):
                docs.extend(_flatten_documents(item))
            else:
                docs.append(_make_document(item))
        return docs

    if hasattr(retrieved, "flatten"):
        return _flatten_documents(list(retrieved))

    return [_make_document(retrieved)]


class BM25LangChainRetriever(BaseRetriever):
    """LangChain wrapper around bm25s.BM25."""

    def __init__(
        self,
        corpus: Sequence[Any],
        k: int = 10,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        from . import BM25
        self.bm25 = bm25
        self.corpus = corpus
        self.k = k
        self.tokenizer = tokenizer or tokenization.tokenize

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_tokens = self.tokenizer(query)
        results = self.bm25.retrieve(
            [query_tokens],
            corpus=self.corpus,
            k=self.k,
            return_as="documents",
            show_progress=False,
        )

        if isinstance(results, list) or hasattr(results, "shape"):
            results = results[0]
        return _flatten_documents(results)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


class DPRLangChainRetriever(BaseRetriever):
    """LangChain wrapper around bm25s.DPR."""

    def __init__(
        self,
       
        corpus: Sequence[Any],
        k: int = 10,
        embedding_fn: Optional[Callable[[Sequence[str]], Any]] = None,
    ):
        from . import DPR
        self.dpr = dpr
        self.corpus = corpus
        self.k = k
        self.embedding_fn = embedding_fn or getattr(dpr, "embedding_fn", None)

        if self.embedding_fn is None:
            raise ValueError(
                "DPRLangChainRetriever requires an embedding_fn or a DPR object with embedding_fn set."
            )

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embedding_fn([query])
        results = self.dpr.retrieve(
            queries=None,
            query_embeddings=query_embedding,
            corpus=self.corpus,
            k=self.k,
            return_as="documents",
            show_progress=False,
        )

        if isinstance(results, list) or hasattr(results, "shape"):
            results = results[0]
        return _flatten_documents(results)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


class HybridLangChainRetriever(BaseRetriever):
    """LangChain wrapper around bm25s.Hybrid."""

    def __init__(
        self,
        corpus: Sequence[Any],
        k: int = 10,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        embedding_fn: Optional[Callable[[Sequence[str]], Any]] = None,
    ):
        from . import HYBRID
        self.hybrid = hybrid
        self.corpus = corpus
        self.k = k
        self.tokenizer = tokenizer or tokenization.tokenize
        self.embedding_fn = embedding_fn or getattr(hybrid.dpr, "embedding_fn", None)

        if self.hybrid.bm25 is None or self.hybrid.dpr is None:
            raise ValueError("HybridLangChainRetriever requires a Hybrid object with both bm25 and dpr indexed.")

        if self.embedding_fn is None:
            raise ValueError(
                "HybridLangChainRetriever requires an embedding_fn or a Hybrid object with dpr.embedding_fn set."
            )

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_tokens = self.tokenizer(query)
        query_embedding = self.embedding_fn([query])

        results = self.hybrid.retrieve(
            query_tokens=[query_tokens],
            queries=[query],
            query_embeddings=query_embedding,
            corpus=self.corpus,
            k=self.k,
            return_as="documents",
            show_progress=False,
        )

        if isinstance(results, list) or hasattr(results, "shape"):
            results = results[0]
        return _flatten_documents(results)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


__all__ = [
    "BM25LangChainRetriever",
    "DPRLangChainRetriever",
    "HybridLangChainRetriever",
]
