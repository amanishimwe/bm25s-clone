"""Results NamedTuple and shared helper functions."""
from typing import List
import numpy as np
from ._base import tqdm


class Results:
    """
    NamedTuple with two fields: documents and scores.
    """
    __slots__ = ("documents", "scores")
    _fields = ("documents", "scores")

    def __init__(self, documents: np.ndarray, scores: np.ndarray):
        self.documents = documents
        self.scores = scores

    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        yield self.documents
        yield self.scores

    @classmethod
    def merge(cls, results: List["Results"]) -> "Results":
        documents = np.concatenate([r.documents for r in results], axis=0)
        scores = np.concatenate([r.scores for r in results], axis=0)
        return cls(documents=documents, scores=scores)


def get_unique_tokens(
    corpus_tokens, show_progress=True, leave_progress=False, desc="Create Vocab"
):
    unique_tokens = set()
    for doc_tokens in tqdm(
        corpus_tokens, desc=desc, disable=not show_progress, leave=leave_progress
    ):
        unique_tokens.update(doc_tokens)
    return unique_tokens


def is_list_of_list_of_type(obj, type_=int):
    if not isinstance(obj, list):
        return False
    if len(obj) == 0:
        return False
    first_elem = obj[0]
    if not isinstance(first_elem, list):
        return False
    if len(first_elem) == 0:
        return False
    return isinstance(first_elem[0], type_)


def _is_tuple_of_list_of_tokens(obj):
    if not isinstance(obj, tuple) or len(obj) == 0:
        return False
    first_elem = obj[0]
    if not isinstance(first_elem, list) or len(first_elem) == 0:
        return False
    return isinstance(first_elem[0], str)