"""Shared imports, constants, and utilities used across ams26 modules."""
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np


__version__ = "0.1.0"
json_functions = json


# # --- Numba ---
# try:
#     from numba import njit
#     from .numba import selection as selection_jit
#     NUMBA_AVAILABLE = True
# except ImportError:
#     njit = lambda x: x  # type: ignore
#     selection_jit = None
#     NUMBA_AVAILABLE = False

# # --- Scipy ---



# try:
#     import scipy.sparse as sp
#     SCIPY_AVAILABLE = True
# except ImportError:
#     sp = None
#     SCIPY_AVAILABLE = False

# # --- Numba retrieve utils ---
# try:
#     from .numba.retrieve_utils import _retrieve_numba_functional
# except ImportError:
#     _retrieve_numba_functional = None


# # --- tqdm ---
# def _faketqdm(*args, **kwargs):
#     return args[0] if len(args) > 0 else None


# if os.environ.get("DISABLE_TQDM", False):
#     tqdm = _faketqdm
# else:
#     try:
#         from tqdm.auto import tqdm
#     except ImportError:
#         tqdm = _faketqdm


# --- holds the tokenized text with respective token ids and vocabulary per document ---
@dataclass
class Tokenized:
    ids: list
    vocab: dict

# converts the text to a list of tokens, by first converting the text to lowercase and then splitting the text into a list of tokens
def tokenize(text: str):
    return text.lower().split()


def convert_tokenized_to_string_list(tokenized: Tokenized):
    inv_vocab = {idx: token for token, idx in tokenized.vocab.items()}
    return [[inv_vocab.get(tok_id, "") for tok_id in row] for row in tokenized.ids]


tokenization = SimpleNamespace(
    Tokenized=Tokenized,
    tokenize=tokenize,
    convert_tokenized_to_string_list=convert_tokenized_to_string_list,
)


# --- selection of top k documents based on the scores ---
def _topk(scores: np.ndarray, k: int = 10, sorted: bool = False, backend: str = "auto"):

    # if k is less than or equal to 0, return an empty array
    if k <= 0:
        return np.array([], dtype=scores.dtype), np.array([], dtype=np.int32)
    # if k is greater than the number of scores, set k to the number of scores
    k = min(k, len(scores))
    # partition the scores into k groups, and get the indices of the top k scores
    idx = np.argpartition(-scores, k - 1)[:k]
    # get the top k scores
    top_scores = scores[idx]
    if sorted:
        order = np.argsort(-top_scores)
        idx = idx[order]
        top_scores = top_scores[order]
    return top_scores, idx

selection_np = SimpleNamespace(topk=_topk)

# --- stopwords ---
stopwords = SimpleNamespace(DEFAULT_STOPWORDS=set())

# --- utils.corpus ---
class JsonlCorpus:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            self._rows = [json.loads(line) for line in f]

    def __getitem__(self, indices):
        arr = np.asarray(indices)
        if arr.ndim == 0:
            return self._rows[int(arr)]
        flat = [self._rows[int(i)] for i in arr.flatten().tolist()]
        return np.array(flat, dtype=object).reshape(arr.shape)

    def __len__(self):
        return len(self._rows)


def _find_newline_positions(path, show_progress=True):
    positions = [0]
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            positions.append(f.tell())
    return positions


def _save_mmindex(mmidx, path):
    np.save(Path(str(path) + ".mmidx.npy"), np.asarray(list(mmidx), dtype=np.int64))


utils = SimpleNamespace(
    json_functions=json_functions,
    corpus=SimpleNamespace(
        JsonlCorpus=JsonlCorpus,
        find_newline_positions=_find_newline_positions,
        save_mmindex=_save_mmindex,
    ),
)


# --- scoring ---
def _select_tfc_scorer(method):
    method = (method or "lucene").lower()
    if method in ("lucene", "bm25"):
        return _tfc_bm25
    if method == "bm25+":
        return _tfc_bm25_plus
    if method == "bm25l":
        return _tfc_bm25l
    return _tfc_bm25


def _select_idf_scorer(method):
    return _idf_lucene


def _idf_lucene(df, n_docs):
    return np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def _tfc_bm25(tf, l_d, l_avg, k1, b, delta):
    denom = tf + k1 * (1.0 - b + b * (l_d / l_avg))
    return tf * (k1 + 1.0) / (denom + 1e-12)


def _tfc_bm25_plus(tf, l_d, l_avg, k1, b, delta):
    return _tfc_bm25(tf, l_d, l_avg, k1, b, delta) + delta


def _tfc_bm25l(tf, l_d, l_avg, k1, b, delta):
    ctd = tf / (1.0 - b + b * (l_d / l_avg) + 1e-12)
    return ((k1 + 1.0) * (ctd + delta)) / (k1 + ctd + delta + 1e-12)


def _calculate_doc_freqs(corpus_tokens, unique_tokens, show_progress=True, leave_progress=False):
    doc_freqs = np.zeros(len(unique_tokens), dtype=np.int32)
    for doc in corpus_tokens:
        for tok in set(doc):
            if tok < len(doc_freqs):
                doc_freqs[tok] += 1
    return doc_freqs


def _build_idf_array(doc_frequencies, n_docs, compute_idf_fn, dtype="float32"):
    return compute_idf_fn(doc_frequencies.astype(np.float64), float(n_docs)).astype(dtype)


def _build_nonoccurrence_array(
    doc_frequencies, n_docs, compute_idf_fn, calculate_tfc_fn, l_d, l_avg, k1, b, delta, dtype="float32"
):
    idf = compute_idf_fn(doc_frequencies.astype(np.float64), float(n_docs))
    tfc_zero = calculate_tfc_fn(0.0, l_d=l_d, l_avg=l_avg, k1=k1, b=b, delta=delta)
    return (idf * tfc_zero).astype(dtype)


def _build_scores_and_indices_for_matrix(
    corpus_token_ids,
    idf_array,
    avg_doc_len,
    doc_frequencies,
    k1,
    b,
    delta,
    show_progress=True,
    leave_progress=False,
    dtype="float32",
    int_dtype="int32",
    method="lucene",
    nonoccurrence_array=None,
):
    tfc_fn = _select_tfc_scorer(method)
    data = []
    rows = []
    cols = []
    for doc_id, token_ids in enumerate(corpus_token_ids):
        counts = Counter(token_ids)
        l_d = len(token_ids)
        for tok_id, tf in counts.items():
            tfc = tfc_fn(float(tf), l_d=l_d, l_avg=avg_doc_len, k1=k1, b=b, delta=delta)
            score = float(idf_array[tok_id]) * float(tfc)
            data.append(score)
            rows.append(doc_id)
            cols.append(tok_id)
    return np.asarray(data, dtype=dtype), np.asarray(rows, dtype=int_dtype), np.asarray(cols, dtype=int_dtype)


def _np_csc_python(data, rows, cols, shape):
    n_rows, n_cols = shape
    order = np.lexsort((rows, cols))
    rows_sorted = rows[order]
    cols_sorted = cols[order]
    data_sorted = data[order]
    indptr = np.zeros(n_cols + 1, dtype=np.int32)
    np.add.at(indptr, cols_sorted + 1, 1)
    indptr = np.cumsum(indptr, dtype=np.int32)
    return data_sorted, rows_sorted.astype(np.int32), indptr


def _np_csc_jit_ready(data, rows, cols, shape):
    return _np_csc_python(data, rows, cols, shape)


def _compute_relevance_from_scores_jit_ready(data, indptr, indices, num_docs, query_tokens_ids, dtype):
    indptr_starts = indptr[query_tokens_ids]
    indptr_ends = indptr[query_tokens_ids + 1]
    scores = np.zeros(num_docs, dtype=dtype)
    for i in range(len(query_tokens_ids)):
        start, end = indptr_starts[i], indptr_ends[i]
        np.add.at(scores, indices[start:end], data[start:end])
    return scores
 