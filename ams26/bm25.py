"""BM25 sparse retrieval."""
import os
import logging
import json
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Tuple, Iterable, List, Union
import numpy as np

from ._base import (
    tqdm, __version__, tokenization, utils,
    NUMBA_AVAILABLE, SCIPY_AVAILABLE,
    sp, njit, selection_jit, selection_np,
    _retrieve_numba_functional,
    _select_tfc_scorer, _select_idf_scorer,
    _build_scores_and_indices_for_matrix,
    _calculate_doc_freqs, _build_idf_array,
    _build_nonoccurrence_array,
    _np_csc_python, _np_csc_jit_ready,
    json_functions,
)
from .results import Results, get_unique_tokens, is_list_of_list_of_type, _is_tuple_of_list_of_tokens

logger = logging.getLogger("bm25s")
logger.setLevel(logging.DEBUG)


class BM25:
    def __init__(
        self,
        k1=1.5,
        b=0.75,
        delta=0.5,
        method="lucene",
        idf_method=None,
        dtype="float32",
        int_dtype="int32",
        corpus=None,
        backend="numpy",
        csc_backend="numpy",
        auto_compile=True,
    ):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.dtype = dtype
        self.int_dtype = int_dtype
        self.method = method
        self.idf_method = idf_method if idf_method is not None else method
        self.methods_requiring_nonoccurrence = ("bm25l", "bm25+")
        self.corpus = corpus
        self._original_version = __version__
        self.csc_backend = csc_backend

        if backend == "auto":
            self.backend = "numba" if NUMBA_AVAILABLE else "numpy"
        else:
            self.backend = backend

        if csc_backend == "auto":
            self.csc_backend = "scipy" if SCIPY_AVAILABLE else "numpy"

        if self.backend == "numba" and not NUMBA_AVAILABLE:
            raise ImportError("Numba is not installed. pip install numba")

        if csc_backend == "scipy" and not SCIPY_AVAILABLE:
            raise ImportError("scipy is not installed.")

        NUMBA_IS_DISABLED = os.environ.get("NUMBA_DISABLE_JIT") in [None, False]
        if auto_compile and self.backend == "numba" and not NUMBA_IS_DISABLED:
            self.compile(activate_numba=True, warmup=False)

    @staticmethod
    def _infer_corpus_object(corpus):
        if hasattr(corpus, "ids") and hasattr(corpus, "vocab"):
            return "object"
        elif isinstance(corpus, tuple) and len(corpus) == 2:
            c1, c2 = corpus
            if isinstance(c1, list) and isinstance(c2, dict):
                return "tuple"
            else:
                raise ValueError("Invalid corpus tuple format.")
        elif isinstance(corpus, Iterable):
            if is_list_of_list_of_type(corpus, type_=int):
                return "token_ids"
            else:
                return "tokens"
        else:
            raise ValueError("Unrecognised corpus format.")

    @staticmethod
    def _compute_relevance_from_scores(
        data, indptr, indices, num_docs, query_tokens_ids, dtype
    ):
        indptr_starts = indptr[query_tokens_ids]
        indptr_ends = indptr[query_tokens_ids + 1]
        scores = np.zeros(num_docs, dtype=dtype)
        for i in range(len(query_tokens_ids)):
            start, end = indptr_starts[i], indptr_ends[i]
            np.add.at(scores, indices[start:end], data[start:end])
        return scores

    def build_index_from_ids(self, unique_token_ids, corpus_token_ids,
                              show_progress=True, leave_progress=False):
        if self.csc_backend not in ["scipy", "numpy"]:
            raise ValueError(f"Invalid csc_backend: {self.csc_backend}")

        avg_doc_len = np.array([len(d) for d in corpus_token_ids]).mean()
        n_docs = len(corpus_token_ids)
        n_vocab = len(unique_token_ids)

        doc_frequencies = _calculate_doc_freqs(
            corpus_tokens=corpus_token_ids,
            unique_tokens=unique_token_ids,
            show_progress=show_progress,
            leave_progress=leave_progress,
        )

        if self.method in self.methods_requiring_nonoccurrence:
            self.nonoccurrence_array = _build_nonoccurrence_array(
                doc_frequencies=doc_frequencies, n_docs=n_docs,
                compute_idf_fn=_select_idf_scorer(self.idf_method),
                calculate_tfc_fn=_select_tfc_scorer(self.method),
                l_d=avg_doc_len, l_avg=avg_doc_len,
                k1=self.k1, b=self.b, delta=self.delta, dtype=self.dtype,
            )
        else:
            self.nonoccurrence_array = None

        idf_array = _build_idf_array(
            doc_frequencies=doc_frequencies, n_docs=n_docs,
            compute_idf_fn=_select_idf_scorer(self.idf_method), dtype=self.dtype,
        )

        scores_flat, doc_idx, vocab_idx = _build_scores_and_indices_for_matrix(
            corpus_token_ids=corpus_token_ids, idf_array=idf_array,
            avg_doc_len=avg_doc_len, doc_frequencies=doc_frequencies,
            k1=self.k1, b=self.b, delta=self.delta,
            show_progress=show_progress, leave_progress=leave_progress,
            dtype=self.dtype, int_dtype=self.int_dtype,
            method=self.method, nonoccurrence_array=self.nonoccurrence_array,
        )

        if self.csc_backend == "scipy":
            score_matrix = sp.csc_matrix(
                (scores_flat, (doc_idx, vocab_idx)),
                shape=(n_docs, n_vocab), dtype=self.dtype,
            )
            data, indices, indptr = score_matrix.data, score_matrix.indices, score_matrix.indptr
        else:
            data, indices, indptr = self._np_csc(
                data=scores_flat, rows=doc_idx, cols=vocab_idx, shape=(n_docs, n_vocab)
            )
            data = data.astype(self.dtype)

        return {"data": data, "indices": indices, "indptr": indptr, "num_docs": n_docs}

    def build_index_from_tokens(self, corpus_tokens, show_progress=True, leave_progress=False):
        unique_tokens = get_unique_tokens(
            corpus_tokens, show_progress=show_progress,
            leave_progress=leave_progress, desc="BM25S Create Vocab",
        )
        vocab_dict = {token: i for i, token in enumerate(unique_tokens)}
        unique_token_ids = [vocab_dict[token] for token in unique_tokens]
        corpus_token_ids = [
            [vocab_dict[token] for token in tokens]
            for tokens in tqdm(corpus_tokens, desc="BM25S Convert tokens to indices",
                               leave=leave_progress, disable=not show_progress)
        ]
        scores = self.build_index_from_ids(
            unique_token_ids=unique_token_ids, corpus_token_ids=corpus_token_ids,
            show_progress=show_progress, leave_progress=leave_progress,
        )
        return scores, vocab_dict

    def index(self, corpus, create_empty_token=True, show_progress=True, leave_progress=False):
        inferred = self._infer_corpus_object(corpus)

        if inferred == "tokens":
            scores, vocab_dict = self.build_index_from_tokens(
                corpus, leave_progress=leave_progress, show_progress=show_progress
            )
        else:
            if inferred == "tuple":
                corpus_token_ids, vocab_dict = corpus
            elif inferred == "object":
                corpus_token_ids, vocab_dict = corpus.ids, corpus.vocab
            elif inferred == "token_ids":
                corpus_token_ids = corpus
                unique_ids = set()
                for doc_ids in corpus_token_ids:
                    unique_ids.update(doc_ids)
                if create_empty_token:
                    unique_ids.add(0 if 0 not in unique_ids else max(unique_ids) + 1)
                vocab_dict = {token_id: i for i, token_id in enumerate(unique_ids)}
            else:
                raise ValueError("Internal error: invalid corpus object.")

            unique_token_ids = list(vocab_dict.values())
            scores = self.build_index_from_ids(
                unique_token_ids=unique_token_ids, corpus_token_ids=corpus_token_ids,
                leave_progress=leave_progress, show_progress=show_progress,
            )

        if create_empty_token and inferred != "token_ids" and "" not in vocab_dict:
            vocab_dict[""] = max(vocab_dict.values()) + 1

        self.scores = scores
        self.vocab_dict = vocab_dict
        self.unique_token_ids_set = set(self.vocab_dict.values())

    def get_tokens_ids(self, query_tokens):
        return [self.vocab_dict[t] for t in query_tokens if t in self.vocab_dict]

    def get_scores_from_ids(self, query_tokens_ids, weight_mask=None):
        data = self.scores["data"]
        indices = self.scores["indices"]
        indptr = self.scores["indptr"]
        num_docs = self.scores["num_docs"]
        int_dtype = np.dtype(self.int_dtype)
        query_tokens_ids = np.asarray(query_tokens_ids, dtype=int_dtype)

        if int(query_tokens_ids.max(initial=0)) >= len(indptr) - 1:
            raise ValueError("Token ID exceeds index size.")

        scores = self._compute_relevance_from_scores(
            data=data, indptr=indptr, indices=indices,
            num_docs=num_docs, query_tokens_ids=query_tokens_ids,
            dtype=np.dtype(self.dtype),
        )
        if weight_mask is not None:
            scores *= weight_mask
        if self.nonoccurrence_array is not None:
            scores += self.nonoccurrence_array[query_tokens_ids].sum()
        return scores

    def get_scores(self, query_tokens_single, weight_mask=None):
        if not isinstance(query_tokens_single, list):
            raise ValueError("query_tokens must be a list.")
        if isinstance(query_tokens_single[0], str):
            query_tokens_ids = self.get_tokens_ids(query_tokens_single)
        elif isinstance(query_tokens_single[0], int):
            query_tokens_ids = query_tokens_single
        else:
            raise ValueError("query_tokens must be strings or ints.")
        return self.get_scores_from_ids(query_tokens_ids, weight_mask=weight_mask)

    def _get_top_k_results(self, query_tokens_single, k=1000, backend="auto",
                            sorted=False, weight_mask=None):
        if len(query_tokens_single) == 0:
            scores_q = np.zeros(self.scores["num_docs"], dtype=self.dtype)
        else:
            scores_q = self.get_scores(query_tokens_single, weight_mask=weight_mask)

        if backend.startswith("numba"):
            if not NUMBA_AVAILABLE:
                raise ImportError("Numba not installed.")
            return selection_jit.topk(scores_q, k=k, sorted=sorted, backend=backend)
        return selection_np.topk(scores_q, k=k, sorted=sorted, backend=backend)

    def retrieve(self, query_tokens, corpus=None, k=10, sorted=True, return_as="tuple",
                 show_progress=True, leave_progress=False, n_threads=0, chunksize=50,
                 backend_selection="auto", weight_mask=None):
        num_docs = self.scores["num_docs"]
        if k > num_docs:
            raise ValueError(f"k={k} > num_docs={num_docs}")
        if return_as not in ["tuple", "documents"]:
            raise ValueError("`return_as` must be 'tuple' or 'documents'")
        if n_threads == -1:
            n_threads = os.cpu_count()

        if is_list_of_list_of_type(query_tokens, type_=int):
            query_tokens_filtered = []
            for query in query_tokens:
                filtered = [t for t in query if t in self.unique_token_ids_set]
                if not filtered:
                    if "" not in self.vocab_dict:
                        raise ValueError("Empty query and no empty token in vocab.")
                    filtered = [self.vocab_dict[""]]
                query_tokens_filtered.append(filtered)
            query_tokens = query_tokens_filtered

        if isinstance(query_tokens, tuple) and not _is_tuple_of_list_of_tokens(query_tokens):
            if len(query_tokens) != 2:
                raise ValueError("Expected tuple of (ids, vocab).")
            ids, vocab = query_tokens
            query_tokens = tokenization.Tokenized(ids=ids, vocab=vocab)

        if isinstance(query_tokens, tokenization.Tokenized):
            query_tokens = tokenization.convert_tokenized_to_string_list(query_tokens)

        corpus = corpus if corpus is not None else self.corpus

        if weight_mask is not None:
            if not isinstance(weight_mask, np.ndarray) or weight_mask.ndim != 1:
                raise ValueError("weight_mask must be a 1D numpy array.")
            if len(weight_mask) != num_docs:
                raise ValueError("weight_mask length must match corpus size.")

        if self.backend == "numba":
            if _retrieve_numba_functional is None:
                raise ImportError("Numba not installed.")
            backend_selection = "numba" if backend_selection == "auto" else backend_selection
            if is_list_of_list_of_type(query_tokens, type_=int):
                query_tokens_ids = query_tokens
            elif is_list_of_list_of_type(query_tokens, type_=str):
                query_tokens_ids = [self.get_tokens_ids(q) for q in query_tokens]
            else:
                raise ValueError("query_tokens must be list of str or int lists.")
            res = _retrieve_numba_functional(
                query_tokens_ids=query_tokens_ids, scores=self.scores, corpus=corpus,
                k=k, sorted=sorted, return_as=return_as,
                show_progress=show_progress, leave_progress=leave_progress,
                n_threads=n_threads, chunksize=None,
                backend_selection=backend_selection,
                dtype=self.dtype, int_dtype=self.int_dtype,
                nonoccurrence_array=self.nonoccurrence_array, weight_mask=weight_mask,
            )
            return Results(documents=res[0], scores=res[1]) if return_as == "tuple" else res

        topk_fn = partial(self._get_top_k_results, k=k, sorted=sorted,
                          backend=backend_selection, weight_mask=weight_mask)
        tqdm_kwargs = dict(total=len(query_tokens), desc="BM25S Retrieve",
                           leave=leave_progress, disable=not show_progress)

        if n_threads == 0:
            out = list(tqdm(map(topk_fn, query_tokens), **tqdm_kwargs))
        else:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                out = list(tqdm(ex.map(topk_fn, query_tokens, chunksize=chunksize), **tqdm_kwargs))

        scores_arr, indices = zip(*out)
        scores_arr, indices = np.array(scores_arr), np.array(indices)
        corpus = corpus if corpus is not None else self.corpus

        if corpus is None:
            retrieved_docs = indices
        elif isinstance(corpus, utils.corpus.JsonlCorpus):
            retrieved_docs = corpus[indices]
        elif isinstance(corpus, np.ndarray) and corpus.ndim == 1:
            retrieved_docs = corpus[indices]
        else:
            retrieved_docs = np.array([corpus[i] for i in indices.flatten().tolist()]).reshape(indices.shape)

        return Results(documents=retrieved_docs, scores=scores_arr) if return_as == "tuple" else retrieved_docs

    def save(self, save_dir, corpus=None, data_name="data.csc.index.npy",
             indices_name="indices.csc.index.npy", indptr_name="indptr.csc.index.npy",
             vocab_name="vocab.index.json", params_name="params.index.json",
             nnoc_name="nonoccurrence_array.index.npy", corpus_name="corpus.jsonl",
             allow_pickle=False, show_progress=True):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / data_name, self.scores["data"], allow_pickle=allow_pickle)
        np.save(save_dir / indices_name, self.scores["indices"], allow_pickle=allow_pickle)
        np.save(save_dir / indptr_name, self.scores["indptr"], allow_pickle=allow_pickle)
        if self.nonoccurrence_array is not None:
            np.save(save_dir / nnoc_name, self.nonoccurrence_array, allow_pickle=allow_pickle)
        with open(save_dir / vocab_name, "wt", encoding="utf-8") as f:
            f.write(json_functions.dumps(self.vocab_dict, ensure_ascii=False))
        params = dict(k1=self.k1, b=self.b, delta=self.delta, method=self.method,
                      idf_method=self.idf_method, dtype=self.dtype, int_dtype=self.int_dtype,
                      num_docs=self.scores["num_docs"], version=__version__, backend=self.backend)
        with open(save_dir / params_name, "w") as f:
            json.dump(params, f, indent=4)
        corpus = corpus if corpus is not None else self.corpus
        if corpus is not None:
            import logging as _logging
            with open(save_dir / corpus_name, "wt", encoding="utf-8") as f:
                for i, doc in enumerate(corpus):
                    if isinstance(doc, str):
                        doc = {"id": i, "text": doc}
                    elif not isinstance(doc, (dict, list, tuple)):
                        _logging.warning(f"Skipping doc at {i}: unsupported type.")
                        continue
                    try:
                        f.write(json_functions.dumps(doc, ensure_ascii=False) + "\n")
                    except Exception as e:
                        _logging.warning(f"Error saving doc {i}: {e}")
            mmidx = utils.corpus.find_newline_positions(save_dir / corpus_name, show_progress=show_progress)
            utils.corpus.save_mmindex(mmidx, path=save_dir / corpus_name)

    def load_scores(self, save_dir, data_name="data.csc.index.npy",
                    indices_name="indices.csc.index.npy", indptr_name="indptr.csc.index.npy",
                    num_docs=None, mmap=False, allow_pickle=False):
        save_dir = Path(save_dir)
        mmap_mode = "r" if mmap else None
        self.scores = {
            "data": np.load(save_dir / data_name, allow_pickle=allow_pickle, mmap_mode=mmap_mode),
            "indices": np.load(save_dir / indices_name, allow_pickle=allow_pickle, mmap_mode=mmap_mode),
            "indptr": np.load(save_dir / indptr_name, allow_pickle=allow_pickle, mmap_mode=mmap_mode),
            "num_docs": num_docs,
        }

    @classmethod
    def load(cls, save_dir, data_name="data.csc.index.npy", indices_name="indices.csc.index.npy",
             indptr_name="indptr.csc.index.npy", vocab_name="vocab.index.json",
             params_name="params.index.json", nnoc_name="nonoccurrence_array.index.npy",
             corpus_name="corpus.jsonl", load_corpus=False, mmap=False, allow_pickle=False,
             load_vocab=True, override_params=None, **kwargs):
        if not isinstance(mmap, bool):
            raise ValueError("`mmap` must be a boolean")
        save_dir = Path(save_dir)
        with open(save_dir / params_name, "r") as f:
            params = json_functions.loads(f.read())
        if override_params:
            params.update(override_params)
        if kwargs:
            params.update(kwargs)
        vocab_dict = {}
        if load_vocab:
            with open(save_dir / vocab_name, "r", encoding="utf-8") as f:
                vocab_dict = json_functions.loads(f.read())
        original_version = params.pop("version", None)
        num_docs = params.pop("num_docs", None)
        obj = cls(**params)
        obj.vocab_dict = vocab_dict
        obj._original_version = original_version
        obj.unique_token_ids_set = set(obj.vocab_dict.values())
        obj.load_scores(save_dir=save_dir, data_name=data_name, indices_name=indices_name,
                        indptr_name=indptr_name, mmap=mmap, num_docs=num_docs, allow_pickle=allow_pickle)
        if load_corpus:
            corpus_file = save_dir / corpus_name
            if corpus_file.exists():
                if mmap:
                    obj.corpus = utils.corpus.JsonlCorpus(corpus_file)
                else:
                    with open(corpus_file, "r", encoding="utf-8") as f:
                        obj.corpus = [json_functions.loads(line) for line in f]
        if obj.method in obj.methods_requiring_nonoccurrence:
            nnm_path = save_dir / nnoc_name
            if nnm_path.exists():
                obj.nonoccurrence_array = np.load(nnm_path, allow_pickle=allow_pickle)
            else:
                raise FileNotFoundError(f"Non-occurrence array not found at {nnm_path}")
        else:
            obj.nonoccurrence_array = None
        return obj

    @staticmethod
    def _np_csc(data, rows, cols, shape):
        return _np_csc_python(data, rows, cols, shape)

    def compile(self, activate_numba=True, warmup=False):
        if not NUMBA_AVAILABLE:
            raise ImportError("Numba not installed.")
        if activate_numba:
            self.activate_numba_csc()
            self.activate_numba_scorer()
        if warmup:
            self.warmup_numba_csc()
            self.warmup_numba_scorer()

    def activate_numba_scorer(self):
        try:
            from numba import njit
        except ImportError:
            raise ImportError("Numba not installed.")
        if os.environ.get("NUMBA_DISABLE_JIT"):
            return
        from .scoring import _compute_relevance_from_scores_jit_ready
        self._compute_relevance_from_scores = njit(_compute_relevance_from_scores_jit_ready)

    def activate_numba_csc(self):
        try:
            from numba import njit
        except ImportError:
            raise ImportError("Numba not installed.")
        self._np_csc = njit(_np_csc_jit_ready)

    def warmup_numba_scorer(self):
        if os.environ.get("NUMBA_DISABLE_JIT") or not NUMBA_AVAILABLE:
            return
        self._compute_relevance_from_scores(
            data=np.array([1., 2., 3.], dtype=self.dtype),
            indptr=np.array([0, 3], dtype=self.int_dtype),
            indices=np.array([0, 1, 2], dtype=self.int_dtype),
            num_docs=1,
            query_tokens_ids=np.array([0, 1, 2], dtype=self.int_dtype),
            dtype=np.dtype(self.dtype),
        )

    def warmup_numba_csc(self):
        self._np_csc(
            data=np.array([1., 2., 3.], dtype=self.dtype),
            rows=np.array([0, 1, 2], dtype=self.int_dtype),
            cols=np.array([0, 0, 0], dtype=self.int_dtype),
            shape=(3, 1),
        )