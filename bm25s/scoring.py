"""Scoring functions for BM25."""

import numpy as np
from collections import Counter

def _select_tfc_scorer(method: str):
    """Select term frequency component scorer for different BM25 variants.
    
    Parameters
    ----------
    method : str
        BM25 variant: 'lucene', 'robertson', 'atire', 'bm25l', 'bm25+'
    
    Returns
    -------
    callable
        Function computing TFC (term frequency component)
    """
    if method == "lucene":
        return lambda tf, k1, l_d, l_avg, b: (k1 + 1) * tf / (k1 * (1 - b + b * (l_d / l_avg)) + tf)
    elif method == "robertson":
        return lambda tf, k1, l_d, l_avg, b: tf / (k1 * (1 - b + b * (l_d / l_avg)) + tf)
    elif method == "atire":
        return lambda tf, k1, l_d, l_avg, b: (k1 + 1) * tf / (k1 + tf)
    elif method == "bm25l":
        return lambda tf, k1, l_d, l_avg, b: (k1 + 1) * tf / (k1 * (1 - b + b * (l_d / l_avg)) + tf)
    elif method == "bm25+":
        return lambda tf, k1, l_d, l_avg, b: (k1 + 1) * tf / (k1 * (1 - b + b * (l_d / l_avg)) + tf)
    else:
        raise ValueError(f"Unknown method: {method}")

def _select_idf_scorer(method: str):
    """Select IDF scorer for different BM25 variants.
    
    Parameters
    ----------
    method : str
        BM25 variant: 'lucene', 'robertson', 'atire', 'bm25l', 'bm25+'
    
    Returns
    -------
    callable
        Function computing IDF (inverse document frequency)
    """
    if method in ["lucene", "robertson", "atire", "bm25l", "bm25+"]:
        return lambda df, n: np.log(1 + (n - df + 0.5) / (df + 0.5))
    else:
        raise ValueError(f"Unknown method: {method}")

def _calculate_doc_freqs(corpus_tokens, unique_tokens, show_progress=True, leave_progress=False):
    """Calculate document frequencies for each term.
    
    Parameters
    ----------
    corpus_tokens : List[List[int]]
        Tokenized corpus
    unique_tokens : List[int]
        List of unique token IDs
    show_progress : bool
        Whether to show progress
    leave_progress : bool
        Whether to leave progress bar
    
    Returns
    -------
    dict
        Document frequencies keyed by token ID
    """
    doc_freqs = {token: 0 for token in unique_tokens}
    for doc_tokens in corpus_tokens:
        seen = set(doc_tokens)
        for token in seen:
            if token in doc_freqs:
                doc_freqs[token] += 1
    return doc_freqs

def _build_idf_array(doc_frequencies, n_docs, compute_idf_fn, dtype="float32"):
    """Build IDF array indexed by token ID.
    
    Parameters
    ----------
    doc_frequencies : dict
        Document frequencies keyed by token ID
    n_docs : int
        Total number of documents
    compute_idf_fn : callable
        Function to compute IDF
    dtype : str
        Data type for output array
    
    Returns
    -------
    np.ndarray
        IDF scores indexed by token ID
    """
    max_token_id = max(doc_frequencies.keys()) if doc_frequencies else 0
    idf_array = np.zeros(max_token_id + 1, dtype=dtype)
    
    for token_id, df in doc_frequencies.items():
        idf_array[token_id] = compute_idf_fn(df, n_docs)
    
    return idf_array

def _build_nonoccurrence_array(doc_frequencies, n_docs, compute_idf_fn, calculate_tfc_fn, 
                                l_d, l_avg, k1, b, delta, dtype="float32"):
    """Build non-occurrence array for BM25L/BM25+ variants.
    
    Computes scores for terms that do NOT occur in a document,
    used by BM25L and BM25+ formulas.
    
    Parameters
    ----------
    doc_frequencies : dict
        Document frequencies keyed by token ID
    n_docs : int
        Total number of documents
    compute_idf_fn : callable
        Function to compute IDF
    calculate_tfc_fn : callable
        Function to compute term frequency component
    l_d : float
        Average document length
    l_avg : float
        Average document length (same as l_d)
    k1 : float
        BM25 k1 parameter
    b : float
        BM25 b parameter
    delta : float
        Delta parameter for BM25L/BM25+
    dtype : str
        Data type for output array
    
    Returns
    -------
    np.ndarray
        Non-occurrence scores indexed by token ID
    """
    max_token_id = max(doc_frequencies.keys()) if doc_frequencies else 0
    nonoccurrence_array = np.zeros(max_token_id + 1, dtype=dtype)
    
    for token_id, df in doc_frequencies.items():
        idf = compute_idf_fn(df, n_docs)
        # For non-occurrence (tf=0), compute TFC
        tfc = calculate_tfc_fn(0, k1, l_d, l_avg, b)
        # Add delta for BM25L/BM25+
        nonoccurrence_array[token_id] = (idf * tfc) + delta
    
    return nonoccurrence_array

def _build_scores_and_indices_for_matrix(corpus_token_ids, idf_array, avg_doc_len, 
                                          doc_frequencies, k1, b, delta, show_progress=True,
                                          leave_progress=False, dtype="float32", int_dtype="int32",
                                          method="lucene", nonoccurrence_array=None):
    """Build BM25 score matrix in coordinate format (COO).
    
    Computes BM25 scores for all term-document pairs and returns them
    in coordinate format suitable for sparse matrix construction.
    
    Parameters
    ----------
    corpus_token_ids : List[List[int]]
        Tokenized corpus with token IDs
    idf_array : np.ndarray
        IDF scores indexed by token ID
    avg_doc_len : float
        Average document length
    doc_frequencies : dict
        Document frequencies (used for context)
    k1 : float
        BM25 k1 parameter
    b : float
        BM25 b parameter
    delta : float
        Delta parameter for BM25L/BM25+
    show_progress : bool
        Whether to show progress bar
    leave_progress : bool
        Whether to leave progress bar after completion
    dtype : str
        Data type for scores
    int_dtype : str
        Data type for indices
    method : str
        BM25 variant: 'lucene', 'robertson', 'atire', 'bm25l', 'bm25+'
    nonoccurrence_array : np.ndarray, optional
        Non-occurrence array for BM25L/BM25+
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (scores, doc_indices, vocab_indices) for COO format matrix
    """
    scores_list = []
    doc_indices = []
    vocab_indices = []
    
    tfc_fn = _select_tfc_scorer(method)
    
    num_docs = len(corpus_token_ids)
    
    # Process each document
    for doc_id, doc_tokens in enumerate(corpus_token_ids):
        doc_len = len(doc_tokens)
        token_counts = Counter(doc_tokens)
        
        # Process each unique token in the document
        for token_id, count in token_counts.items():
            if token_id < len(idf_array):
                idf = idf_array[token_id]
                # Compute term frequency component
                tf = tfc_fn(count, k1, doc_len, avg_doc_len, b)
                # Final BM25 score
                score = idf * tf
                
                # Handle BM25L and BM25+ adjustments
                if method in ["bm25l", "bm25+"]:
                    # Add delta adjustment
                    score += delta
                
                scores_list.append(score)
                doc_indices.append(doc_id)
                vocab_indices.append(token_id)
    
    return (
        np.array(scores_list, dtype=dtype),
        np.array(doc_indices, dtype=int_dtype),
        np.array(vocab_indices, dtype=int_dtype),
    )

def _compute_relevance_from_scores_jit_ready(data, indptr, indices, num_docs, query_tokens_ids, dtype):
    """JIT-ready version of compute_relevance_from_scores.
    
    Computes final relevance scores for a query using pre-built sparse matrix.
    Iterates through query tokens and accumulates scores from documents containing them.
    
    Parameters
    ----------
    data : np.ndarray
        Data array from sparse CSC matrix
    indptr : np.ndarray
        Index pointer array from sparse CSC matrix
    indices : np.ndarray
        Row indices array from sparse CSC matrix
    num_docs : int
        Total number of documents
    query_tokens_ids : np.ndarray
        Token IDs in the query
    dtype : np.dtype
        Data type for output scores
    
    Returns
    -------
    np.ndarray
        Relevance scores for each document
    """
    scores = np.zeros(num_docs, dtype=dtype)
    
    for i in range(len(query_tokens_ids)):
        token_id = query_tokens_ids[i]
        
        # Safety check: token_id must be within valid range
        if token_id >= len(indptr) - 1:
            continue
            
        start = indptr[token_id]
        end = indptr[token_id + 1]
        
        # Accumulate scores for all documents containing this token
        for j in range(start, end):
            scores[indices[j]] += data[j]
    
    return scores

def _np_csc_python(data, rows, cols, shape):
    """Pure Python/NumPy CSC (Compressed Sparse Column) matrix builder.
    
    Converts coordinate format (COO) to compressed sparse column (CSC) format.
    CSC is efficient for column-slicing operations, which is used in BM25
    retrieval when looking up scores for query tokens.
    
    Parameters
    ----------
    data : np.ndarray
        Data values (scores)
    rows : np.ndarray
        Row indices (document IDs)
    cols : np.ndarray
        Column indices (token IDs)
    shape : Tuple[int, int]
        Shape of the output matrix (num_docs, num_vocab)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (data, indices, indptr) for CSC format
    """
    num_docs, num_vocab = shape
    
    # Sort by column (token ID) to build CSC format
    if len(cols) > 0:
        sorted_indices = np.argsort(cols, kind='stable')
        data = data[sorted_indices]
        rows = rows[sorted_indices]
        cols = cols[sorted_indices]
    
    # Build indptr (column pointer array)
    # indptr[i] points to the first element in column i
    indptr = np.zeros(num_vocab + 1, dtype=np.int32)
    
    if len(cols) > 0:
        for col in cols:
            indptr[col + 1] += 1
        indptr = np.cumsum(indptr)
    
    # rows becomes indices in CSC format
    indices = rows
    
    return data, indices, indptr

def _np_csc_jit_ready(data, rows, cols, shape):
    """JIT-ready CSC matrix builder (wraps _np_csc_python).
    
    This is the function that gets JIT-compiled for performance.
    """
    return _np_csc_python(data, rows, cols, shape)
