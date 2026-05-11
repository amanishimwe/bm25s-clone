"""Shared imports, constants, and utilities used across ams26 modules."""
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from functools import partial
from pathlib import Path
import json
from typing import Any, Tuple, Dict, Iterable, List, NamedTuple, Union
import numpy as np
from .utils import json_functions as json_functions
 
# --- Numba ---
try:
    from numba import njit
    from .numba import selection as selection_jit
    NUMBA_AVAILABLE = True
except ImportError:
    njit = lambda x: x  # type: ignore
    NUMBA_AVAILABLE = False
 
# --- Scipy ---
try:
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
 
# --- Numba retrieve utils ---
try:
    from .numba.retrieve_utils import _retrieve_numba_functional
except ImportError:
    _retrieve_numba_functional = None
 
# --- tqdm ---
def _faketqdm(*args, **kwargs):
    return args[0] if len(args) > 0 else None
 
if os.environ.get("DISABLE_TQDM", False):
    tqdm = _faketqdm
else:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = _faketqdm
 
# --- Internal imports ---
from . import utils, stopwords, scoring, tokenization
from . import selection as selection_np
from .version import __version__
from .tokenization import tokenize
from .scoring import (
    _select_tfc_scorer,
    _select_idf_scorer,
    _build_scores_and_indices_for_matrix,
    _calculate_doc_freqs,
    _build_idf_array,
    _build_nonoccurrence_array,
    _np_csc_python,
    _np_csc_jit_ready,
)
 