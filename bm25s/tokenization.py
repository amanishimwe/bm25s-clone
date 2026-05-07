"""Tokenization module."""

from typing import List, Dict

class Tokenized:
    """Tokenized corpus representation."""
    
    def __init__(self, ids: List[List[int]], vocab: Dict):
        self.ids = ids
        self.vocab = vocab

def tokenize(text: str) -> List[str]:
    """Simple tokenization: split on whitespace and lowercase."""
    return text.lower().split()

def convert_tokenized_to_string_list(tokenized: Tokenized) -> List[List[str]]:
    """Convert Tokenized object to list of token strings."""
    # Create reverse vocabulary mapping
    reverse_vocab = {v: k for k, v in tokenized.vocab.items()}
    
    # Convert token IDs to strings
    result = []
    for doc_ids in tokenized.ids:
        doc_tokens = [reverse_vocab.get(token_id, '') for token_id in doc_ids]
        result.append(doc_tokens)
    
    return result
