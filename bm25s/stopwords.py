"""Stopwords module stub."""

ENGLISH_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
    'the', 'to', 'was', 'will', 'with'
}

def get_stopwords(language: str = 'english'):
    """Get stopwords for a language."""
    if language.lower() == 'english':
        return ENGLISH_STOPWORDS
    return set()
