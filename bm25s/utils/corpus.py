import json
from pathlib import Path
from typing import List, Union

class JsonlCorpus:
    """Memory-mapped JSONL corpus for efficient loading of large datasets."""
    
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._offsets = []
        self._load_offsets()
    
    def _load_offsets(self):
        """Load line offsets for memory-mapped access."""
        with open(self.path, 'rb') as f:
            while True:
                self._offsets.append(f.tell())
                line = f.readline()
                if not line:
                    break
        # Remove the last offset since it's past EOF
        if self._offsets:
            self._offsets.pop()
    
    def __getitem__(self, idx):
        """Get document(s) by index."""
        if isinstance(idx, int):
            with open(self.path, 'r', encoding='utf-8') as f:
                f.seek(self._offsets[idx])
                line = f.readline()
                return json.loads(line)
        elif isinstance(idx, (list, tuple)):
            return [self[i] for i in idx]
        else:
            raise TypeError("Index must be int or list-like")
    
    def __len__(self):
        return len(self._offsets)

def find_newline_positions(path: Union[str, Path], show_progress: bool = True) -> List[int]:
    """Find positions of newlines in a file."""
    positions = []
    with open(path, 'rb') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            positions.append(pos)
    return positions

def save_mmindex(positions: List[int], path: Union[str, Path]):
    """Save memory-mapped index (stub implementation)."""
    # This is a stub - actual implementation would save index metadata
    pass
