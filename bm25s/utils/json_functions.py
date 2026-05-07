import json

def dumps(obj, ensure_ascii=False):
    """Serialize object to JSON string."""
    return json.dumps(obj, ensure_ascii=ensure_ascii)

def loads(s):
    """Deserialize JSON string to object."""
    return json.loads(s)
