import hashlib
import re
from typing import List, Union

import numpy as np


def _ensure_text_list(text: Union[str, List[str]]) -> List[str]:
    if isinstance(text, str):
        return [text]
    return list(text)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def hash_embedding(text: Union[str, List[str]], dim: int = 256) -> np.ndarray:
    texts = _ensure_text_list(text)
    vectors = np.zeros((len(texts), dim), dtype=np.float32)

    for row_index, raw_text in enumerate(texts):
        content = (raw_text or "").strip().lower()
        if not content:
            continue

        tokens = re.findall(r"\w+", content, flags=re.UNICODE)
        if not tokens:
            tokens = [content]

        for token in tokens:
            token_index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
            vectors[row_index, token_index] += 1.0

        joined = "".join(content.split())
        for start in range(max(len(joined) - 1, 0)):
            gram = joined[start:start + 2]
            gram_index = int(hashlib.sha1(gram.encode("utf-8")).hexdigest(), 16) % dim
            vectors[row_index, gram_index] += 0.5

    return _normalize_rows(vectors)
