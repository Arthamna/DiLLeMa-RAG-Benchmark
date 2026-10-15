from __future__ import annotations

import re
from collections import Counter
from hashlib import blake2b

from qdrant_client.models import SparseVector

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_VOCAB = 2_147_483_647


def tokenize(text: str) -> list[str]:
    """Surface-form multilingual tokens. Stemming is intentionally disabled."""
    return _TOKEN.findall(text.casefold())


def term_index(token: str) -> int:
    return int.from_bytes(blake2b(token.encode("utf-8"), digest_size=4).digest(), "big") % _VOCAB


def encode_sparse(text: str) -> SparseVector:
    counts: Counter[int] = Counter(term_index(token) for token in tokenize(text))
    items = sorted(counts.items())
    return SparseVector(
        indices=[index for index, _ in items],
        values=[float(value) for _, value in items],
    )

