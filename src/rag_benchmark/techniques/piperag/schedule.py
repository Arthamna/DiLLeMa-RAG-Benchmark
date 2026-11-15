from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class StaleAsyncSchedule(Generic[T]):
    """One-interval stale retrieval schedule used by PipeRAG.

    The first boundary blocks. The second boundary reuses the first result and
    starts retrieval for its now-complete query. Later boundaries consume the
    outstanding request and immediately launch the next one.
    """

    def __init__(self, retrieve: Callable[[str, int], T]):
        self.retrieve = retrieve
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piperag-retrieval")
        self.current: T | None = None
        self.pending: Future[T] | None = None
        self.calls = 0

    def boundary(self, query: str, seq_len: int) -> T:
        if self.current is None:
            self.current = self.retrieve(query, seq_len)
            self.calls += 1
            return self.current
        if self.pending is not None:
            self.current = self.pending.result()
        self.pending = self.executor.submit(self.retrieve, query, seq_len)
        self.calls += 1
        return self.current

    def close(self) -> None:
        if self.pending is not None:
            self.pending.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
