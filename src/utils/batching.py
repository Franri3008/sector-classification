from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def throttled_map(
    fn: Callable[[list[T]], list[R]],
    items: list[T],
    batch_size: int,
    rpm: int | None = None,
    on_batch_done: Callable[[list[T], list[R]], None] | None = None,
) -> list[R]:
    min_interval = 60.0 / rpm if rpm else 0.0
    results: list[R] = []
    last_call = 0.0
    for batch in chunked(items, batch_size):
        if min_interval:
            elapsed = time.monotonic() - last_call
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        last_call = time.monotonic()
        batch_results = fn(batch)
        if len(batch_results) != len(batch):
            raise RuntimeError(
                f"batched fn returned {len(batch_results)} items for input of size {len(batch)}"
            )
        results.extend(batch_results)
        if on_batch_done is not None:
            on_batch_done(batch, batch_results)
    return results
