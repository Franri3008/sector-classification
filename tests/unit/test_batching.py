from src.utils.batching import chunked, throttled_map


def test_chunked_exact_and_partial():
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunked([], 3)) == []


def test_throttled_map_checkpoints():
    seen: list[tuple[list[int], list[int]]] = []

    def fn(batch: list[int]) -> list[int]:
        return [x * 2 for x in batch]

    def checkpoint(batch, result):
        seen.append((batch, result))

    out = throttled_map(fn, [1, 2, 3, 4, 5], batch_size=2, on_batch_done=checkpoint)
    assert out == [2, 4, 6, 8, 10]
    assert [s[0] for s in seen] == [[1, 2], [3, 4], [5]]
