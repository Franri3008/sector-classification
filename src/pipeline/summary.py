from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# USD per 1M tokens. Best-effort snapshot; update as OpenAI pricing shifts.
# `embedding` entries have only `input` since embeddings have no output.
_PRICES: dict[tuple[str, str], dict[str, float]] = {
    ("openai", "text-embedding-3-large"): {"input": 0.13},
    ("openai", "text-embedding-3-small"): {"input": 0.02},
    ("openai", "gpt-4o-mini"): {"input": 0.15, "output": 0.60},
    ("openai", "gpt-4o"): {"input": 2.50, "output": 10.00},
    ("openai", "gpt-4.1-mini"): {"input": 0.40, "output": 1.60},
    ("openai", "gpt-4.1"): {"input": 2.00, "output": 8.00},
}


@dataclass
class UsageRecord:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    estimated: bool = False


class UsageTracker:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], UsageRecord] = {}
        self._lock = threading.Lock()

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated: bool = False,
    ) -> None:
        with self._lock:
            key = (provider, model)
            r = self._records.setdefault(key, UsageRecord(provider, model))
            r.input_tokens += int(input_tokens or 0)
            r.output_tokens += int(output_tokens or 0)
            r.calls += 1
            r.estimated = r.estimated or estimated

    def records(self) -> list[UsageRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: (r.provider, r.model))


_active_tracker: UsageTracker | None = None
_tracker_lock = threading.Lock()


def record_usage(
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    estimated: bool = False,
) -> None:
    tracker = _active_tracker
    if tracker is None:
        return
    tracker.record(provider, model, input_tokens, output_tokens, estimated)


@contextmanager
def capture_usage():
    global _active_tracker
    tracker = UsageTracker()
    with _tracker_lock:
        prev = _active_tracker
        _active_tracker = tracker
    try:
        yield tracker
    finally:
        with _tracker_lock:
            _active_tracker = prev


def _cost(record: UsageRecord) -> float | None:
    prices = _PRICES.get((record.provider, record.model))
    if prices is None:
        return None
    total = prices.get("input", 0.0) * record.input_tokens / 1_000_000
    total += prices.get("output", 0.0) * record.output_tokens / 1_000_000
    return total


@dataclass
class RunSummary:
    source: str
    output_path: Path
    input_rows: int
    unique_tags: int
    tags_with_picks: int
    tags_no_pick: int
    output_rows: int
    elapsed_s: float


class WarningCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.total: int = 0
        self.by_logger: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        self.total += 1
        self.by_logger[record.name] = self.by_logger.get(record.name, 0) + 1


@dataclass
class _NullCounter:
    total: int = 0
    by_logger: dict[str, int] = field(default_factory=dict)


@contextmanager
def capture_warnings():
    counter = WarningCounter()
    root = logging.getLogger()
    root.addHandler(counter)
    try:
        yield counter
    finally:
        root.removeHandler(counter)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _format_usage(tracker: UsageTracker | None) -> list[str]:
    if tracker is None:
        return []
    records = tracker.records()
    if not records:
        return []
    out: list[str] = ["usage:"]
    grand_total = 0.0
    any_priced = False
    for r in records:
        cost = _cost(r)
        tokens_part = (
            f"{r.input_tokens:,} in / {r.output_tokens:,} out"
            if r.output_tokens
            else f"{r.input_tokens:,} tokens"
        )
        if cost is None:
            price_part = "(no price configured)"
        elif r.provider != "openai":
            price_part = "(local)"
        else:
            any_priced = True
            grand_total += cost
            marker = " ~est." if r.estimated else ""
            price_part = f"${cost:.4f}{marker}"
        out.append(
            f"  {r.provider:<6} {r.model:<32} {r.calls:>6,} calls  {tokens_part:<32} {price_part}"
        )
    if any_priced:
        out.append(f"  total estimated cost: ${grand_total:.4f}")
    return out


def format_report(
    summaries: list[RunSummary],
    warnings: WarningCounter | _NullCounter,
    total_elapsed_s: float,
    usage: UsageTracker | None = None,
) -> str:
    headers = ("source", "input", "tags", "picked", "no-pick", "output", "time")
    rows = [
        (
            s.source,
            _fmt_int(s.input_rows),
            _fmt_int(s.unique_tags),
            _fmt_int(s.tags_with_picks),
            _fmt_int(s.tags_no_pick),
            _fmt_int(s.output_rows),
            _fmt_duration(s.elapsed_s),
        )
        for s in summaries
    ]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    sep = "  "
    line_width = sum(widths) + len(sep) * (len(widths) - 1)
    bar = "─" * line_width

    def _row(cells: tuple[str, ...]) -> str:
        return sep.join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths, strict=True)))

    out: list[str] = []
    out.append(bar)
    out.append(f"Run summary — total {_fmt_duration(total_elapsed_s)}")
    out.append(bar)
    out.append(_row(headers))
    for r in rows:
        out.append(_row(r))
    out.append(bar)
    out.append(f"warnings: {warnings.total}")
    if warnings.by_logger:
        top = sorted(warnings.by_logger.items(), key=lambda kv: -kv[1])
        out.append("  " + ", ".join(f"{name}={count}" for name, count in top))
    usage_lines = _format_usage(usage)
    if usage_lines:
        out.extend(usage_lines)
    out.append("outputs:")
    for s in summaries:
        out.append(f"  {s.output_path}")
    return "\n".join(out)
