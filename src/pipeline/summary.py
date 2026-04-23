from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


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


def format_report(
    summaries: list[RunSummary],
    warnings: WarningCounter | _NullCounter,
    total_elapsed_s: float,
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
    out.append("outputs:")
    for s in summaries:
        out.append(f"  {s.output_path}")
    return "\n".join(out)
