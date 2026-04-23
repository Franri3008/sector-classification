from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import typer

from src.config import settings
from src.logging_setup import setup_logging
from src.sources.registry import list_sources


class Source(StrEnum):
    crunchbase = "crunchbase"
    regpat = "regpat"
    openalex = "openalex"


app = typer.Typer(add_completion=False, help="NACE sector classification pipeline.")


@app.callback()
def _main(
    log_level: str = typer.Option(settings.log_level, "--log-level", help="Logger verbosity."),
) -> None:
    setup_logging(log_level)


@app.command("run-all")
def run_all_cmd(force: bool = typer.Option(False, "--force", help="Ignore caches.")) -> None:
    import time

    from src.pipeline.runner import run_all
    from src.pipeline.summary import capture_usage, capture_warnings, format_report

    t0 = time.perf_counter()
    with capture_warnings() as warnings, capture_usage() as usage:
        summaries = run_all(force=force)
    typer.echo("")
    typer.echo(format_report(summaries, warnings, time.perf_counter() - t0, usage=usage))


@app.command("run")
def run_cmd(
    source: Source,
    force: bool = typer.Option(False, "--force", help="Ignore caches."),
    skip_embed: bool = typer.Option(
        False,
        "--skip-embed",
        help="Reuse cached tag embeddings even if the set of tags has changed.",
    ),
) -> None:
    import time

    from src.pipeline.runner import run_source
    from src.pipeline.summary import capture_usage, capture_warnings, format_report

    t0 = time.perf_counter()
    with capture_warnings() as warnings, capture_usage() as usage:
        summary = run_source(source.value, force=force, skip_embed=skip_embed)
    typer.echo("")
    typer.echo(format_report([summary], warnings, time.perf_counter() - t0, usage=usage))


@app.command("sync")
def sync_cmd(
    source: Source | None = typer.Argument(None, help="Source to sync (all if omitted)."),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Re-download existing files."),
) -> None:
    from src.io.dropbox_client import DropboxClient

    client = DropboxClient()
    sources = [source.value] if source else list_sources()
    for s in sources:
        dropbox_path = getattr(settings.dropbox_paths, s)
        local_path = settings.paths.raw_dir / s
        typer.echo(f"syncing {dropbox_path} -> {local_path}")
        downloaded = client.sync_folder(dropbox_path, local_path, force=force)
        typer.echo(f"  {len(downloaded)} file(s) downloaded")


@app.command("enrich-sectors")
def enrich_sectors_cmd(force: bool = typer.Option(False, "--force")) -> None:
    from src.llm.factory import build_llm_client
    from src.pipeline.sectors import enrich_sectors

    enriched = enrich_sectors(build_llm_client(), force=force)
    typer.echo(f"enriched {len(enriched)} divisions")


@app.command("embed-sectors")
def embed_sectors_cmd(force: bool = typer.Option(False, "--force")) -> None:
    from src.embeddings.customtools_provider import CustomToolsEmbeddingProvider
    from src.llm.factory import build_llm_client
    from src.pipeline.sectors import embed_sectors

    embed_sectors(CustomToolsEmbeddingProvider(), llm=build_llm_client(), force=force)


@app.command("describe-tags")
def describe_tags_cmd(source: Source) -> None:
    from src.llm.factory import build_llm_client
    from src.pipeline.descriptions import ensure_descriptions
    from src.sources.registry import get_adapter

    adapter = get_adapter(source.value)
    records = adapter.load_records()
    tags = adapter.extract_tags(records).drop_duplicates("tag_id")
    ensure_descriptions(source.value, tags, build_llm_client())


@app.command("embed-tags")
def embed_tags_cmd(source: Source, force: bool = typer.Option(False, "--force")) -> None:
    from src.embeddings.customtools_provider import CustomToolsEmbeddingProvider
    from src.io.local_cache import read_parquet
    from src.pipeline.descriptions import _cache_path as desc_cache
    from src.pipeline.embed_tags import embed_tags

    desc_df = read_parquet(desc_cache(source.value))
    embed_tags(source.value, desc_df, CustomToolsEmbeddingProvider(), force=force)


@app.command("classify")
def classify_cmd(
    source: Source,
    skip_embed: bool = typer.Option(False, "--skip-embed"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    from src.pipeline.runner import run_source

    run_source(source.value, force=force, skip_embed=skip_embed)


@app.command("postprocess")
def postprocess_cmd(source: Source) -> None:
    from src.pipeline.classify import load_classifications
    from src.pipeline.postprocess import build_output
    from src.sources.registry import get_adapter

    adapter = get_adapter(source.value)
    records = adapter.load_records()
    tags_to_keys = adapter.extract_tags(records)
    classifications = load_classifications(source.value)
    out_path, n_rows = build_output(source.value, tags_to_keys, classifications)
    typer.echo(f"wrote {n_rows} rows to {out_path}")


@app.command("build-dashboard")
def build_dashboard_cmd() -> None:
    from src.pipeline.dashboard import build_dashboard_data

    path = build_dashboard_data()
    typer.echo(f"wrote {path}")
    typer.echo("open dashboard/index.html with LiveServer to view.")


@app.command("trace")
def trace_cmd(
    source: Source,
    tag: str | None = typer.Argument(
        None, help="Tag name to trace. If omitted, a random tag is sampled from cache."
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Write the trace to this file in addition to stdout."
    ),
    ranking_limit: int = typer.Option(
        20, "--ranking-limit", help="How many sectors to show in the full ranking (max 88)."
    ),
    seed: int | None = typer.Option(None, "--seed", help="Seed for the random tag picker."),
) -> None:
    from src.pipeline.trace import build_trace, format_trace, save_trace

    trace = build_trace(source.value, tag, seed=seed)
    rendered = format_trace(trace, ranking_limit=ranking_limit)
    typer.echo(rendered)
    if output is not None:
        save_trace(trace, output, ranking_limit=max(88, ranking_limit))
        typer.echo(f"\nwrote full trace (ranking_limit=88) to {output}")


@app.command("cache-info")
def cache_info_cmd() -> None:
    from src.pipeline.sectors import sectors_hash

    typer.echo(f"sectors.csv: {settings.paths.sectors_csv}")
    if settings.paths.sectors_csv.exists():
        typer.echo(f"sectors_hash: {sectors_hash()}")
    else:
        typer.echo("sectors_hash: <sectors.csv missing>")
    typer.echo(f"embedding.model: {settings.embedding.model}")
    typer.echo(f"llm.backend: {settings.llm.backend}")
    typer.echo("")
    typer.echo("cached artifacts:")
    base = settings.paths.processed_dir
    for path in sorted(base.rglob("*")) if base.exists() else []:
        if path.is_file():
            size = path.stat().st_size
            typer.echo(f"  {path.relative_to(settings.paths.root)}  ({size} bytes)")


@app.command("cache-clear")
def cache_clear_cmd(
    source: Source | None = typer.Option(None, "--source"),  # noqa: B008
    step: str | None = typer.Option(
        None, "--step", help="One of: sectors, descriptions, embeddings, classifications"
    ),
) -> None:
    import shutil

    base = settings.paths.processed_dir
    if source is None and step is None:
        typer.confirm(f"Delete ALL under {base}?", abort=True)
        if base.exists():
            shutil.rmtree(base)
        typer.echo(f"cleared: {base}")
        return
    sources = [source.value] if source else list_sources()
    for s in sources:
        target = base / s
        if step:
            target = target / step
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
            typer.echo(f"cleared: {target}")
        else:
            typer.echo(f"(nothing at {target})")


_ = Path
