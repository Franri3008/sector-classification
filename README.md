# sector-classification

> Map tags from Crunchbase, Regpat, and OpenAlex to NACE Rev. 2 sector divisions using embeddings + LLM.

## Stack

- **Python** 3.11+
- **Embeddings**: [Custom-Tools](https://github.com/Franri3008/Custom-Tools) `CustomTools.llm.embedding.embed` (OpenAI `text-embedding-3-large` under the hood; behind a provider interface — swappable)
- **LLM**: OpenAI or local vLLM/Gemma (pluggable via `LLM_BACKEND`)
- **Data**: Dropbox → `data/raw/<source>/` → parquet caches in `data/processed/` → long-format CSVs in `data/outputs/`
- **CLI**: Typer
- **Config**: `pydantic-settings` (`src/config.py` + `.env`)

## Getting started

```bash
cp .env.example .env
# fill DROPBOX_TOKEN, OPENAI_API_KEY, LLM_BACKEND

pip install -r requirements-dev.txt
pip install -e ../Custom-Tools   # sibling repo providing CustomTools.llm.embedding
pip install -e .                 # so `src.*` imports resolve in your shell

# Run the full pipeline over all three sources
sector-classify run-all

# Or one at a time
sector-classify run openalex
```

## Pipeline

```
sectors.csv ── embed ───────────┐
                                v
raw/<source>/ ── load ── extract tags ── describe (LLM, cached) ── embed tags ── cosine top-N
                                                                                      │
                                                                                      v
                                                                              LLM classify (JSON)
                                                                                      │
                                                                                      v
                                                                 postprocess → outputs/<source>__<ts>.csv
```

- **Step 0** `embed-sectors` — embed the 88 NACE divisions once.
- **Step 1** Per source, the `SourceAdapter` loads raw files and emits `(tag_id, tag_name, key)` rows.
- **Step 2** `describe-tags` generates a 1–2 sentence description per unique tag (cached to disk).
- **Step 3** `embed-tags` embeds `tag_name + description` (cached).
- **Step 4** `ranking.top_candidates` takes top-N by cosine similarity with a minimum floor.
- **Step 5** `classify` asks the LLM to pick the best subset and return JSON (cached).
- **Step 6** `postprocess` joins picks back to keys and writes the output CSV.

## CLI

```
sector-classify run-all                        # full pipeline, all sources
sector-classify run SOURCE                     # one source end-to-end
sector-classify embed-sectors                  # step 0
sector-classify describe-tags SOURCE
sector-classify embed-tags SOURCE
sector-classify classify SOURCE [--skip-embed]
sector-classify postprocess SOURCE
sector-classify cache-info                     # show current hashes + cached artifacts
sector-classify cache-clear [--source S] [--step sectors|descriptions|embeddings|classifications]
```

All commands accept `--force` to ignore caches.

## Environment variables

| Variable | Description | Required |
|---|---|---|
| `DROPBOX_TOKEN` | Dropbox access token | ✅ |
| `OPENAI_API_KEY` | OpenAI API key (embeddings + OpenAI LLM backend) | ✅ |
| `LLM_BACKEND` | `openai` or `vllm` | ✅ |
| `VLLM_BASE_URL` | Local vLLM server URL (only when `LLM_BACKEND=vllm`) | only for vLLM |

Tunable defaults live in [src/config.py](src/config.py) — override via nested env vars, e.g. `LLM__OPENAI_MODEL=gpt-4o` or `RANKING__TOP_N=15`.

## Cache layout

```
data/processed/
├── sectors/
│   └── sectors__<embed_model>__<sha256(sectors.csv)>.{npz,meta.json}
└── <source>/
    ├── descriptions/descriptions.parquet         # tag_id, tag_name, description, llm_model
    ├── embeddings/tags__<embed_model>.{npz,meta.json}
    └── classifications/<backend>__<llm_model>__<sectors_hash>__topN<N>_floor<F>.parquet
```

Rules:

- `sectors_hash` invalidates **only** sector embeddings + classifications (tag embeddings and descriptions survive).
- Cache writes are atomic (`tmp + os.replace`) — crashes never leave half-written parquet.
- Every cache file has a `.meta.json` sidecar recording `model_id`, `dim`, `created_at`.

## Project structure

```
sector-classification/
├── data/                     # raw, processed caches, outputs (gitignored)
├── notebooks/                # exploration only
├── src/
│   ├── config.py             # pydantic-settings
│   ├── logging_setup.py
│   ├── cli.py                # Typer entrypoint
│   ├── io/                   # dropbox_client, atomic local cache helpers
│   ├── sources/              # adapters: crunchbase, regpat, openalex
│   ├── embeddings/           # provider interface + OpenAI impl
│   ├── llm/                  # LLMClient ABC, OpenAI + vLLM, shared schemas + prompts
│   ├── pipeline/             # sectors, descriptions, embed_tags, ranking, classify, postprocess, runner
│   └── utils/                # hashing, retry, batching
├── tests/
│   ├── conftest.py           # FakeEmbedder + FakeLLM fixtures, tmp_project fixture
│   ├── unit/
│   └── integration/          # end-to-end with fakes — no external calls
├── sectors.csv               # 88 NACE divisions (reference data)
├── .env.example
├── Makefile                  # install, run, test, lint, format, cache-info
├── pyproject.toml            # ruff + pytest + pyright config
├── requirements.txt
├── requirements-dev.txt
└── pyrightconfig.json
```

## Running tests

```bash
make test
```

The integration test `tests/integration/test_pipeline_tiny.py` runs the full pipeline with `FakeEmbedder` and `FakeLLM` — no network calls required.

## Pitfalls handled in the implementation

- **Embedding model drift** — `model_id` + `dim` stamped into cache filenames and `.meta.json`.
- **`sectors.csv` mutation** — `sha256(sectors.csv)` folded into the cache key.
- **Nondeterministic LLM** — `temperature=0`, `seed=42`, pydantic schema validation with one bounded repair retry.
- **Partial-failure recovery** — checkpoints every 50 tags; atomic parquet rewrites.
- **Tag normalization** — lowercase/strip/collapse-whitespace before hashing so near-duplicate tags share a description cache.
- **Cosine similarity** — explicit L2 normalization via `embed_normalized`.
- **Dropbox token expiry** — single seam in `DropboxClient` where refresh-on-401 will live.
- **JSON-mode parity** — OpenAI and vLLM both return through the same pydantic schemas; prompts are shared.

## Next steps (not in scope for the scaffold)

- Wire a proper Dropbox refresh-token flow (the scaffold uses a raw long-lived token).
- Add `--dry-run` token/cost estimation to the `run` commands.
- Add a `models/` local sentence-transformers provider alongside OpenAI.
- Explore source-specific prompt variants (Regpat IPC codes benefit from a taxonomy-aware prompt).
