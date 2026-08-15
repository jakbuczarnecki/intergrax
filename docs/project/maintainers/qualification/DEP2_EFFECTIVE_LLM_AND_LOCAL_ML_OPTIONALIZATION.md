# DEP-2 — Effective LLM and Local-ML Optionalization

**Status:** `READY_FOR_REVIEW`
**Qualification date:** 2026-08-10
**Scope correction:** `openai-whisper` is the only DEP-3-class media dependency
moved early in DEP-2. No other media, parser, vector, UI, or integration
dependency ownership was changed by this correction.

## Media Whisper

- core ownership: none
- extra: `media-whisper`
- clean core installed: no (`openai-whisper` absent)
- fresh extra import: PASS
- `whisper.load_model`: callable
- model downloaded: no
- resolved Torch in `media-whisper`: `2.13.0`
- combined with `rag-local-embeddings`: PASS

The extra keeps the intended declaration
`openai-whisper>=20240930,<20250626`. Whisper selection without the extra
raises the existing `IntegrationDependencyError` contract and identifies
`Intergrax-ai[media-whisper]`; it does not expose a raw
`ModuleNotFoundError`.

## Clean core

- torch: NOT INSTALLED
- sentence-transformers: NOT INSTALLED
- transformers: NOT INSTALLED
- openai-whisper: NOT INSTALLED
- distributions: `179`
- site-packages footprint: `764.75 MiB`
- Harness import: `17.703 s`
- `import intergrax`, `intergrax.runtime.nexus`, `intergrax.harness`: PASS
- provider registry enumeration: PASS; 19 providers, optional SDKs not loaded
- default core `langchain*`: `0`
- default core `langgraph*`: `0`

The clean-core measurement is a fresh Python 3.12 environment installed from
the default package only. Torch remains present in `uv.lock` only because the
lock includes optional extras.

## Local-ML and media matrix

- default core: PASS
- `rag-local-embeddings`: PASS; Torch `2.2.2+cpu`, sentence-transformers
  `5.7.0`, transformers `4.57.6`, Whisper absent
- `media-whisper`: PASS; Whisper `20250625`, `load_model` callable
- `rag-local-embeddings + media-whisper`: PASS; Whisper and
  sentence-transformers imports, HF registry construction, and callable
  `load_model`
- HF mock embedding construction: PASS
- aggregate local-ML/media probe: PASS

## LLM extras

Fresh resolver gates passed for each moved extra:
`llm-anthropic`, `llm-mistral`, `llm-ollama`, `llm-gemini`, `llm-bedrock`,
and `llm-cohere-native`. Fresh `llm-all` installation imported all native
SDKs successfully and did not install LangChain or LangGraph.

- `llm-*`: PASS
- `llm-all`: PASS

## Regression and static checks

- focused DEP-2 regression tests: `27 PASS`
- Ruff/linter: `PASS`

## Local-ML aggregate failure

- command: `build/dep2-matrix-combined/Scripts/python.exe -c "import os; exec(os.environ['DEP2_CODE'])"`
- exit: `0`
- stdout: `aggregate PASS`
- stderr: empty
- failure phase: none in the reproducible combined probe
- root cause: the previously reported process failure was not reproducible;
  independent imports, provider mock construction, and the combined fresh
  environment all pass
- classification: `AGGREGATE_TEST_HARNESS_DEFECT`
- blocking: no

The classification is limited to the aggregate process failure; no provider
runtime regression was observed.

## Scope exception

- moved media dependencies: `openai-whisper` only
- other DEP-3 dependency ownership moves: `0`

## Chronological review-fix: Harness import regression forensics

- original measurement: `17.703 s` for `import intergrax.harness`
- reproduction: fresh Python 3.12.11 environments, five fresh subprocesses per
  package; current DEP-2 cold runs were `5.597`, `2.265`, `2.247`, `2.353`,
  and `2.446 s` (min `2.247 s`, median `2.353 s`, max `5.597 s`); the same
  installed environment repeated with five fresh processes was min `2.268 s`,
  median `2.295 s`, max `2.624 s`; exact DEP-1 commit `987547708a19e0f88c30c49f2559160f00d27466`
  measured min `2.258 s`, median `2.508 s`, max `4.714 s`
- root cause: the `17.703 s` result was not reproducible; the bounded Harness
  import chain has no eager optional provider or local-ML SDK import
- classification: `MEASUREMENT_VARIANCE`
- final median: `2.353 s` for current DEP-2 cold fresh-process runs
- import-profile findings: standard `-X importtime` reported
  `intergrax.harness` at `2.195 s` cumulative, owned mainly by
  `intergrax.applications.contracts.graph_builder` (`2.052 s`),
  `intergrax.applications.contracts.agent_ref` (`1.704 s`), and
  `intergrax.runtime.nexus.engine.runtime_context` (`0.982 s`); local ML
  modules and optional LLM SDKs were absent; `numpy` was present, while
  `openai` and `boto3` were absent after the Harness import
- runtime change required: `no`; stale Bedrock registry metadata for
  `mypy_boto3_bedrock_runtime` was removed because the adapter imports only
  `boto3`
