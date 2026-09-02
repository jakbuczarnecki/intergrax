# LCI-7C Compatibility Stability Forensics

**Task:** `LCI-7C-STABILITY-1-REPRODUCIBLE-TORCH-IMPORT-FAILURE-FORENSICS`
**Date:** 2026-08-09
**Platform:** Windows 10 (`win32`)
**Python:** 3.12.11
**uv:** 0.8.15
**Validated HEAD:** `924478ee499b182bf12cd0ff2567a16194b802d6`
**Status:** `ROOT_CAUSE_RESOLVED - READY_FOR_FINAL_GATE_RERUN`

## Preflight and scope

Branch `development`, HEAD and `origin/development` were both `924478ee`. Required ancestors `20bc04d`, `079123ab`, and `3b36ac38` passed. Later parallel commits `30238c5a` and `924478ee` change vendor-knowledge/RAG qualification work only; they do not change packaging or the compatibility gate. Unrelated unstaged work was preserved and the staging index was empty.

No production runtime, `pyproject.toml`, `uv.lock`, or RAG path was modified by this investigation.

## Reproduction matrix

Every run used a unique task-owned temporary virtual environment, Python 3.12.11, a fresh Python process for each probe, and `PYTHONNOUSERSITE=1`. Runs 1–4 used the normal uv cache; run 5 used a task-owned isolated uv cache.

| Family | Run | Environment | Canonical case | Cache | Result |
|---|---:|---|---|---|---|
| `rag-langchain-loaders` | 1 | `rag-langchain-loaders-run1` | E | normal | FAIL: `NameError: name 'torch' is not defined` |
| `rag-langchain-loaders` | 2 | `rag-langchain-loaders-run2` | E | normal | FAIL: same traceback fingerprint |
| `rag-langchain-loaders` | 3 | `rag-langchain-loaders-run3` | E | normal | FAIL: same traceback fingerprint |
| `rag-langchain-loaders` | 4 | `rag-langchain-loaders-run4` | E | normal | FAIL: same traceback fingerprint |
| `rag-langchain-loaders` | 5 | `rag-langchain-loaders-run5` | E | isolated | FAIL: same traceback fingerprint |
| `rag-langchain-splitters` | 1 | `rag-langchain-splitters-run1` | F | normal | FAIL: `NameError: name 'torch' is not defined` |
| `rag-langchain-splitters` | 2 | `rag-langchain-splitters-run2` | F | normal | FAIL: same traceback fingerprint |
| `rag-langchain-splitters` | 3 | `rag-langchain-splitters-run3` | F | normal | FAIL: same traceback fingerprint |
| `rag-langchain-splitters` | 4 | `rag-langchain-splitters-run4` | F | normal | FAIL: same traceback fingerprint |
| `rag-langchain-splitters` | 5 | `rag-langchain-splitters-run5` | F | isolated | FAIL: same traceback fingerprint |

The loader canonical path failed 5/5 and the splitter canonical path failed 5/5. Full process tracebacks and package fingerprints were captured in the task-owned forensic run directory outside the repository.

## Exact failing traceback fingerprint

Exception:

```text
NameError: name 'torch' is not defined
```

First project-independent failing source:

```text
transformers/integrations/tensor_parallel.py:465
class _AllReduceBackward(torch.autograd.Function):
```

Failing module: `transformers.integrations.tensor_parallel` during module/class-body evaluation. At the failing frame, `torch_in_module_dict=False` and the `torch` global is absent. At the same time:

```text
sys.modules["torch"] = torch.__init__ from the fresh venv
sys.modules["torch"].__version__ = "2.2.2+cpu"
```

Loader import chain reaches the failure through `langchain_community.document_loaders` → `langchain_core.document_loaders` → `langchain_text_splitters` → `sentence_transformers` → `transformers`. Splitter import chain reaches it through Intergrax's default embedding bootstrap → `hf_embedding_provider` → `sentence_transformers` → `transformers`.

The installed `transformers/utils/import_utils.py:150-156` reports `is_torch_available() == False` because it explicitly disables PyTorch below `2.4.0`, while `transformers/integrations/tensor_parallel.py:28-35` imports `torch` only inside `if is_torch_available():`. The class at line 465 is nevertheless unconditional. This is an upstream conditional-import defect exposed by the incompatible version tuple.

## Package fingerprint

Current failing runs resolved the same versions from the same configured indexes:

| Package | Current failing resolution |
|---|---|
| Python | 3.12.11 |
| `torch` | 2.2.2+cpu; PyTorch CPU index; `torchvision`: NOT INSTALLED; `torchaudio`: NOT INSTALLED |
| `sentence-transformers` | 5.7.0 |
| `transformers` | 5.14.1 |
| `tokenizers` | 0.22.2 |
| `langchain-core` | 1.5.3 |
| `langchain-community` | 0.4.2 in loaders; NOT INSTALLED in splitters |
| `langchain-text-splitters` | 1.1.2 |
| `langchain-classic` | 1.0.8 in loaders; NOT INSTALLED in splitters |
| `numpy` | 1.26.4 |
| `scipy` | 1.17.1 |
| `huggingface-hub` | 1.27.0 |
| `safetensors` | 0.8.0 |

The current installation source is the checkout plus PyPI and the configured PyTorch CPU index (`https://download.pytorch.org/whl/cpu`). Installer metadata reports `uv`. `uv pip check` passes in both representative current environments despite the runtime import defect.

Normal-cache and isolated-cache runs selected the same versions and the same wheel URLs, including the PyTorch CPU wheel and the PyPI `transformers-5.14.1`, `sentence_transformers-5.7.0`, `langchain_community-0.4.2`, and `langchain_text_splitters-1.1.2` artifacts. The differing generated `RECORD` hashes for executable wrappers are environment-path effects, not different wheel provenance.

## Import-order matrix

Each case ran in a separate fresh Python process. `G` and `H` are the required canonical cases with `import torch` added first.

| Case | Import sequence | Loaders env | Splitters env |
|---|---|---|---|
| A | `torch`, `transformers`, `sentence_transformers`, `langchain_text_splitters` | FAIL: NameError | FAIL: NameError |
| B | `transformers`, `torch`, `sentence_transformers`, `langchain_text_splitters` | FAIL: NameError | FAIL: NameError |
| C | `sentence_transformers` | FAIL: NameError | FAIL: NameError |
| D | `langchain_text_splitters` | FAIL: NameError | FAIL: NameError |
| E | loader canonical gate order | FAIL: NameError | NOT APPLICABLE: loader extra absent |
| F | splitter canonical gate order | FAIL: NameError | FAIL: NameError |
| G | loader order with `import torch` first | FAIL: NameError | NOT APPLICABLE: loader extra absent |
| H | splitter order with `import torch` first | FAIL: NameError | FAIL: NameError |

Pre-importing `torch` does not mask the failure. This is not an import-order workaround or a Python module-cache artifact.

## PASS control and resolution comparison

A separate fresh control environment explicitly installed the previously accepted tuple:

```text
torch==2.2.2+cpu
sentence-transformers==5.2.0
transformers==4.57.6
tokenizers==0.22.2
langchain-core==1.2.7
langchain-community==0.4.1
langchain-text-splitters==1.1.0
```

Loader case E, splitter case F, and direct splitter case D all PASS; `uv pip check` also passes. Therefore the required comparison result is:

```text
DIFFERENT_RESOLUTION
```

The earlier PASS was not the same resolved dependency set. The current unbounded `uv pip install` resolves a newer Transformers release whose own availability gate rejects the pinned Torch version, while its import surface still references the missing global. The cache comparison does not support `E - uv/cache resolution instability` as the root cause.

## Invocation and lifecycle audit

The GitHub workflow and the local canonical invocation both call the same source of truth:

```text
scripts/ci/check_langchain_compatibility_install.py --family <extra>
```

The only platform difference is `Scripts/python.exe` versus the CI Unix path. The caller creates the environment and runs `uv pip install --python <env> ".[<extra>]"`; no shared `.venv` or `--system` install is used. The local forensic runs used unique paths and failed if a path already existed. No environment reuse or cross-family contamination was found. `PYTHONNOUSERSITE=1` was set for every probe. The final gate's timestamped local paths were also unique.

The two accepted historical PASS environments and the current failing environments use the same canonical gate implementation; the material difference is live resolver output, not gate invocation. The parallel commits `3b36ac38`, `30238c5a`, and `924478ee` are unrelated to this dependency closure.

## Root cause and decision

**Category: `A - dependency version incompatibility`.**

The project pins `torch==2.2.2` and allows `sentence-transformers>=3.0`; it does not constrain the transitive Transformers major line. Current resolution selects `transformers 5.14.1`, whose `is_torch_available()` policy requires Torch `>=2.4`, but whose tensor-parallel module still unconditionally evaluates `torch.autograd.Function`. This is why both compatibility families fail through their shared sentence-transformers import surface.

No production, RAG, gate, `pyproject.toml`, or lockfile fix was applied. A packaging decision is required before any constraint is added. The tested candidate compatibility tuple is the exact historical tuple documented above; it should not be added without operator approval.

## Audits

- Inventory and boundary audits remain read-only and are rerun after evidence changes.
- `uv lock --check` remains read-only and is rerun after evidence changes.
- The four known generated LangGraph findings remain outside this task.

## Conclusion

The former classification `E - prior failure non-reproducible` is superseded. The failure is recurrent and deterministic for the current resolution: 5/5 fresh loader environments and 5/5 fresh splitter environments fail with the same upstream traceback. Final system gate rerun is not authorized until the dependency constraint decision and compatibility requalification are complete. `LCI-8A` was not started.

## LCI-7C-STABILITY-2 requalification

**Operator decision:** keep `torch==2.2.2`; add the direct compatibility constraint
`transformers>=4.41,<5`. `sentence-transformers>=3.0` remains unchanged. Torch
was not upgraded because the project support matrix intentionally retains the
2.2.2 CPU runtime; the failure crossed the Transformers major-version
compatibility boundary, so constraining that existing transitive runtime
dependency is the minimal repair.

The lock resolver selected this same tuple on Windows and Linux markers:

| Package | Resolved version |
|---|---|
| `torch` | `2.2.2+cpu` |
| `sentence-transformers` | `5.7.0` |
| `transformers` | `4.57.6` |
| `tokenizers` | `0.22.2` |
| `langchain-core` | `1.5.3` |
| `langchain-community` | `0.4.2` in loaders |
| `langchain-text-splitters` | `1.1.2` |

Three unique fresh Python 3.12 environments were installed and qualified for
each target family, outside the checkout, with `PYTHONNOUSERSITE=1`:

- `rag-langchain-loaders`: 3/3 PASS
- `rag-langchain-splitters`: 3/3 PASS

In the first fresh environment of each target family, the direct regression
probe imported `torch`, `transformers`, `sentence_transformers`, and
`langchain_text_splitters`; `transformers.utils.is_torch_available()` returned
`True`; and `transformers.integrations.tensor_parallel` imported successfully.

The remaining 7C families also passed once each in fresh environments:
`llm-langchain-ollama`, `rag-langchain-embeddings`, and `langgraph-legacy`.
Native Ollama remained the registry default and the native RAG splitter
remained the default.

The fresh 7B clean-core gate passed with zero installed LangChain and LangGraph
distributions. Targeted regression tests passed (`62 passed`), and inventory,
boundary, and `uv lock --check` audits passed.

**Current status:** `ROOT_CAUSE_RESOLVED - READY_FOR_FINAL_GATE_RERUN`.
The Final System Gate was not started and `LCI-8A` was not started.
