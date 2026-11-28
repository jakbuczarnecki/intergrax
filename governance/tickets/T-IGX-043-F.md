# T-IGX-043-F — EQO-A remediation: CONFIG-X semantic scan extension for vendor-slug defaults (gate-meaning change)

## Scope

Extend the CONFIG-X semantic production scan so that silent vendor/backend slug defaults (EQOA-A4-019 blind-spot statement; FRZ-CFG-02, FRZ-CFG-06 and FRZ-CFG-08 are PASS under CONFIG-X per the checklist) become visible to the gate: one edit of `tests/qualification/config_x/_config_x_semantic_production_scan.py` and one new test file `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py`, 12 negative tests (NT-FC3-CX1..12). Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589). Split out of `T-IGX-043-C` after the independent delegate review.

**Classification: gate-meaning change.** It changes what the gate reports about FRZ-CFG-02/06/08. It is personal to the Operator: it is not approvable by a delegate or a commissioned session, and no implementer writes its approval line. Dependency order: it depends on `T-IGX-043-C` (its repo-level test is green only after the product fixes) and merges after C, or in the same wave behind C with the Operator's ruling, never before C; C does not depend on it for its own correctness.

Tests only, no product path: not ACT-N11 sensitive; outside the EBH-7 identity allowlist, so it merges before the final EBH-7 pin. No closure, PASS, FRZ promotion or independent-verification claim is made by this ticket. The Operator's ruling of 2026-11-16 (extend the scan, scoped to the eight FC-3 paths with a recorded finding per hit outside them; same wave behind `T-IGX-043-C`; scratch measurement F-A2 authorised and not yet run) is recorded in the plan, section 14, from `FC/OPERATOR_ANSWERS_TRANSCRIPTION.md` (a session artefact, not a repository record); the approval line stays the Operator's own and is not written; the authorization entries (including the scratch measurement) are in `AUTHORIZATION_REQUESTS_T-IGX-043-F.md`. The LAB exemption is syntactic (a default reachable only under a `lab_opt_in` argument), with no module or path exemption.

## Plan digest

Plan digest: 60d2a16a20e23b167bcfebfe7977b72d869e26b09512904d630e93d8c5d303cf

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-043-F.md`](plans/PLAN_T-IGX-043-F.md). The digest was computed with `scripts/governance/plan_digest.py` on the draft; it must be recomputed after any edit of the plan.

## Approved

PENDING. The approval line goes here after plan review and is written by the Operator personally, because this ticket is a gate-meaning change; it is not written by a commissioned session and never by the implementer. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).

## Test contract

One process at a time; none was run when this ticket was drafted.

- `uv run pytest tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py -p no:cacheprovider` (12 negative tests; red at the baseline where stated; with `T-IGX-043-C` in the tree the repo-level test asserts an explicit expected-residual list, initially empty).
- `uv run pytest tests/qualification/config_x -p no:cacheprovider` (whole directory).
- Regression: one command over the 11 existing CONFIG-X node IDs listed in `regression_nodes_F.txt` of the plan package (verified to exist at the baseline and at the current head).
- `uv run python -m scripts.governance.igx_gov ticket` before push.
- Independent audit at the exact wave SHA by a separate non-implementing session (replay of the 12 tests, mutation, comparison of scan findings before and after); the implementer records no PASS.
- Rollback: revert the two-file commit (tests only).

## Out of scope

Every product path, `scripts/`, `.github/`, `docs/`, `governance/` other than this ticket and its plan, the inventory and classification modules of CONFIG-X, the EBH-3 scan extension and boundary gates (`PT-G01`), all files of `T-IGX-043-A..D`, any allowlist of providers, FRZ promotion, Docker, new vendor integrations and production configuration (C10/C15 freeze).

## Lifecycle state

`draft`

## Checkpoint — AF-W3-PT-FC3-F-01 (F-A2 pre-approval scratch measurement, 2026-11-28)

Session: `INTEGRAX | ARCH-FREEZE | W3 | PT-FC3-F | AF-W3-PT-FC3-F-01`. Authorization: **F-A2** (non-certifying; gate meaning unchanged). Canonical plan digest unchanged: `60d2a16a20e23b167bcfebfe7977b72d869e26b09512904d630e93d8c5d303cf`. Approval line: **PENDING** (not written by implementer).

| Field | Value |
|---|---|
| Measurement SHA (detached worktree) | `2f38f0031b0e9c1d5b1aba1df7db8b3dfaf31809` |
| Development execution SHA (preflight `HEAD`) | `8f03bae217386e26ce1d7804ca673f57f6bfccfc` |
| Verified remote baseline ancestor | `2f38f0031b0e9c1d5b1aba1df7db8b3dfaf31809` ∈ `origin/development` — PASS |
| Catalog slug count | 199 |
| Selector enum `.value` count | 12 |
| Unique vendor value count | 205 |
| Vendor value set digest (SHA256) | `4828bbc9f4db83be169adcefe7c4d7e715ac43ca4a5b06914fcc0d79e9aef6a4` |
| CONFIG-X inventory paths scanned | 1662 |
| FC-3 physical paths measured | 9 (plan text: eight functional groups) |
| Scratch output SHA256 | `903cafef91af71401bb3b120097c2782e03ac8476ba65bd730897abcced96032` |
| Existing CONFIG-X blocking findings at measurement SHA | 0 |
| Scratch vendor-default hits in FC-3 paths | 3 |
| Scratch LAB-contract hits in FC-3 paths | 4 |
| Scratch vendor-default hits outside FC-3 | 3 |
| Synthetic controls NT-FC3-CX1..CX12 | 12/12 PASS |
| Gate meaning changed | NO |
| Product / tracked test / plan edits in this pass | NONE (ticket only) |

**Preflight predecessors:** `SCOPE=CLOSED`; PT-WB publication present (`T-IGX-043-H`, AF-W3-PT-WB-01..03); PT-FC3 publication present (`T-IGX-043-C`, AF-W3-PT-FC3-01). F-A2 authorised; not a FRZ-CFG-08 PASS and not formal CONFIG-X qualification.

**Scope discrepancy (recorded, gate scope unchanged):** plan §5 names eight FC-3 path groups; nine physical files are listed (adds `huggingface_inference_vision.py` alongside `triton_vision.py`). F-A2 scanned all nine without widening canonical gate scope language.

**Evidence references (session-local, outside tracked tree):** scratch detector `.tmp/session/af-w3-pt-fc3-f-01/fa2_scratch_detector.py`; JSON log `.tmp/session/af-w3-pt-fc3-f-01/output/fa2_measurement.json`; detached worktree `.tmp/worktrees/af-w3-pt-fc3-f-01` @ `2f38f0031b0e9c1d5b1aba1df7db8b3dfaf31809`.

### Group A — vendor-default scratch findings (FC-3 physical scope)

| repo_path | line | enclosing_symbol | expression | resolved_vendor_value | detection_mechanism | LAB exempt | reason |
|---|---|---:|---|---|---|---|---|
| `intergrax/model_inference/adapters/huggingface_inference_vision.py` | 29 | `HuggingFaceInferenceVisionAdapter` | `"huggingface_inference"` | `huggingface_inference` | `class_field_default:string_literal` | NO | `class_field_never_lab_exempt` |
| `intergrax/model_inference/adapters/triton_vision.py` | 34 | `TritonVisionServingAdapter` | `"vision_serving"` | `vision_serving` | `class_field_default:string_literal` | NO | `class_field_never_lab_exempt` |
| `intergrax/runtime/interactions/verification/factory.py` | 47 | `InboundVerifierSettings` | `InboundVerifierMode.NONE` | `none` | `class_field_default:local_enum_member` | NO | `class_field_never_lab_exempt` |

Other FC-3 paths (`provider_authority.py`, `profile.py`, `resolver.py`, `integration_vectorstore.py`, `backend_contract.py`, `llm_guardrail/_factory.py`) produced **no** scratch vendor-default hits at measurement SHA (post–`T-IGX-043-C` selector posture).

### Group A — LAB named-constant scratch findings (FC-3 physical scope)

| repo_path | line | enclosing_symbol | expression | resolved_vendor_value | detection_mechanism | LAB exempt | reason / disposition |
|---|---|---:|---|---|---|---|---|
| `intergrax/rag/embedding/registry/provider_authority.py` | 21 | `lab_embedding_provider_slug` | `_LAB_EMBEDDING_PROVIDER` | `ollama` | `lab_named_constant` | REFUSED | **Mandatory case A:** literal syntactic-contract violation (return load without §3.1 guard); not asserted as unguarded production execution path. |
| `intergrax/rag/vectorstore/bootstrap/integration_vectorstore.py` | 105 | `lab_inmemory_vector_store_slug` | `_LAB_INMEMORY_VECTOR_STORE_SLUG` | `inmemory` | `lab_named_constant` | REFUSED | **Mandatory case B (helper return):** literal syntactic-contract violation. |
| `intergrax/rag/vectorstore/bootstrap/integration_vectorstore.py` | 135 | `_resolve_lab_inmemory_slug` | `_LAB_INMEMORY_VECTOR_STORE_SLUG` | `inmemory` | `lab_named_constant` | REFUSED | **Mandatory case B (validation load):** constant used to validate typed LAB opt-in selector; guard shape not §3.1; disposition for ChatGPT. |
| `intergrax/rag/vectorstore/bootstrap/integration_vectorstore.py` | 138 | `_resolve_lab_inmemory_slug` | `_LAB_INMEMORY_VECTOR_STORE_SLUG` | `inmemory` | `lab_named_constant` | REFUSED | Same as line 135 (comparison operand). |

### Group B — vendor-default scratch findings outside FC-3 (record only; gate scope unchanged)

| repo_path | line | enclosing_symbol | expression | resolved_vendor_value | detection_mechanism | LAB exempt | reason |
|---|---|---:|---|---|---|---|---|
| `intergrax/integrations/providers/cloud_platform/aws/adapter.py` | 29 | `_AwsCloudPlatform` | `"aws"` | `aws` | `class_field_default:string_literal` | NO | `class_field_never_lab_exempt` |
| `intergrax/integrations/providers/cloud_platform/azure/adapter.py` | 30 | `_AzureCloudPlatform` | `"azure"` | `azure` | `class_field_default:string_literal` | NO | `class_field_never_lab_exempt` |
| `intergrax/integrations/providers/cloud_platform/gcp/adapter.py` | 30 | `_GcpCloudPlatform` | `"gcp"` | `gcp` | `class_field_default:string_literal` | NO | `class_field_never_lab_exempt` |

### Group C — existing CONFIG-X semantic scan (baseline at measurement SHA)

Mechanical `discover_semantic_production_findings()` blocking set: **empty** (0 paths). The scratch detector therefore reports **3 net new vendor-default shapes** in FC-3 (enum member class-field default on `InboundVerifierSettings.mode`; vision adapter `provider` class fields) and **3** outside FC-3, plus **4** LAB syntactic-contract rows not evaluated by the current scanner.

### Synthetic controls (scratch; not canonical tests)

| ID | Result |
|---|---|
| CX1 `os.getenv` vendor default | PASS |
| CX2 enum class-field default | PASS |
| CX3 `os.environ.get` + enum `.value` | PASS |
| CX4 module-level selector scanned | PASS |
| CX5 semantic `mode` parameter default | PASS |
| CX6 catalog slug vs non-slug | PASS |
| CX7 FC-3 physical scope measurement | PASS (3 vendor + 4 LAB rows as above) |
| CX8 synthetic embedding vendor default | PASS |
| CX9 guarded `_LAB_*` load | PASS |
| CX10 unguarded `_LAB_*` load | PASS |
| CX11 mixed guarded/unguarded `_LAB_*` | PASS |
| CX12 unguarded default in lab-named module | PASS |

### Program state (preserved)

`RED_AT_PREFIX=22`; `EVIDENCE_MISSING=11`; `RED_FIRST_OUT=5`; `RED_FIRST_LATER_WAVE=1`; `OP_B_PENDING=11`; `RED_FIRST_EVIDENCE=INCOMPLETE`; `FRZ_EXE_02=EVIDENCE_MISSING`; `FRZ_PLG_05=EVIDENCE_MISSING`; `STRICT_HOST_AUTHORITY=OPEN_RESIDUAL`; `SCOPE=CLOSED`; `PT_Q02=PENDING`; `G2=OPEN`; `RB2=NOT_PERFORMED`; `FRZ_PROMOTION=NONE`.

### Unresolved (for Operator / ChatGPT; no implementer decision)

1. Whether `InboundVerifierSettings.mode = InboundVerifierMode.NONE` remains in `EXPECTED_RESIDUAL` after gate extension (dataclass default vs fail-closed selector posture).
2. Whether vision adapter `provider` class literals (`huggingface_inference`, `vision_serving`) are in-scope blocking for the eight-group gate or FRZ-CFG-05 residuals (model-name literals explicitly out of scope per plan §3.4).
3. Mandatory LAB helper dispositions (cases A/B): syntactic exemption vs validation-only load vs helper return — **not decided** in this measurement pass.
4. Three cloud-platform adapter class-field hits outside FC-3: findings for accounting only (Operator ruling §14 item 3).

**F-A2 terminal status:** measurement complete; implementation of `_config_x_semantic_production_scan.py` extension **not started**; FRZ-CFG-08 **not** certified PASS.
