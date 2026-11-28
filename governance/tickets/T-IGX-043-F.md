# T-IGX-043-F — EQO-A remediation: CONFIG-X semantic scan extension for vendor-slug defaults (gate-meaning change)

## Scope

Extend the CONFIG-X semantic production scan so that silent vendor/backend slug defaults (EQOA-A4-019 blind-spot statement; FRZ-CFG-02, FRZ-CFG-06 and FRZ-CFG-08 are PASS under CONFIG-X per the checklist) become visible to the gate: one edit of `tests/qualification/config_x/_config_x_semantic_production_scan.py` and one new test file `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py`, 12 negative tests (NT-FC3-CX1..12). Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589). Split out of `T-IGX-043-C` after the independent delegate review.

**Classification: gate-meaning change.** It changes what the gate reports about FRZ-CFG-02/06/08. It is personal to the Operator: it is not approvable by a delegate or a commissioned session, and no implementer writes its approval line. Dependency order: it depends on `T-IGX-043-C` (its repo-level test is green only after the product fixes) and merges after C, or in the same wave behind C with the Operator's ruling, never before C; C does not depend on it for its own correctness.

Tests only, no product path: not ACT-N11 sensitive; outside the EBH-7 identity allowlist, so it merges before the final EBH-7 pin. No closure, PASS, FRZ promotion or independent-verification claim is made by this ticket. The Operator's ruling of 2026-11-16 (extend the scan, scoped to the eight FC-3 paths with a recorded finding per hit outside them; same wave behind `T-IGX-043-C`; scratch measurement F-A2 authorised and not yet run) is recorded in the plan, section 14, from `FC/OPERATOR_ANSWERS_TRANSCRIPTION.md` (a session artefact, not a repository record); the approval line stays the Operator's own and is not written; the authorization entries (including the scratch measurement) are in `AUTHORIZATION_REQUESTS_T-IGX-043-F.md`. The LAB exemption is syntactic (a default reachable only under a `lab_opt_in` argument), with no module or path exemption.

## Plan digest

Plan digest: b434679cdc9be69d755cda28234a4e543605cd5122ae617800c0f70456dc2707

Previous plan digest (revision 3 / pre OD-AF-W3-FC3-F-01): 60d2a16a20e23b167bcfebfe7977b72d869e26b09512904d630e93d8c5d303cf

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
| Vendor value set digest (SHA256, reconciled) | `4828bbc9f4db83be169adcefe7c4d7e715ac43ca4a5b06914fcc0d79e9aef6a4` |
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

**Vendor digest reconciliation (AF-W3-PT-FC3-F-02):** Composer session AF-W3-PT-FC3-F-01 conflated a plan digest with the vendor value digest. Authoritative vendor digest remains `4828bbc9f4db83be169adcefe7c4d7e715ac43ca4a5b06914fcc0d79e9aef6a4` (sorted newline SHA256 over 205 derived values; recomputed at `9efe643045c40255ce868e99400e9468d2fee4f4`). F-A2 artifact SHA256: `903cafef91af71401bb3b120097c2782e03ac8476ba65bd730897abcced96032` (file hash `eb6b5ebff111e8df5a4faec75286980fd7a395e9d3d17b379dcc2dd05cacfd0e`).

## Checkpoint — AF-W3-PT-FC3-F-02 (CONFIG-X scoped vendor gate, 2026-11-28)

Session: `INTEGRAX | ARCH-FREEZE | W3 | PT-FC3-F | AF-W3-PT-FC3-F-02`. Operator: **OD-AF-W3-FC3-F-01 variant A DECIDED** (recorded in Operator Decision Register). Plan digest: `204146325aaf01b8a1eda7e391b40f4f1dea2d4a4706078bcc87b69a10e285f9`.

| Item | Status |
|---|---|
| Scoped physical paths | 9 |
| EXPECTED_RESIDUAL fingerprints | 3 (`InboundVerifierSettings.mode`, LAB helper returns) |
| Vision provider identity | `HuggingFaceInferenceVisionAdapter.slug`, `TritonVisionServingAdapter.slug` |
| Validation-only LAB loads | `_resolve_lab_inmemory_slug` comparison + error message |
| NT-FC3-CX1..12 + behavioral controls | GREEN at execution SHA (see git) |
| CONFIG-X directory regression | 96 passed |
| FRZ-CFG-08 owner regression node | PASS |
| FC3 behavioral regression (Q5) | 33 passed |
| Product code delta | NONE |
| PT-Q02 | PENDING |
| Independent PASS / FRZ promotion | NONE |

**Outside-scope inventory (record only):** AWS/Azure/GCP cloud platform adapter class `slug` fields.

**RED-first note:** F-A2 scratch RED at `2f38f0031b0e9c1d5b1aba1df7db8b3dfaf31809` preserved; formal NT-FC3 RED at baseline SHA not re-recorded in this pass (GREEN qualification executed on authorized implementation tree).

**Lifecycle:** implementation evidence recorded; approval line remains Operator-only; not a PT-Q02 PASS.

## Checkpoint — AF-W3-PT-FC3-F-CORRECTIVE-01 (audit findings F01–F08, 2026-11-28)

Session: `INTEGRAX | ARCH-FREEZE | W3 | PT-FC3-F | AF-W3-PT-FC3-F-CORRECTIVE-01`. Operator contract: close independent review findings **F01–F08** in the scoped CONFIG-X FC3 detector and bounded tests only; no product-path edits.

| Finding | Corrective |
|---|---|
| F01 | Structural `LabSelectorOptIn` annotation; strict `If` true-branch guard (no whole-`If` `ast.walk` else bleed) |
| F02 | `VALIDATION_ONLY` via AST comparison/error context, not path suffix |
| F03 | `VisionInferenceAdapter` via import bindings, not classname substring |
| F04 | `Counter` multiset in `evaluate_fc3_scoped_gate` / pure `evaluate_fc3_scoped_gate_from_inventory` |
| F05 | Cross-module `_LAB_*` import flags |
| F06 | CX4/CX8/CX11/global scan/Null verifier behavioral completion |
| F07 | `derive_fc3_vendor_values` child-process catalog snapshot (no parent catalog pollution) |
| F08 | Per-scan `_Fc3ScanContext`; enum module bindings; env fallback chains |

| Verification | Result |
|---|---|
| `test_config_x_fc3_vendor_slug_defaults.py` | 28 passed |
| `tests/qualification/config_x` | 105 passed |
| Product delta | NONE |
| PT-Q02 / FRZ promotion | NONE |

**Note:** RED corrective assertions landed before detector GREEN in separate commits on `development`; approval line unchanged (Operator-only).


## Checkpoint — AF-W3-PT-FC3-F-CORRECTIVE-02 (2026-11-28)

Execution session: INTEGRAX | ARCH-FREEZE | W3 | PT-FC3-F | AF-W3-PT-FC3-F-CORRECTIVE-02.
SESSION_NAME_APPLIED=NO: cursor-app-control.rename_chat unavailable; the Codex equivalent
returned a truncated title, confirmed by read_thread. The exact required name was provided
inline for manual setting; no exact-title application is claimed.
Base: 36ff100d313ddb5e209d3b3e4f9901d728208e6e. Authorization: OD-AF-W3-FC3-F-01 variant A.
Implementation/qualification evidence only; personal approval PENDING, lifecycle draft, PT-Q02 PENDING.

The Operator explicitly authorized one exception after the first RED: correct only the overly strict
global AST equality assertion, preserve semantics and Pyright 0, and document new RED/GREEN.
The corrected test normalizes exactly the pre-existing typing-only const_ref guard and compares all
30 original function ASTs. The guard is semantically identical because unresolved is returned only
for ast.Name; the test also checks this fact. All other assertions are unchanged.

Initial RED: db82ca86db0b846a3e5aa8c1925a9a2f542cbeb0; test blob
72d9b2408c1392125c4ef4dc51f13c1426b95736; exit 1; 49 failed, 63 passed;
log SHA256 8fee8f696ebbe85ab596da7439412169327f6601b2c2386022f3d61ab8407031.
An earlier collection SyntaxError was discarded and is not mechanism-aligned RED evidence.

Refreshed RED: 0682143083d107e5afcd6e28a5601334a4445cab; test blob 26eaa88f3a439b1f0e083fa81fa98c1e01beaf46; exit 1;
log SHA256 fb0305c5c0e9707e7e1caf461538af8d2b6417752e3ed69b978eb18090b21f5c.
RED started at 0682143083d107e5afcd6e28a5601334a4445cab; ended at 367cbbabd3264e329d411f043e675c2dd08a3c5e after a concurrent foreign
adapter setup commit. The CONFIG-X tree is identical across both SHAs (git diff exit 0).
JUnit start timestamp 2026-11-28T11:16:17.509563+01:00; foreign commit timestamp 2026-11-28T11:19:48+01:00.
The original post-execution SHA record is retained; its start/end identity was explicitly settled.
GREEN execution: 460982a4e347923927fdd5d3c2a03727153f4954; test blob 26eaa88f3a439b1f0e083fa81fa98c1e01beaf46; exit 0;
log SHA256 4db3735a3f87ac4d3582a83a4eb8b4c98e8d6ad238fca83657b56c9893a99acf. Test blob identity: TRUE.
Full batch: 226 passed, 0 failed, 0 skipped.
Ruff exit=0; Ruff format --check exit=0;
Pyright exit=0 (0 errors, 0 warnings, 0 informations).
All tests used D:/Projekty/_wt/_locks/pytest.lock, serial execution.

Actual RED/GREEN summaries:

```text
================== 48 failed, 64 passed in 358.67s (0:05:58) ==================
================= 226 passed, 2 warnings in 487.05s (0:08:07) =================
```

Actual RED assertions:

```text
test_corrective_lab[LAB-08]
AssertionError: []
assert not [] is False

test_corrective_validation[VAL-06]
AssertionError: [Fc3VendorDefaultFinding(repo_path='synthetic/validation.py', enclosing_symbol='f', semantic_field='_LAB_X', expression='_LAB_X', normalized_expression='_LAB_X', resolved_vendor_value='inmemory', detection_mechanism='lab_named_constant', disposition=<Fc3VendorDisposition.VALIDATION_ONLY: 'validation_only'>, source_line=5, detail='validation_only_lab_load')]
assert False
 +  where False = all(<generator object test_corrective_validation.<locals>.<genexpr> at 0x0000029CB0E0F440>)

test_corrective_identity[ID-05]
assert True is False
 +  where True = any(<generator object test_corrective_identity.<locals>.<genexpr> at 0x0000029CB0B9C2B0>)

test_corrective_lab_imports[LABIMPORT-03]
AssertionError: ()
assert 0 == 1
 +  where 0 = len(())

test_corrective_environment[ENV-02]
AssertionError: assert [] == [('f', 'provi...ing_literal')]

  Right contains one more item: ('f', 'provider', 'ollama', 'env_get_default:string_literal')

  Full diff:
  + []
  - [
  -     (...

  ...Full output truncated (6 lines hidden), use '-vv' to show
```

MULTI-07 originally raised SyntaxError from the deliberately invalid input source;
that exception is not claimed as mechanism-aligned assertion RED. Its final acceptance
executes the same evaluator and explicitly checks ok is False. Counter was already
implemented and the baseline GREEN controls are retained without fabricated RED.

### Mandatory acceptance cases (87)

| Case | Exact pytest node ID | Expected | Actual | Result |
|---|---|---|---|---|
| LAB-01 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-01] | Exempt | Exempt | PASS |
| LAB-02 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-02] | Refused | Refused | PASS |
| LAB-03 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-03] | Exempt | Exempt | PASS |
| LAB-04 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-04] | Refused | Refused | PASS |
| LAB-05 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-05] | Exempt | Exempt | PASS |
| LAB-06 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-06] | Refused | Refused | PASS |
| LAB-07 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-07] | Refused | Refused | PASS |
| LAB-08 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-08] | Refused | Refused | PASS |
| LAB-09 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-09] | Refused | Refused | PASS |
| LAB-10 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-10] | Refused | Refused | PASS |
| LAB-11 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-11] | Refused | Refused | PASS |
| LAB-12 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-12] | Refused | Refused | PASS |
| LAB-13 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab[LAB-13] | Exempt | Exempt | PASS |
| VAL-01 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_validation[VAL-01] | VALIDATION_ONLY | VALIDATION_ONLY | PASS |
| VAL-02 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_validation[VAL-02] | VALIDATION_ONLY | VALIDATION_ONLY | PASS |
| VAL-03 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_validation[VAL-03] | Refused | Refused | PASS |
| VAL-04 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_validation[VAL-04] | Refused | Refused | PASS |
| VAL-05 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_validation[VAL-05] | Refused | Refused | PASS |
| VAL-06 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_validation[VAL-06] | Refused | Refused | PASS |
| VAL-07 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_validation[VAL-07] | Refused | Refused | PASS |
| VAL-08 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_validation[VAL-08] | Refused | Refused | PASS |
| VAL-09 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_validation[VAL-09] | Refused | Refused | PASS |
| VAL-10 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_validation[VAL-10] | Refused | Refused | PASS |
| VAL-11 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_validation[VAL-11] | Refused | Refused | PASS |
| ID-01 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_identity[ID-01] | Identity | Identity | PASS |
| ID-02 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_identity[ID-02] | Identity | Identity | PASS |
| ID-03 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_identity[ID-03] | Identity | Identity | PASS |
| ID-04 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_identity[ID-04] | Identity | Identity | PASS |
| ID-05 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_identity[ID-05] | Refused | Refused | PASS |
| ID-06 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_identity[ID-06] | Refused | Refused | PASS |
| ID-07 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_identity[ID-07] | Refused | Refused | PASS |
| ID-08 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_identity[ID-08] | Refused | Refused | PASS |
| ID-09 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_identity[ID-09] | Refused | Refused | PASS |
| ID-10 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_identity[ID-10] | provider=ollama detected | provider=ollama detected | PASS |
| MULTI-01 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_multiset[MULTI-01] | ok=True | ok=True | PASS |
| MULTI-02 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_multiset[MULTI-02] | ok=False | ok=False | PASS |
| MULTI-03 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_multiset[MULTI-03] | ok=False | ok=False | PASS |
| MULTI-04 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_multiset[MULTI-04] | ok=False | ok=False | PASS |
| MULTI-05 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_multiset[MULTI-05] | ok=False | ok=False | PASS |
| MULTI-06 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_multiset[MULTI-06] | ok=False | ok=False | PASS |
| MULTI-07 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_multiset[MULTI-07] | ok=False | ok=False | PASS |
| LABIMPORT-01 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab_imports[LABIMPORT-01] | LAB_REFUSED; ollama; resolved | LAB_REFUSED; ollama; resolved | PASS |
| LABIMPORT-02 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab_imports[LABIMPORT-02] | LAB_REFUSED; ollama; resolved | LAB_REFUSED; ollama; resolved | PASS |
| LABIMPORT-03 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab_imports[LABIMPORT-03] | LAB_REFUSED; ollama; resolved | LAB_REFUSED; ollama; resolved | PASS |
| LABIMPORT-04 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab_imports[LABIMPORT-04] | LAB_REFUSED; ollama; resolved | LAB_REFUSED; ollama; resolved | PASS |
| LABIMPORT-05 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab_imports[LABIMPORT-05] | LAB_REFUSED; unresolved | LAB_REFUSED; unresolved | PASS |
| LABIMPORT-06 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab_imports[LABIMPORT-06] | __all__ export refused | __all__ export refused | PASS |
| LABIMPORT-07 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab_imports[LABIMPORT-07] | unresolved outside-scope included | unresolved outside-scope included | PASS |
| LABIMPORT-08 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_lab_imports[LABIMPORT-08] | one unguarded load refused | one unguarded load refused | PASS |
| CAT-01 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_catalog[CAT-01] | custom entry preserved | custom entry preserved | PASS |
| CAT-02 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_catalog[CAT-02] | full snapshot identical | full snapshot identical | PASS |
| CAT-03 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_catalog[CAT-03] | revision/generation unchanged | revision/generation unchanged | PASS |
| CAT-04 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_catalog[CAT-04] | bootstrap flag unchanged | bootstrap flag unchanged | PASS |
| CAT-05 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_catalog[CAT-05] | parent state preserved on child failure | parent state preserved on child failure | PASS |
| CAT-06 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_catalog[CAT-06] | deterministic repeated derivation | deterministic repeated derivation | PASS |
| CAT-07 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_catalog[CAT-07] | state preserved in actual test order; reset fixture absent | state preserved in actual test order; reset fixture absent | PASS |
| ENV-01 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-01] | f/provider/ollama/env_get_default:string_literal | f/provider/ollama/env_get_default:string_literal | PASS |
| ENV-02 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-02] | f/provider/ollama/env_get_default:string_literal | f/provider/ollama/env_get_default:string_literal | PASS |
| ENV-03 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-03] | f/provider/ollama/env_get_default:string_literal | f/provider/ollama/env_get_default:string_literal | PASS |
| ENV-04 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-04] | f/provider/ollama/env_get_default:string_literal | f/provider/ollama/env_get_default:string_literal | PASS |
| ENV-05 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-05] | f/provider/ollama/env_get_default:string_literal | f/provider/ollama/env_get_default:string_literal | PASS |
| ENV-06 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-06] | f/provider/ollama/env_or_fallback:string_literal | f/provider/ollama/env_or_fallback:string_literal | PASS |
| ENV-07 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-07] | f/provider/ollama/env_or_fallback:string_literal | f/provider/ollama/env_or_fallback:string_literal | PASS |
| ENV-08 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-08] | C/provider/ollama/env_get_default:string_literal | C/provider/ollama/env_get_default:string_literal | PASS |
| ENV-09 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-09] | <module>/provider/ollama/env_get_default:string_literal | <module>/provider/ollama/env_get_default:string_literal | PASS |
| ENV-10 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-10] | f/provider/ollama/env_get_default:string_literal | f/provider/ollama/env_get_default:string_literal | PASS |
| ENV-11 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-11] | f/mode/none/positional_default:string_literal | f/mode/none/positional_default:string_literal | PASS |
| ENV-12 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-12] | f/channel/stub/keyword_only_default:string_literal | f/channel/stub/keyword_only_default:string_literal | PASS |
| ENV-13 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-13] | no finding | no finding | PASS |
| ENV-14 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-14] | no finding | no finding | PASS |
| ENV-15 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-15] | no canonical enum finding | no canonical enum finding | PASS |
| ENV-16 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-16] | f/mode/none/positional_default:enum_member | f/mode/none/positional_default:enum_member | PASS |
| ENV-17 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-17] | ollama local enum; second module unresolved | ollama local enum; second module unresolved | PASS |
| ENV-18 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-18] | backend=log; channel=stub; <return:4>=none | backend=log; channel=stub; <return:4>=none | PASS |
| ENV-19 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_corrective_environment[ENV-19] | one outer.inner/provider/ollama finding | one outer.inner/provider/ollama finding | PASS |
| NT-FC3-CX1 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx1_os_getenv_vendor_default | environment default detected | environment default detected | PASS |
| NT-FC3-CX2 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx2_enum_class_field_default | enum field detected | enum field detected | PASS |
| NT-FC3-CX3 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx3_os_environ_get_enum_value | enum environment default detected | enum environment default detected | PASS |
| NT-FC3-CX4 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx4_module_level_selector_scanned | resolve_settings/provider/ollama/env_get_default:string_literal | resolve_settings/provider/ollama/env_get_default:string_literal | PASS |
| NT-FC3-CX5 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx5_semantic_parameter_defaults_including_positional_only | mode=none; backend=log; channel=stub; required provider absent | mode=none; backend=log; channel=stub; required provider absent | PASS |
| NT-FC3-CX6 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx6_catalog_slug_vs_non_slug | provider ollama detected; provider non-vendor absent | provider ollama detected; provider non-vendor absent | PASS |
| NT-FC3-CX7 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx7_repo_scoped_gate_expected_residual_inventory | ok=True; exact three residuals | ok=True; exact three residuals | PASS |
| NT-FC3-CX8 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx8_synthetic_embedding_vendor_default_refused | mutated real profile.py; evaluator ok=False | mutated real profile.py; evaluator ok=False | PASS |
| NT-FC3-CX9 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx9_guarded_lab_constant_not_blocking | Exempt | Exempt | PASS |
| NT-FC3-CX10 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx10_unguarded_lab_constant_blocking | Refused | Refused | PASS |
| NT-FC3-CX11 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx11_mixed_guarded_and_unguarded_lab_loads | unguarded local load and actual cross-module import refused | unguarded local load and actual cross-module import refused | PASS |
| NT-FC3-CX12 | tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py::test_nt_fc3_cx12_unguarded_default_in_lab_named_module | stub detected without path exemption | stub detected without path exemption | PASS |

### Qualification and outside scope

Scoped gate ok=True; expected residual multiset match=True;
three residuals, unexpected=0. No expected-residual expansion.
Global semantic-I blockers=[]; named-constant blind spots=[];
unclassified provider surfaces=[].

| Path:line | Symbol | Field | Expression | Value | Mechanism | Disposition |
|---|---|---|---|---|---|---|
| intergrax/integrations/providers/cloud_platform/aws/adapter.py:29 | _AwsCloudPlatform | slug | "aws" | aws | class_field_default:string_literal | outside_scope_record |
| intergrax/integrations/providers/cloud_platform/azure/adapter.py:30 | _AzureCloudPlatform | slug | "azure" | azure | class_field_default:string_literal | outside_scope_record |
| intergrax/integrations/providers/cloud_platform/gcp/adapter.py:30 | _GcpCloudPlatform | slug | "gcp" | gcp | class_field_default:string_literal | outside_scope_record |

The original regression_nodes_F.txt was not found in the tracked repository; the canonical obligations
map to these actual executed node IDs (all passed):

- tests/qualification/config_x/test_config_x_final_reconciliation_gates.py::test_frz_cfg_02_missing_integration_config_fails_closed
- tests/qualification/config_x/test_config_x_final_reconciliation_gates.py::test_frz_cfg_05_semantic_production_selection_closed_world_zero
- tests/qualification/config_x/test_config_x_final_reconciliation_gates.py::test_frz_cfg_06_activation_families_no_bypasses
- tests/qualification/config_x/test_config_x_final_reconciliation_gates.py::test_frz_cfg_08_regression_protects_current_closed_world_classification
- tests/qualification/config_x/test_config_x_qualification_gates.py::test_cx_q01_concern_inventory_closed_world
- tests/qualification/config_x/test_config_x_qualification_gates.py::test_cx_q16_mechanical_concern_classification_covers_inventory
- tests/qualification/config_x/test_config_x_qualification_gates.py::test_cx_q18_frz_cfg_05_semantic_selection_gate
- tests/qualification/config_x/test_config_x_r1_r1_r1_semantic_constant_resolution.py::test_named_constant_localhost_resolves_to_sanctioned_on_vector_store_surface
- tests/qualification/config_x/test_config_x_r1_r1_r1_semantic_constant_resolution.py::test_named_constant_localhost_unsanctioned_surface_is_hard_coded_selection
- tests/qualification/config_x/test_config_x_r1_r1_r1_semantic_constant_resolution.py::test_unresolved_named_constant_semantic_default_is_blind_spot
- tests/qualification/config_x/test_config_x_adversarial_cx_gates.py::test_cx_a_missing_required_integration_config_fails_closed

NT-CFG-08: tests/unit/architecture/test_nt_ent_iso_red_first_probes.py::test_nt_cfg_08_embedding_profile_must_not_hardcode_provider_getenv_default — passed.
The final batch included complete CONFIG-X plus inbound, embedding, vectorstore and vision behavioral controls.

### Governance integrity

Previous current ticket digest: 204146325aaf01b8a1eda7e391b40f4f1dea2d4a4706078bcc87b69a10e285f9; preserved here as history.
Actual plan digest before BOM removal: 66ee11930a45177ced36e4a4fdf8d9acfa973a57fb3e713e2a8c64b96d32c101.
Current plan digest: b434679cdc9be69d755cda28234a4e543605cd5122ae617800c0f70456dc2707. Only BOM removed; amendments retained, no new plan amendment.
Two real build_rw_plan.py runs: semantic reproducibility=True.
Canonical owner/negative bindings, merge order, hard predecessors, serialization and OP-B inventory retained.
Historical generator metadata correction: [{"frz_id": "FRZ-BND-06", "before": [{"nt_id": "NT-BND-06-1", "node": "tests/unit/architecture/test_rw_gates_transport_semantics.py", "mechanism": "A hosting, queue, MCP or HTTP transport adapter that constructs or interprets a RuntimeEvent turns the transport-semantics gate red. GREEN with AF-W3-PT-FC3-F-CORRECTIVE-01 (28 tests; F01–F08 closed) The versioned plan v21 row table assigns this file to node T-AF2-RW-GATES; the matrix inventory owner is T-IGX-043-J. The file is not in the write set extracted for T-IGX-043-J; the owner plan has to add it.", "owner_ticket": "T-IGX-043-J", "file_state_at_pin": "exists"}], "after": [{"nt_id": "NT-BND-06-1", "node": "tests/unit/architecture/test_rw_gates_transport_semantics.py", "mechanism": "A hosting, queue, MCP or HTTP transport adapter that constructs or interprets a RuntimeEvent turns the transport-semantics gate red. GREEN with AF-W3-PT-FC3-F-CORRECTIVE-01 (28 tests; F01–F08 closed) The versioned plan v21 row table assigns this file to node T-AF2-RW-GATES; the matrix inventory owner is T-IGX-043-J. The file is not in the write set extracted for T-IGX-043-J; the owner plan has to add it.", "owner_ticket": "T-IGX-043-J", "file_state_at_pin": "absent"}]}].

Source digests checked=38;
matched=37;
mismatched=1.
Unauthorized historical mismatch not refreshed:
- continuation/RED_FIRST_GA2R_RESULT.md: expected 9adf3e0b09a2dc8aa60663afe523cae76d4c6adefba79cf4a16028795e361cae; actual b9cdc341832a00810afdbb51a7de14a0277728b9a8c48df750fdeb2e5e927c2f

phase check3 exit=0; pre-G2 exit=1 (11 OP-B);
traceability exit=0; igx_gov ticket exit=0.
Governance console recovery: Both builds completed and governance.py passed first==second (excluding generated_at), then completed all child validators. Console printing of ticket output raised UnicodeEncodeError after its actual exit 0 and after persisting the raw log. No validator/test was rerun for recovery.

```text
phase:
PLAN_STRUCTURE=PASS
CODE_FIX_OWNERSHIP=39/39
NAMED_NEGATIVE_TESTS=39/39
RED_FIRST_EVIDENCE=INCOMPLETE
RW_PLAN_AUDIT=NOT_PERFORMED
CHECK3_STRUCTURE_GREEN=YES

pre_g2:
PRE_G2_OP_B_SETTLEMENT=FAIL
PLAN_STRUCTURE=PASS
CODE_FIX_OWNERSHIP=39/39
NAMED_NEGATIVE_TESTS=39/39
RED_FIRST_EVIDENCE=INCOMPLETE
RW_PLAN_AUDIT=NOT_PERFORMED
CHECK3_STRUCTURE_GREEN=YES

traceability:
[OK] verify_arch_freeze_plan_traceability: all checks passed

ticket:
PASSED job=push-fast (3.9s)
PASS  push-fast                     exit=0  3.9s
[INFO] igx_gov ticket: OK
```

Historical FRZ-CFG-08 red-first and F-A2 artifact identity mismatch retained; F-A2 was not repeated.
Personal approval belongs to governance/tickets/T-IGX-043-F.md, section ## Approved,
format: Approved: operator @ <ISO8601 timestamp with timezone offset>. No signature written.

Preserved state: RED_AT_PREFIX=22; EVIDENCE_MISSING=11; RED_FIRST_OUT=5;
RED_FIRST_LATER_WAVE=1; OP_B_PENDING=11; RED_FIRST_EVIDENCE=INCOMPLETE;
FRZ_EXE_02=EVIDENCE_MISSING; FRZ_PLG_05=EVIDENCE_MISSING; STRICT_HOST_AUTHORITY=OPEN_RESIDUAL;
SCOPE=CLOSED; PT_Q02=PENDING; G2=OPEN; RB2=NOT_PERFORMED; FRZ_PROMOTION=NONE.

MANUAL_ATTACHMENT_REQUIRED=NO. No independent PASS or FRZ promotion is claimed.
