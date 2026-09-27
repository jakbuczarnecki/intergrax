# GR-11 — Governance Plugin Enterprise Certification

**Status:** `READY FOR AUDIT`  
**Parent:** GOV-X1 (CURRENT)  
**Mechanical SSOT:** `tests/qualification/governance/gr11/catalog.py`  
**Mechanical gate:** `tests/qualification/governance/gr11/test_gr11_plugin_enterprise_certification.py`

GR-11 certifies **Governance-relevant extensibility** on a closed-world nine-row inventory. It does **not** claim whole-platform plugin certification (EBH-5 / CONFIG-X / COMPAT-X / PROD-Q).

## Qualification baseline

| Field | Value |
| ----- | ----- |
| START_HEAD (pre-GR-11-R1 wave) | `a7f575a7fe3eea37df28eef73336db1eb276d7e3` |
| Pre-audited operator pin (no GR-11 delta) | `53c01772914b525f938e36b1a3d32bf86d86fe0d` |
| GR-12 accepted evidence SHA (historical dependency) | `03dde6c68a37ac0a8fe19cc5bcf683da8a3afc06` |
| GR-12 semantic baseline | `b706c2c72a900575ec360b7f217c97a3656c71b9` |
| Qualification bundle commit | set by GR-11-R1 cohesive commit on `development` |

Parallel-session delta vs `53c01772914b525f938e36b1a3d32bf86d86fe0d`: UCA scenario documentation only — **no GR-11 / Governance plugin semantic change**.

## Evidence model (GR-11-R1)

Each `QUALIFIED` row binds four **typed** proof-node tuples (not catalog metadata alone):

| Category | Meaning |
| -------- | ------- |
| `structural_replaceability_proof_nodes` | Custom/external implementation through canonical contract seam |
| `authority_proof_nodes` | What the extension may decide / own |
| `composition_proof_nodes` | Sanctioned composition owner wires implementation |
| `negative_bypass_proof_nodes` | Fail-closed, deny, or non-authority guarantees |

Mechanical ownership per row:

- `semantic_owner_module` — contract or semantic owner module defining the port/policy
- `composition_owner_module` — sanctioned composition entrypoint
- `consumer_scan_modules` — production consumers scanned for implementation branching and port redeclaration

### Mechanical ownership evidence (GR-11-R2)

**G07 (semantic uniqueness, closed world):** For each contract segment, AST proves the canonical `class` lives in the segment’s `intergrax.*` defining module; among `semantic_owner_module`, `composition_owner_module`, and `consumer_scan_modules` only that defining module may declare the symbol (G09 reuses the same consumer redeclaration scan).

**G08 (composition ownership):** Positive proof uses **only** `composition_owner_module` plus **delegate modules** directly imported and invoked from its `build_*` / `wire_*` entrypoints. Accepted evidence is constructor injection, `build_*` / `wire_*` parameter or return wiring, or direct `Symbol(...)` construction inside those entrypoints — not a module-wide AST name reference. A symbol appearing only in an unrelated consumer or as an incidental reference cannot satisfy the gate. Consumers do not **compose** the contract (`build_*` / `wire_*` returning or constructing the port) unless they delegate via calls imported from the declared composition owner. `plugin_spi` reconciliation symbols may additionally be evidenced by a `consumer_scan_modules` registry resolution return type when the composition owner exposes a `build_*` entrypoint.

## Closed-world inventory (9 rows)

| ID | Contract | Semantic owner | Composition owner | Default impl |
| -- | -------- | -------------- | ----------------- | ------------ |
| GR11-ROOT-ADMISSION | `RuntimeExecutionPolicyAdmissionPort` | Governance / root admission | `execution_admission_composition` | Allowing/Denying admission |
| GR11-INNER-GUARD | `CanonicalInnerExecutionGuardPort` | Governance / inner enforcement | MSE authorization composition | `DefaultCanonicalInnerExecutionGuard` |
| GR11-DECISION-REQUIREMENT | `DecisionRequirementPolicy` | Governance / Decision-bound MSE | MSE authorization composition | Permissive/Configured policies |
| GR11-RUNTIME-POLICY-EVALUATOR | `RuntimePolicyEngine` (MSE seam) | Governance / runtime policy | `execution_admission_composition` | Default rule bundles |
| GR11-CONTROL-PLANE-EVALUATOR | `ControlPlaneMutationPolicyEvaluator` | Governance / CLA-04 boundary | CLA-04 authorization boundary | Bundle-backed evaluator |
| GR11-CONTINUATION | `ExecutionContinuationPort` | **Execution Runtime** | suspended-operation composition | Execution-owned stores |
| GR11-PROVIDER-INVOCATION-STORE | `ProviderInvocationStore` | Enterprise Reliability facts | governed external-work production runtime | In-memory store default |
| GR11-PROVIDER-RELIABILITY-COLLABORATOR | admission + reconciliation SPI | Governance host + Reliability | governed external-work composition | Injectable bridge defaults |
| GR11-RELIABILITY-OBSERVATION-EVIDENCE | `ProviderInvocationReliabilityEvidenceObserver` | Reliability evidence projection | reliability evidence emit path | Null observer |

Full proof-node lists and scan module paths: `GR11_EXTENSION_SURFACES` in catalog SSOT.

### Continuation (mandatory GR-11-R1 correction)

- **Structural replaceability:** `tests/unit/contracts/test_execution_continuation.py::test_pluginability_two_implementations[...]` (two implementations).
- **Authority (not structural):** `test_mp4r3_no_duplicate_continuation_lifecycle_authority`.
- **Composition:** `test_mp4r3_contract_only_continuation_dependency`.

## Historical reconciliation (GOV-FINAL-4 pluginability)

| Capability | Before | After (mechanical) |
| ---------- | ------ | ------------------ |
| Continuation | PARTIAL | QUALIFIED — two-implementation pluginability + MP-4R3 gates |
| Provider integration | PARTIAL | QUALIFIED — typed admission + reconciliation collaborators |
| ProviderInvocationStore | PARTIAL | QUALIFIED — custom durable store via production composition |
| Reliability observer | PARTIAL | QUALIFIED — optional non-authoritative evidence sink |

## Weak-boundary scan

**Scope:** `GR11_WEAK_BOUNDARY_SCAN_MODULES` (contracts + runtime seams + governed external-work composition).

**Method:** AST on semantic extension contract classes (`Port` / `Protocol` / `Store`); forbid `Any`, bare `object`, `dict[str, Any]`, `Mapping[str, Any]`, reflection calls on contract modules. Factory helpers on non-port models are excluded from port-method scan.

**Result (current HEAD):** 0 semantic-boundary blockers in scope.

## Implementation-branch scan

**Scope:** `GR11_IMPLEMENTATION_BRANCH_SCAN_MODULES` (union of semantic, composition, and consumer modules for all nine rows).

**Forbidden patterns:** `isinstance(..., Default*)`, plugin/provider name equality branches, `plugin_name ==`, `provider == "..."`.

**Result (current HEAD):** 0 matches in scope.

## Dynamic registration

All Governance permission seams: `COMPOSITION_TIME_ONLY`. Composition modules scanned for self-registration markers (`register_plugin`, `auto_discover`, dynamic import). Reliability observer: `NOT_APPLICABLE` (optional sink).

## Authority summary (bound to proof nodes)

| Class | May | Must not |
| ----- | --- | -------- |
| Governance permission plugins | Contract decision results | Execute work, mint execution identity, self-expand scope, override stricter authority |
| Decision requirement | Material prerequisites | Grant execution permission, convert Decision → ALLOW alone |
| Execution continuation | Pause/resume lifecycle | Governance permission |
| ProviderInvocationStore | Durable invocation facts | Governance permission |
| Reliability observer | Observe/project facts | Grant permission, mutate execution truth |

Fail-closed admission regression nodes remain registered (unconfigured / unavailable evaluator).

## Targeted verification (GR-11-R1)

```text
uv run --frozen pytest tests/qualification/governance/gr11/ -q -p no:xdist
uv run --frozen pytest tests/qualification/governance/ -q -p no:xdist  # ×2, same count
uv run --frozen pyright tests/qualification/governance/gr11/catalog.py tests/qualification/governance/gr11/test_gr11_plugin_enterprise_certification.py
uv run --frozen ruff check <changed gr11 files>
uv run --frozen ruff format --check <changed gr11 files>
git diff --check
```

## FRZ scoped evidence (global criteria remain OPEN)

| FRZ | Scoped contribution |
| --- | ------------------- |
| FRZ-PLG-01..08 | Typed plugin replaceability + composition proof per row |
| FRZ-RPL-01..04 | Provider store + reliability collaborator + observer rows |
| FRZ-GOV-01..04,06,09 | Permission extension authority + fail-closed admission |
| FRZ-CTR-01,02,05,06 | Contract port weak-boundary scan |
| FRZ-TYP-01..06 | AST annotation gate on contract ports (as applicable) |
| FRZ-REG-02,03,06,09 | Governance qualification regression batches |

No global FRZ PASS claimed.

## Unresolved findings

| Finding | Class |
| ------- | ----- |
| — | IN-SCOPE BLOCKER: **0** |

## Recommended status

`GR-11-R2 = READY FOR AUDIT` · `GR-11 = READY FOR AUDIT` · `GOV-X1 = CURRENT` · `GR-13 = BLOCKED` pending independent exact-SHA GR-11 audit/closure.

**Not claimed:** GR-11 CLOSED · GOV-X1 CLOSED · Governance Plane enterprise CLOSED · FRZ-* global PASS.
