# INTEGRAx-FINAL-BASELINE-SELECTION

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-FINAL-BASELINE-SELECTION` |
| **Date** | 2026-09-13 |
| **Branch** | `development` |
| **Repository HEAD (at selection capture)** | `17db424fcf2369b1de0de368489a2d96e24564e5` |
| **Working tree (at selection capture)** | **clean** (no staged/unstaged tracked changes) |

## Baseline semantics

**Certified code baseline** = last **production-semantic** commit in the fully audited chain (`a185403d0c7524c29bea2fe09212f9508e6bccd8`).

**Certification / evidence HEAD** = latest commit whose primary purpose is qualification, evidence, or documentation (`17db424fcf2369b1de0de368489a2d96e24564e5` at capture; advanced by this selection record commit).

**Current repository HEAD** = tip of `development` after this task’s commit (see git log).

Post-`a185403d0…` commits do **not** advance the certified **code** baseline unless they introduce new unaudited production semantics (none found).

---

## Certified code baseline

```text
Certified code baseline SHA:
a185403d0c7524c29bea2fe09212f9508e6bccd8
```

Message: `feat(execution): EE-B1.1 reliability contracts and failure semantics`

---

## Evidence HEAD (at capture, pre-selection commit)

```text
Certification/evidence HEAD:
17db424fcf2369b1de0de368489a2d96e24564e5
```

Message: `INTEGRAx-EXECUTION-RELIABILITY-CONTRACT-SCOPED-RECERTIFICATION`

---

## Current HEAD (at capture)

```text
Current repository HEAD:
17db424fcf2369b1de0de368489a2d96e24564e5
```

---

## Commit inventory (after `a185403d0…`)

| SHA | Message | Production? | Type | Audited? | Baseline impact |
| --- | ------- | ----------: | ---- | -------: | --------------- |
| `524ce5b9bf84b47398f3065fb58e6e2ab8848c8c` | INTEGRAx-CLEAN-BASELINE-FREEZE-READINESS | No (semantic) | **MIXED** (qualification + SSOT + **FORMAT_ONLY PRODUCTION FILE CHANGE**) | N/A (non-semantic) | Does **not** move code baseline; resolves ruff format drift on tracing + EE-B1.1 files already in `a185403d0…` |
| `4138c1e0a080372668b3894b7dfa583c9ef62f14` | INTEGRAx-NEXUS-COMPOSITION-OWNERSHIP-SCOPED-AUDIT | No | **QUALIFICATION** | Audits `fef16c3b9…` | Evidence only |
| `17db424fcf2369b1de0de368489a2d96e24564e5` | INTEGRAx-EXECUTION-RELIABILITY-CONTRACT-SCOPED-RECERTIFICATION | No | **QUALIFICATION** | Audits `a185403d0…` | Evidence only; confirms baseline eligibility |

### FORMAT_ONLY PRODUCTION FILE CHANGE (`524ce5b9…`)

Verified on diff (not message): line wrapping only in:

- `intergrax/contracts/tracing/__init__.py`
- `intergrax/contracts/tracing/values.py`
- `intergrax/contracts/execution_reliability/failure_classification_contract.py`
- `intergrax/runtime/execution/reliability/default_failure_classifier.py`

---

## Audited production chain (baseline body)

| SHA | Type | Production semantics? | Audited? | Verdict (SSOT) |
| --- | ---- | --------------------: | -------: | -------------- |
| `118798759e8a198b9a1d21ecd93293fe601fd7d9` | PRODUCTION | Yes | Yes | **ACCEPT WITH OBSERVATIONS** — [`INTEGRAX_POST_BASELINE_COMMITS_RECONCILIATION.md`](INTEGRAX_POST_BASELINE_COMMITS_RECONCILIATION.md) |
| `fef16c3b951401e2222bc81769442f7150cad9fc` | PRODUCTION | Yes | Yes | **ACCEPT WITH OBSERVATIONS** (CLASS B) — [`INTEGRAX_NEXUS_COMPOSITION_OWNERSHIP_SCOPED_AUDIT.md`](INTEGRAX_NEXUS_COMPOSITION_OWNERSHIP_SCOPED_AUDIT.md) |
| `a185403d0c7524c29bea2fe09212f9508e6bccd8` | PRODUCTION | Yes | Yes | **SCOPED RECERTIFIED WITH OBSERVATIONS** (CLASS C) — [`INTEGRAX_EXECUTION_RELIABILITY_CONTRACT_SCOPED_RECERTIFICATION.md`](INTEGRAX_EXECUTION_RELIABILITY_CONTRACT_SCOPED_RECERTIFICATION.md) |

No unaudited production commit exists after `a185403d0…`.

---

## Chain continuity

```bash
git merge-base --is-ancestor 118798759e8a198b9a1d21ecd93293fe601fd7d9 a185403d0c7524c29bea2fe09212f9508e6bccd8  # exit 0 → YES
git merge-base --is-ancestor fef16c3b951401e2222bc81769442f7150cad9fc a185403d0c7524c29bea2fe09212f9508e6bccd8  # exit 0 → YES
```

**Result:** **PASS**

---

## Architecture invariants (INV-1 … INV-10)

Sourced from scoped qualification records on the audited chain; not re-opened as full architecture audit.

| Invariant | Result |
| --------- | ------ |
| INV-1 Decision ≠ Execution | **PASS** |
| INV-2 Governance fail-closed | **PASS** |
| INV-3 Canonical Execution owner | **PASS** |
| INV-4 No parallel runtime | **PASS** |
| INV-5 Contracts first | **PASS** |
| INV-6 Plugin extensibility | **PASS** |
| INV-7 Persistence abstraction | **PASS** |
| INV-8 Identity authority | **PASS** |
| INV-9 Evidence ≠ control | **PASS** |
| INV-10 Qualification ≠ production | **PASS** |

---

## Reliability invariants (REL-1 … REL-8)

EE-B1.1 scoped recertification @ `a185403d0…`:

| ID | Result |
| --- | ------ |
| REL-1 reliability does not own execution | **PASS** |
| REL-2 no second retry engine | **PASS** |
| REL-3 no local recovery owner | **PASS** |
| REL-4 no persistence ownership takeover | **PASS** |
| REL-5 mandatory evidence fail-closed | **PASS** |
| REL-6 integrity fail-closed | **PASS** |
| REL-7 shutdown contract only | **PASS** |
| REL-8 classifier replaceable / provider-neutral | **PASS** |

---

## Nexus ownership

**shared wiring ≠ concrete Nexus owner:** **PASS** (NPSC-4.2 scoped audit @ `fef16c3b9…`).

**Raw `NexusLoop` imports:** remain only at approved composition owners; U2 compensation / child wiring no longer imports `NexusLoop` (gate evidence in NPSC audit + `test_npsc4_2_residual_compatibility_gate.py`).

---

## Canonical Decision / Execution path

```text
Decision → Governance → DecisionExecutionAuthorization → ExecutionRequest → ExecutionRuntime
```

**Result:** **PASS** (no new bypass in audited production chain; zero-bypass gate at HEAD).

---

## Persistence / plugin / DI review

**Persistence boundary** (`engine/core → contracts → providers → vendors`): **PASS** (unchanged; EE-B1.1 contract-only persistence semantics).

**Pluginability** (replaceable providers, contract-first, DI, no service locator in new scope): **PASS**.

---

## Hidden dependency scan (audited production scope)

Targeted review of commits `118798759…`, `fef16c3b9…`, `a185403d0…` plus format-only touch in `524ce5b9…`:

- No new unapproved `getattr` / `setattr` ownership bypass in scoped modules.
- EE-B1.1 contracts: frozen models, no `dict[str, Any]` in public contract surface.
- No direct vendor coupling introduced in baseline chain.
- **Observation:** module-level `default_execution_failure_classifier()` singleton (stateless, replaceable) — documented in EE-B1.1 recertification.

---

## Static quality (baseline-relevant scope @ selection)

| Tool | Scope | Result |
| ---- | ----- | ------ |
| `ruff check` | tracing + execution_reliability + NPSC wiring files | **PASS** |
| `ruff format --check` | same | **PASS** (after **FORMAT_ONLY** on `production_delegated_subtask_child_execution_wiring.py` in this task) |
| `pyright` | tracing + execution_reliability runtime/contracts | **PASS** (0 errors) |

---

## Tests (selection slice @ HEAD)

| Command | Result |
| ------- | ------ |
| `uv run pytest tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py -q` | **PASS** |
| `uv run pytest tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py -q` | **PASS** |
| `uv run pytest tests/unit/runtime/architecture/test_ee_b1_1_failure_semantics_certification.py -q` | **PASS** |
| `uv run pytest tests/unit/contracts/test_tracing_public_contract.py -q` | **PASS** |
| `uv run pytest tests/unit/runtime/architecture/test_platform_execution_unification_u5_final_zero_bypass.py -q` | **PASS** |

**Combined:** **53 passed** (single batched invocation).

---

## Remaining blockers

```text
None for baseline selection.
Formal freeze remains a separate task.
```

**Not performed:** Certified Core Platform = **FROZEN** (explicitly forbidden in this task).

---

## Verdict

```text
BASELINE SELECTED
```

**Certified code baseline SHA:** `a185403d0c7524c29bea2fe09212f9508e6bccd8`

---

## Findings

| Severity | Finding |
| -------- | ------- |
| **Observation** | `524ce5b9…` mixed qualification + format-only production file edits; baseline unchanged. |
| **Observation** | EE-B1.1 wiring consumption of `ExecutionFailureClassifier` deferred (documented in scoped recert). |
| **Minor** | Pre-selection `ruff format --check` failed on one NPSC wiring file; corrected as FORMAT_ONLY in selection commit. |

No **Critical** or **Major** baseline-selection blockers.
