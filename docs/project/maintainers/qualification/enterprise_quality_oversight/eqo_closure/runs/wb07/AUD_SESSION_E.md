# EQO-W3-WB07-AUD — session E verdict (wave AUD-2)

**Verdict (WB07): `FAIL`**
**Auditor session:** independent session E (not the EXE agent; no EXE/DIS artefact in this WB authored by this session).
**Date:** 2026-11-21
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469` (source read at the pin, read-only)
**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Verdict vocabulary

| Verdict | Meaning |
|---------|---------|
| `PASS` | Evidence at the pin supports the stated disposition. |
| `FAIL` | Source at the pin confirms the defect or contradicts the stated disposition. The finding stays OPEN. |
| `BLOCKED` | The pytest slice does not produce evidence to decide, or an owner/operator decision is a prerequisite. |

Aggregate rule: any `FAIL` gives `FAIL`; otherwise any `BLOCKED` gives `BLOCKED`; otherwise `PASS`.

## Packet checks

| Check | Result |
|-------|--------|
| TRAIN_PIN in manifest / DES_PREREG / packet = `28c27cb8…` | YES (object exists locally) |
| `pytest_command_sha256` recomputed from `pytest_command` (UTF-8) | MATCH `cf4a7b80…` (manifest = DES_PREREG) |
| Log vs manifest summary | MATCH: 1 failed, 56 passed, 1 skipped in 69.73s |
| Log records command, node ids or its own hash | NO (summary lines only; log hash not verifiable) |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |

**What the slice executes:** decision durable-recovery unit tests (three files), the TRACE-X P6 adversarial bundle and the HARNESS-01 gates. The skip is the TRACE-X P6 adversarial bundle (`test_trace_x_p6_adversarial_bundle.py:69`: "adversarial bundle session manifest missing; rerun ... with `TRACE_X_P6_PASS1=1`"), so the one test closest to A2-049 produced no evidence. The single failure is `test_harness_01_higher_layer_nexus_imports_are_classified` (`test_harness_01_gates.py:502`): higher-layer importers of Nexus exist at the pin that are not in the HARNESS-01 inventory. The `--tb=line` log does not name them. None of the four findings is tested by a durable-recovery test, so the 56 passes decide none of them.

## Per-finding verdicts

| Finding | Interim DIS | AUD verdict | Rationale (evidence at pin) |
|---------|-------------|-------------|-----------------------------|
| EQOA-A2-043 | BLOCKED | **BLOCKED** | The question is whether W4 permit/cancellation-race surfaces (EBH-2E/2G/2H) changed after closure without a requalification record. The slice runs no SHA-versus-delta check, and the passing durable-recovery tests do not show the surfaces are unchanged. The HARNESS-01 failure (unclassified Nexus importers) is consistent with later change, but it is not proof for this finding. Closure needs a delta replay against the certified SHAs or an owner record. |
| EQOA-A2-049 | BLOCKED | **FAIL** | Hypothesis confirmed in source. `tests/qualification/trace_x/_trace_x_p6_restart_resume_classification.py` maps whole directory prefixes to `A_CANONICAL_RESUME` in `_PREFIX_CLASSIFICATION` (for example `intergrax/runtime/execution/`, `intergrax/integrations/`, `intergrax/skills/`, `intergrax/queueing/`, `intergrax/applications/_shared/`, `applications/governed_contractor_application/`). `classify_restart_resume_path` returns `G_UNCLEAR` only for paths outside every prefix. A new module under a listed directory is therefore classified as canonical resume without review, so "unclassified = 0" is structurally weak. The docstring says unknown surfaces must not default to canonical resume, which the prefix rows contradict. The adversarial bundle that could test this was skipped. |
| EQOA-A4-034 | BLOCKED | **FAIL** | Duplication and dormancy confirmed in source at the pin. Retry-budget mechanisms: the `RetryBudgetPort` contract (`intergrax/contracts/retry_budget.py`), the inline budget in `intergrax/llm_adapters/_shared/resilience.py` (`_InlineRetryBudgetLogicalCall`, the active one), `intergrax/runtime/resilience/local_provider_retry_budget.py` (`LocalProviderRetryBudget`), `intergrax/runtime/reliability/step_retry_budget.py` (`StepRetryBudget`) and `intergrax/fastapi_core/execution/governance/contracts.py:41` (`RetryBudget`). `StepRetryBudget` is imported only by tests. `LocalProviderRetryBudget` is imported only by tests and qualification inventories. The `fastapi_core` `RetryBudget` is referenced only by that package's `defaults.py`. No test in the slice covers them. Route: owner decision to remove, unify or sanction each mechanism. |
| EQOA-A4-045 | BLOCKED | **FAIL** | Coexistence confirmed in source. Besides `DurableArtifactRef` (`intergrax/contracts/artifact_ref.py:141`), the pin defines `ArtifactRef` (:229), `ObservabilityArtifactReference`, `ReliabilityDiagnosticArtifactRefs`, `SharedArtifactRef`, `ApplicationArtifactRef`, `WorkspaceArtifactRef`, `SandboxArtifactRef`, `PersistedTraceArtifactRef`, `TraceArtifactRef`, `OptimizationArtifactReference` and four runtime evidence `*ArtifactRef` models. `artifact_ref.py` has no text that maps or relates them, and a search of `docs/project/architecture` and `docs/project/maintainers/plans` finds no mapping that names `DurableArtifactRef`. No test in the slice covers it. Route: owner publishes the mapping or records the sanctioned set. |

## Rationale for the aggregate

`FAIL` because three findings (A2-049, A4-034, A4-045) are confirmed by source at the pin and the slice gives them no coverage. A2-043 needs a delta replay. The EXE failure (HARNESS-01 inventory) is real but outside the four hypotheses.

## Required follow-up (no product fix here)

1. A2-049: re-run the TRACE-X P6 bundle with `TRACE_X_P6_PASS1=1` and replace the directory-wide prefix rows with per-file classification, or record the accepted limitation.
2. A4-034: owner decision on the five retry-budget mechanisms.
3. A4-045: owner mapping of the artifact-reference vocabularies.
4. A2-043: replay the EBH-2E/2G/2H surfaces against their certified SHAs, or record the requalification.
5. HARNESS-01: classify the new higher-layer Nexus importers (separate defect, not decided here).

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
