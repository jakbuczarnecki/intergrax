# EQO-W3-WB07 — COR/DRV disposition paths (wave 9)

**Task ID:** `EQO-W3-WB07-COR`
**COR outcome:** `NOT_APPLICABLE_WITH_EVIDENCE` for product CL-01 in this wave — documentation disposition paths only.
**AUD session E (aggregate):** `FAIL` (`runs/wb07/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb07/AUD_SESSION_E.md` | Independent session E verdict (input) |
| `runs/wb07/W3_CLS.md` | Partial CLS pre-close (wave 5) |
| `runs/wb07/COR_DRV.md` | This wave-9 docs routing record |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Finding disposition paths (docs-only @ pin)

| Finding / obligation | AUD verdict | Docs disposition path | Target artefact | Product / EXE route (deferred) |
|----------------------|-------------|----------------------|-----------------|-------------------------------|
| EQOA-A2-043 | The question is whether W4 permit/cancellation-race surfaces (EBH-2E/2G/2H) changed after closure without a requalification record. The slice runs no SHA-versus-delta check, and the passing durable-recovery tests do not show the surfaces are unchanged. The HARNESS-01 failure (unclassified Nexus importers) is consistent with later change, but it is not proof for this finding. Closure needs a delta replay against the certified SHAs or an owner record. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A2-043 | DEFERRED — wave 10 EXE lane (out of wave 9) |
| EQOA-A2-049 | Hypothesis confirmed in source. `tests/qualification/trace_x/_trace_x_p6_restart_resume_classification.py` maps whole directory prefixes to `A_CANONICAL_RESUME` in `_PREFIX_CLASSIFICATION` (for example `intergrax/runtime/execution/`, `intergrax/integrations/`, `intergrax/skills/`, `intergrax/queueing/`, `intergrax/applications/_shared/`, `applications/governed_contractor_application/`). `classify_restart_resume_path` returns `G_UNCLEAR` only for paths outside every prefix. A new module under a listed directory is therefore classified as canonical resume without review, so "unclassified = 0" is structurally weak. The docstring says unknown surfaces must not default to canonical resume, which the prefix rows contradict. The adversarial bundle that could test this was skipped. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A2-049 | NONE — docs path sufficient at pin |
| EQOA-A4-034 | Duplication and dormancy confirmed in source at the pin. Retry-budget mechanisms: the `RetryBudgetPort` contract (`intergrax/contracts/retry_budget.py`), the inline budget in `intergrax/llm_adapters/_shared/resilience.py` (`_InlineRetryBudgetLogicalCall`, the active one), `intergrax/runtime/resilience/local_provider_retry_budget.py` (`LocalProviderRetryBudget`), `intergrax/runtime/reliability/step_retry_budget.py` (`StepRetryBudget`) and `intergrax/fastapi_core/execution/governance/contracts.py:41` (`RetryBudget`). `StepRetryBudget` is imported only by tests. `LocalProviderRetryBudget` is imported only by tests and qualification inventories. The `fastapi_core` `RetryBudget` is referenced only by that package's `defaults.py`. No test in the slice covers them. Route: owner decision to remove, unify or sanction each mechanism. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A4-034 | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |
| EQOA-A4-045 | Coexistence confirmed in source. Besides `DurableArtifactRef` (`intergrax/contracts/artifact_ref.py:141`), the pin defines `ArtifactRef` (:229), `ObservabilityArtifactReference`, `ReliabilityDiagnosticArtifactRefs`, `SharedArtifactRef`, `ApplicationArtifactRef`, `WorkspaceArtifactRef`, `SandboxArtifactRef`, `PersistedTraceArtifactRef`, `TraceArtifactRef`, `OptimizationArtifactReference` and four runtime evidence `*ArtifactRef` models. `artifact_ref.py` has no text that maps or relates them, and a search of `docs/project/architecture` and `docs/project/maintainers/plans` finds no mapping that names `DurableArtifactRef`. No test in the slice covers it. Route: owner publishes the mapping or records the sanctioned set. | `DOC_PATCH` | eqo_closure/ or owner qualification record | DEFERRED — wave 10 EXE lane (out of wave 9) |

## Aggregate rationale (session E)

Session E aggregate **`FAIL`** — corrective product work is explicitly out of scope for wave 9; routes above are limited to qualification / governance documentation under `eqo_closure/` and linked owner records.

## DRV trigger (pin / train move)

- After #680 / train pin advances: execute `EQO-W3-WB07-DRV` (`RV-01`); do not open a new EQO campaign.
- Product fixes and pytest replays: wave 10+ lanes per `EQO_WAVE9_MANIFEST.md`.

## Limits and non-claims

- Derived from `AUD_SESSION_E.md` and existing DIS artefacts only; no pytest re-run in this lane.
- No G6 closure, IIAC IN_FORCE, FRZ promotion, or FCS readiness claim.
- No product code, scripts, tests, or register apply in this write set.
