# ERL-QUAL-004 — platform diagnostics gap (ERL-QUAL-004 Diagnostics Integration)

## Business requirement

Enterprise payment uncertainty recovery must give operators **why** the platform reached a reliability posture and **what to do next**, not only a chronological trace. Material states include:

- external payment effect **UNKNOWN** (retry may be unsafe),
- reconciliation **truth established** vs **unavailable / insufficient evidence**,
- resolution decision with evidence basis and resulting posture,
- governance allow / approval / block with policy reason,
- recovery continue / terminate / wait / escalate with business meaning.

Diagnostics must correlate with execution identity (`trace_id`, `correlation_id`, `execution_id`, `run_id`, evidence references) and must **not** duplicate the trace timeline or override governance.

## Observed limitation

ERL-QUAL-004 today records lifecycle steps as `TraceEvent` rows with scenario-owned `DiagnosticPayload` (`ErlQual004LifecycleStepDiagV1`) on the **Harness Observability Spine** (`intergrax.contracts.tracing`). That answers **what happened** along the run. It does **not** integrate with the **central platform diagnostics** spine (`intergrax.contracts.diagnostics` + runtime Problem lifecycle / operator investigation projections).

There is **no** public contract or runtime hook that:

1. Emits operator-meaningful reliability findings from `intergrax.runtime.enterprise_reliability` admission → reconciliation → evidence → resolution → governance → recovery.
2. Maps those facts into durable `Problem` / occurrence records with remediation hints without distorting execution-centric `DiagnosticFindingKind` values.
3. Allows the scenario proof executor (direct ERL API calls, no hosted execution engine) to publish diagnostics through the same path as production applications (`TerminalExecutionDiagnosticTrigger` → `DiagnosticOrchestrator` → `ProblemLifecycleEngine`).

Attempting to “integrate” by adding scenario-local engines, duplicate DTOs, or importing `intergrax.runtime.diagnostics.*` from scenario orchestration would violate architecture gates used elsewhere (e.g. ai_incident_investigation) and the task’s **no bypass** rule.

## Impacted contracts (audited)

| Layer | Artifact | Role |
| --- | --- | --- |
| Public persistence | `intergrax.contracts.diagnostics` (`ProblemPersistence`, `PersistedProblem`, `ProblemId`, `ProblemStatus`) | Durable Problem rows — **read/write ports only**, no emission API |
| Public investigation enums | `intergrax.contracts.diagnostic_investigation` (`DiagnosticRecommendationKind`, `DiagnosticInvestigationSeverity`, …) | Operator read projection vocabulary — **not** write/emission |
| Public extension | `intergrax.contracts.diagnostic_extension_evidence` | Enrichment SPI — assumes central Problem spine already exists |
| Observability spine | `intergrax.contracts.tracing.DiagnosticPayload`, `TraceEvent` | Typed payloads on trace events — **not** central diagnostics |
| Narrow public observer | `intergrax.contracts.execution_evidence.persistence_reliability_diagnostics_contract` | Persistence reliability decisions only — **not** ERL external-effect uncertainty |
| Runtime (internal) | `DiagnosticOrchestrator`, `ProblemLifecycleEngine`, `PlatformProblemSignal`, `DiagnosticInvestigationView`, `TerminalExecutionDiagnosticTrigger` | Canonical diagnostics processing — **not** a documented public boundary for applications/proofs |

**Enterprise Reliability Layer:** `intergrax/contracts/enterprise_reliability/**` defines governance, evidence, recovery decisions but **no diagnostic emission or Problem correlation contract**.

## Why a scenario-local workaround is unacceptable

- **Duplicate model:** A `PaymentDiagnosticEngine` or parallel `diagnostic_records` tuple of ad-hoc DTOs would recreate `DiagnosticInvestigationView` / `ProblemOccurrence` semantics outside the platform spine.
- **Trace masquerading as diagnostics:** More `TraceEvent` payloads (including per-step spam) do not satisfy “why it matters” without a distinct operator finding + recommendation lifecycle.
- **Runtime bypass:** Wiring `DiagnosticOrchestrator` only inside the proof scenario would require `intergrax.runtime.diagnostics` imports in orchestration code — forbidden by qualification architecture and inconsistent with hosted-application integration (`HostedApplicationDiagnosticEventPublisher`).
- **Semantic distortion:** Mapping UNKNOWN / reconciliation exhaustion / governance block into existing `DiagnosticFindingKind` (execution failure, lifecycle anomaly) would misrepresent proven facts and break operator trust.

## Required future architecture (no implementation decision here)

Design a **generic** (domain-neutral) reliability diagnostics bridge, owned by platform/runtime + ERL contracts, for example:

1. **Public emission contract** (contracts layer): observer or signal port for external-effect uncertainty episodes, reconciliation disposition, evidence sufficiency, resolution/governance/recovery posture — with stable correlation fields and evidence refs.
2. **Problem grouping strategy** (runtime): reconcile occurrences by `correlation_id` / reliability case identity / external-effect contract id (not payment-specific hard-coding in `intergrax/`).
3. **Operator read projection** (runtime read models): extend investigation composition to surface reliability timeline entries (see existing `DiagnosticTimelineEntryKind.GOVERNANCE`) from **canonical ERL journal / decision artifacts**, not from proof-local logs.
4. **Execution-engine integration point:** document how lab/proof runs and hosted runs both feed the same port (today proof runs bypass `TerminalExecutionDiagnosticTrigger`).
5. **Plugin boundary:** optional `DiagnosticExtensionEvidence` contributors for domain wording (payment retry double-charge risk) while platform owns identity, severity, correlation, lifecycle.

Suggested separate design task name: **`ERL-DIAG-001 External-Effect Reliability Operator Diagnostics Architecture`**.

## Generic capability (not payment-specific)

Any external-effect workflow with UNKNOWN admission, reconciliation against external truth, evidence-limited resolution, governance gates, and recovery handoff needs the same operator semantics. Payment is the qualification instance; the missing capability is **enterprise reliability operator diagnostics on the central Problem spine**.

## Affected platform layers

- **Tier-0 contracts:** new public emission/read ports (diagnostics + possibly ERL journal correlation).
- **Tier-1 runtime:** `enterprise_reliability` → diagnostics bridge; orchestrator grouping strategies; investigation projection.
- **Tier-3 proofs/applications:** consume public ports only; scenario owns payment phrasing via extension or payload schema ids.

## Ownership

- **Platform / runtime:** diagnostic identity, severity/category, correlation, persistence, investigation read models, evidence refs, lifecycle.
- **ERL contracts:** stable identifiers for reliability case, reconciliation disposition, governance/recovery decisions linked into diagnostics.
- **Scenario/application:** domain-specific operator wording and business remediation text via approved extension or namespaced payload schemas.

## Likely lifecycle

1. Architecture design + ADR (public contracts, correlation model, trace vs diagnostics boundary).
2. Runtime implementation behind contracts (no scenario shortcuts).
3. ERL-QUAL-004 re-qualification: integrate scenario through public ports only; add tests listed in ERL-QUAL-004 Diagnostics Integration task.

## Plugin / provider considerations

Reuse `DiagnosticExtensionEvidence` contributors for domain facts; do **not** invent a parallel registry in the scenario. Problem grouping strategies and scope providers may need an **ERL/reliability subject** provider alongside execution-scoped providers.

## Compatibility impact

Adding a new public emission port and read enrichment is backward compatible if optional. Changing `TraceEvent` semantics is **not** required. Existing trace-based ERL-QUAL-004 qualification remains valid until diagnostics integration is replanned.

## Current scenario state (reference)

Tracing integration is complete (`intergrax.contracts.tracing`, `RecordingScenarioExecutionTrace`, tests in `test_erl_qual_004_execution_tracing.py`). Central diagnostics integration is **blocked** on the gap above.
