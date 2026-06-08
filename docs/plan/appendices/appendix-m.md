**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Appendix M — Full architecture audit traceability (Phase FAUDIT-32)

**Purpose:** 100% mapping from 32-layer [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §8 audit to concrete **FAUDIT.\*** remediation IDs. **Canonical phase narrative:** [Phase FAUDIT-32](#phase-faudit-32--full-architecture-audit-closeout).

**Status:** **Done** (2026-06-06) · **23/23 remediation Done** + [§6.1ai](#61ai-harness-implementation-queue--faudit-32-follow-up-closed) follow-up · gate **901**

### M.1 Layer → FAUDIT ID matrix (High + Critical only)

| Layer | AUDIT_MAP § | Gap summary | Severity | FAUDIT ID |
|-------|-------------|-------------|----------|-----------|
| Tier boundaries | §2 | `intergrax/runtime/architecture/capability_graph_applications.py` imports `applications.*` | **Critical** | FAUDIT-TIER.1, FAUDIT-TIER.2 |
| Task intake | §3 | No `TaskEnvelope`; worker≡HTTP parity incomplete | High | FAUDIT-INTAKE.1, FAUDIT-INTAKE.2 |
| Identity | §4 | No service/agent identity; delegation scope | High | FAUDIT-ID.1, FAUDIT-ID.2 |
| Policy | §5 | Pre-LLM/pre-output hooks absent | High | FAUDIT-POL.1 |
| LLM adapters | §6 | No policy-driven routing | High | FAUDIT-LLM.1 |
| Cognition | §7 | No `DecisionRecord` per step | High | FAUDIT-COG.1 |
| Orchestration | §9 | No backpressure | High | FAUDIT-ORCH.1 |
| Subagents | §10 | No `SubtaskContract` | High | FAUDIT-SUB.1 |
| Memory | §15 | Entity graph memory; STM retention | High | FAUDIT-MEM.1 |
| Prompts | §17 | No golden prompt CI | High | FAUDIT-PE.1 |
| Registry | §19 | Snapshot omits agents/eval | High | FAUDIT-REG.1 |
| Capability graph | §20 | Missing prompt nodes; no release impact gate | High | FAUDIT-CG.1, FAUDIT-CG.2 |
| Observability | §21 | Missing `LLM_CALL`/`POLICY_DECISION` events | High | FAUDIT-OBS.1 |
| Reliability | §22 | Shallow error taxonomy | High | FAUDIT-REL.1 |
| Security | §23 | No `DataClassification` | High | FAUDIT-SEC.1 |
| Cost | §24 | Tenant attribution not mandatory | High | FAUDIT-COST.1 |
| Evaluation | §25 | Release baseline not CI-enforced | High | FAUDIT-EVAL.1 |
| Lifecycle | §31 | State catalog mismatch; weak adoption | High | FAUDIT-ALG.1 |
| Ops / SLOs | §30 | `release_cycles.json` artifact policy | High | FAUDIT-OPS.1 |

### M.2 Cross-layer themes

| Theme | Layers affected | Risk |
|-------|-----------------|------|
| **Closeout vs maturity** | §17–§25, §31 | Plan **Done** on wiring; AUDIT_MAP **L2** on depth — do not conflate |
| **Dual-path telemetry** | §21, §6 | **L4 Done:** [Phase OBS-BUS](#phase-obs-bus--unified-observability-spine) — unified journal, `ObservabilityEmitter`, typed payloads, emission coverage, journal export |
| **Tier boundary drift** | §2, §28 | Single Critical violation undermines canon §7.4.4 |
| **Identity / intake naming** | §3, §4 | Resolved — `TaskEnvelope` in `intergrax/contracts/task_envelope.py`; parity tests in `test_faudit_remediation.py` |

### M.3 Paydown log

| Date | FAUDIT ID | Summary |
|------|-----------|---------|
| 2026-06-06 | FAUDIT-32.0 | Full 32-layer audit (`scope: C`, `audit-and-fix`); scorecard + §6.1ah queue + Appendix M; gate **893**; boundary scripts OK |
| 2026-06-06 | FAUDIT-TIER.1–OPS.1 | **23/23** remediation implemented; tier gate + intake + observability + registry depth |
| 2026-06-06 | FAUDIT-PE.1+/ALG.1+/MEM.1+ | Golden prompt CI, reference agent lifecycle metadata, STM retention wiring; gate **901** |
| 2026-06-07 | OBS-DEPTH.* + T12 + LEG depth | Unified journal + trace bridge gate + live bus emit + 170-tool catalog + §21 L3 depth gate; gate **967** |
| 2026-06-07 | T13 + CRIT-V-2.* | `eval.judge` + `eval.trajectory`; catalog **172**; doc sync; gate **990** |
| 2026-06-07 | CRIT-V-3.1–3.3 | `CriticOrchestrator`, `L0Gateway`, `L1Gateway`, `CriticEvalToolClient` | gate **996** |

---
