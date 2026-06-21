# EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE — production gates (§40+)

**Parent hub:** [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)

# 43. Architecture Metrics, Debt, and CI Gates

Architecture health MUST be measured, not inferred.

## 43.1 Metric families

- modularity and coupling indicators,
- dependency graph health and incompatibility rate,
- observability and governance coverage on critical paths,
- policy / context / prompt / test coverage,
- architecture debt index with trend tracking.

**Code:** `runtime/architecture/architecture_metrics.py`, `architecture_metrics_pipeline.py`, `debt_governance.py`, `architecture_coverage.py`, `maturity_gate_evidence.py`.

## 43.2 Developer experience surface

| Surface | Role |
|---------|------|
| `intergrax/scaffold/` | `new-agent`, `new-application`, `new-skill` |
| `intergrax/cli/doctor.py` | Harness health checks |
| `docs/bootstrap/idea_audit.txt` | **Mode I** — idea intake procedure (natural language in chat; [`.cursor/rules/intergrax-idea-audit.mdc`](../../.cursor/rules/intergrax-idea-audit.mdc)) |
| `scripts/test.bat` / `pytest -m gate` | Mandatory merge gates |
| `guides/AGENT_CREATION_GUIDE.md` | Author workflow |

**TTFRun** (idea → first Nexus run) is the primary DX metric. **Plan:** Phase DX, AA, W-OPS in [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md).

## 43.3 Operational L3 evidence

Release cycles, SLO snapshots, and ops sign-off are tracked via `scripts/phase_w_ops_evidence.py` and release cycle artifacts under `build/architecture_hardening/`.

## 43.4 Evidence-backed Harness Onboarding Path

Intergrax exposes a local evidence-backed onboarding path that lets a developer or early adopter verify the harness without trusting claims, demos, or external services.

The path is intentionally local and deterministic. It proves that the harness can produce, package, aggregate, and explain evidence artifacts across the core platform surfaces.

### Canonical local proof path

```bash
uv run intergrax certify core --level L2
uv run intergrax trace export
uv run intergrax evidence live-core
uv run intergrax evidence eval
uv run intergrax evidence cost
uv run intergrax evidence posture
uv run intergrax evidence posture export
```

### Evidence surfaces

| Surface                     | Command                                         | What it proves                                                              |
| --------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------- |
| Core certification          | `intergrax certify core --level L2`             | Deterministic CORE-L* contract evidence                                     |
| Trace evidence              | `intergrax trace export`                        | Report-derived trace timeline evidence                                      |
| Selected live Tier-0 probes | `intergrax evidence live-core`                  | Selected local no-network live probes using mock LLM/tools                  |
| Eval evidence               | `intergrax evidence eval`                       | Eval regression evidence packaging without real LLM/provider evaluation     |
| Cost evidence               | `intergrax evidence cost`                       | Local cost/budget/trace evidence packaging without billing/provider pricing |
| Evidence posture            | `intergrax evidence posture` / `posture export` | Read-only aggregation into an operator-facing evidence scoreboard           |

### Architectural boundary

This path is a harness proof path, not a product feature demo.

It does not:

* execute provider calls,
* use network,
* run real LLM evaluation,
* compute provider pricing,
* implement billing,
* certify full production runtime behavior,
* replace product-specific acceptance tests,
* replace security or compliance attestation.

### Relationship to ROI roadmap

The implementation status and remaining ROI tasks are tracked in:

```text
docs/plan/HARNESS_EVIDENCE_PACK.md
```

The README exposes the short operator-facing version of this proof path for developers and early adopters.

---

# 44. MVP-to-Product Evolution Layer

Every Intergrax product starts as a **prototype or MVP** on the same Harness stack. The platform MUST provide **systematic tools** for iterative design, evaluation, real-life and simulated testing, and evidence-based promotion to production — not ad-hoc scripts per team.

This layer is a **competitive differentiator**: developers ship fast; the Harness supplies feedback, gates, and promotion discipline.

## 44.1 Product maturity stages

```text
PROTOTYPE → MVP → BETA → PRODUCTION → OPTIMIZE (L4 / AHI)
```

| Stage | Goal | Harness posture | Evidence required |
|-------|------|-----------------|-------------------|
| **PROTOTYPE** | Validate idea in Nexus | `execution_mode=EXPLORATORY`, shadow eval, lab host | Smoke run + trace |
| **MVP** | First real users, narrow scope | `BALANCED`, offline + online eval sampling | Baseline eval scores, TTFRun |
| **BETA** | Scale testing, feedback loops | Stricter policy, HITL on risky paths | KPI trends, satisfaction samples |
| **PRODUCTION** | SLO-backed operation | `STRICT`, critic gates, reliability profiles | SLO window, incident budget, PRR |
| **OPTIMIZE** | Closed-loop improvement | AHI proposals, bounded policy learning | Eval deltas + human approval |

Promotion between stages is **evidence-driven** — see §44.5 and [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md).

## 44.2 Developer toolchain (as-built + target)

| Tool / surface | Role in MVP evolution | Status |
|----------------|----------------------|--------|
| `intergrax/scaffold` (`new-agent`, `new-stack`, `--minimal`) | Zero-to-first-run in minutes | **Done** — DX phase |
| `intergrax doctor` | Harness health before iterate | **Done** |
| `intergrax run` / lab `POST /v1/lab/run` | Fast local and harness validation | **Done** |
| **Agent Lab** (`lab_application`) | Compose and probe agents without product polish | **Done** |
| **Evaluation subsystem** (§42) | Offline / online / shadow / human modes | **Done** — EVAL phase |
| **Shadow workspace** | Compare candidate path without user impact | **Done** — REL Phase F |
| **Replay environment** | Deterministic re-run from trace store | **Partial** — `intergrax mvp replay` CLI (MVP-EVOL.3); no Tier-3 HTTP router |
| **Agent simulator** | Multi-agent contention and failure injection | **Partial** — `intergrax mvp simulate` CLI + `test_orchestration_cfg_simulation.py`; not wired to product hosts |
| **Trace Explorer** | Decision / tool / context visibility | **Partial** — lab debug APIs; UI deferred (GOV-PROD.1 §6.3) |
| **Promotion gates** | MVP → Beta evidence | **Done** — `scripts/check_mvp_promotion_gates.py` (MVP-EVOL.1) |
| **Product KPI / satisfaction** | Tenant metrics + CSAT bridge | **Done** — `product_kpi_registry.py`, `user_satisfaction.py` (MVP-EVOL.4–5); export surfaces CLI-only |

**IDEAL reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §22 (Developer Experience Layer).

## 44.3 Evaluation and testing strategy

| Mode | When | Mechanism |
|------|------|-----------|
| **Unit / contract** | Every PR | `pytest -m gate`, agent contract tests |
| **Golden offline eval** | Pre-merge / nightly | `EvaluationProfile` + `evaluation_assets` |
| **Shadow production** | MVP → Beta | `online_evaluation_registry` — candidate vs baseline |
| **Simulation** | Before real users | Harness CFG matrix, orchestration sim tests, future agent simulator |
| **Real-life pilot** | Beta | Sampled online eval + observability SLOs |
| **Human rubric** | Regulated / subjective quality | CVL + HITL scoring |

Raw text outputs are insufficient — structured results feed evaluators (§39.5, §42).

## 44.4 KPI, metrics, and user satisfaction

Platform and products SHOULD declare measurable outcomes. Intergrax provides **hooks and registries**; product teams define domain KPIs.

| Signal class | Examples | Harness hook |
|--------------|----------|----------------|
| **Technical KPIs** | Latency p95, success rate, cost per task, retry rate | Observability spine, `TASK_COMPLETED` payloads |
| **Quality KPIs** | Eval score trends, critic pass rate, schema validity | `evaluation_registry_trends`, CVL |
| **Product KPIs** | Task completion, time-to-value, feature adoption | Tier-3 app metrics export (product-owned) |
| **User satisfaction** | CSAT, NPS, thumbs up/down on responses | `feedback.*` integration pattern; online eval human mode |
| **Architecture health** | Debt index, gate coverage | §43 architecture metrics pipeline |

```text
Run → trace + eval score → trend registry → promotion gate / AHI proposal
User feedback → online eval registry → dashboard + optional HITL review
```

**Rule:** satisfaction and product KPIs are **not** inferred silently — explicit capture adapters or UI events with tenant scope.

## 44.5 Promotion gates (prototype → product)

| Gate | Checks |
|------|--------|
| **G0 — Runnable** | Scaffold smoke, `doctor` clean, one Nexus path green |
| **G1 — Eval baseline** | Offline golden set registered; score recorded |
| **G2 — Policy** | `ReliabilityProfile` + autonomy ceiling documented |
| **G3 — Multi-agent** | If N>1: `graph_spec` + merge + CFG proof |
| **G4 — Ops** | SLO catalog, runbook stub, checkpoint/resume if long-running |
| **G5 — Production PRR** | Phase V evidence, compatibility graph, owner sign-off |

Gates G0–G2 are **platform-enforced** via CI scripts; G3–G5 are product checklists in Tier-3 `ARCHITECTURE.md`.

## 44.6 Feedback into platform improvement

MVP iteration MUST feed the Harness — not only the product:

| Feedback source | Consumer |
|-----------------|----------|
| Eval regression | Block merge; CVL rubric updates |
| Trace anomalies | Observability alerts; optional AHI pattern detection |
| User dissatisfaction | Online eval + HITL queue; autonomy downgrade |
| Cost overrun | Cost profile tuning; model routing (AHI) |
| Failure patterns | Resilience policy proposals (REL-ADV) |

**Cross-ref:** L4 adaptive loop [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md) · agent lifecycle [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §20.

**Plan:** [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) Phase MVP-EVOL.

---
