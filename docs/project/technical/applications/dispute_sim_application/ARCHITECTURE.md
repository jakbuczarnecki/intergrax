# Dispute Simulation Workspace (DSW) — architecture

**Status:** Architecture baseline v1 (2026-06-07) — scaffold + product design  
**Tier:** Tier-3 application (`dispute_sim_application`)  
**Agents:** Tier-2 `dispute_intake`, `dispute_analyst`, `dispute_strategist`, `dispute_scenario`  
**Canonical plan row:** [`docs/project/architecture/intergrax_runtime_architecture.md` §6.3a DSW.*](../../../architecture/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated)
**Derived plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
**Decision record:** [`adr/ADR-DISPUTE_SIM-001.md`](adr/ADR-DISPUTE_SIM-001.md)

---

## 0. How to use this document

| Need | Read section |
|------|----------------|
| Product philosophy, legal boundary | §3 · §4 |
| Agent roster and capabilities | §6 |
| Material intake and case model | §7 |
| Multi-agent pipeline | §8 |
| HITL and correspondence safety | §9 |
| Request flows | §10 |
| Implementation waves | §15 |

**Rule:** change architecture first, then update [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) and platform [`§6.3a`](../../../architecture/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated).

---

## 1. Strategic purpose

**Dispute Simulation Workspace (DSW)** is a **decision-support environment** for in-house legal teams and contract managers who prepare for potential litigation with subcontractors, clients, or suppliers.

DSW does **not** replace counsel. It:

1. **Organizes** dispute materials (contracts, offers, emails, settlements, reports, annexes).
2. **Maps** factual and legal arguments — strengths, weaknesses, gaps, chronology.
3. **Proposes** attack/defense lines and emphasis priorities for negotiation or court.
4. **Reviews** draft correspondence (emails, demand letters, pre-litigation notices) for tone, procedural, and evidentiary pitfalls.
5. **Simulates** court-process variants (settlement path, injunction, full trial, appeal) with outcome bands and risk notes.

**Strategic frame:** explicit product reprioritization (2026-06-07) — second business environment after LKW; validates RAG on legal corpora, multi-agent graphs, HITL for outbound correspondence, and Phase CRIT-V critic hooks on high-stakes outputs.

---

## 2. Problem statement

Organizations accumulate dispute evidence across silos. Before engaging external counsel, teams need to answer:

| Question | Example |
|----------|---------|
| **What do we have?** | "Do we have a signed annex with a payment deadline?" |
| **How strong are we?** | "Which clauses support our position and which are risky?" |
| **What line to take?** | "Attack on delay vs force majeure defense — which is better in this dispute?" |
| **What not to send?** | "Does this email admit fault or reveal a weak argument?" |
| **What if we go to court?** | "Scenario A: settlement; B: payment order; C: full proceedings — cost/risk?" |

DSW answers these through **structured agent pipelines** on a **case-scoped RAG index**, with **human approval** before any outbound legal communication.

---

## 3. Product philosophy

### 3.1 What DSW is

- A **hosted Agent OS product** (`dispute_sim_application`) wiring four bounded Tier-2 agents.
- A **case workspace** — each dispute is an isolated corpus (RAG collection + metadata).
- A **simulation and prep tool** — outputs are labeled *decision support*, not legal advice.

### 3.2 What DSW is not

| Not this | Why |
|----------|-----|
| Licensed law firm / attorney | No attorney-client relationship; mandatory disclaimer on every run |
| Autonomous sender of legal letters | Outbound drafts require HITL (L2 critic gate) |
| Replacement for `legal_application` | `legal` = contract review SKU; DSW = dispute lifecycle simulation |
| Unrestricted document dump | Intake validates types, PII policy, and case binding |
| Nexus fork | Composition only — reuse Tier-0 RAG, policy, trace, CVL hooks |

### 3.3 Design principles

1. **Case-first** — every task binds to `case_id`; no cross-case retrieval.
2. **Evidence-linked answers** — analyst/strategist outputs cite ingested chunks.
3. **Shadow drafts only** — correspondence drafts land in shadow workspace until HITL release.
4. **Simulation ≠ prediction** — scenario outputs are *bands* (favorable / neutral / adverse) with assumptions explicit.
5. **Polish procedural context default** — prompts and rubrics target Polish civil/commercial procedure; locale configurable later.
6. **Harness honesty** — gaps feed §6.1 platform queue, not runtime forks.

---

## 4. Legal and compliance boundary

| Layer | Requirement |
|-------|-------------|
| **UI / API response** | Persistent disclaimer: *"Material is decision support only; it does not constitute legal advice."* |
| **Correspondence drafts** | HITL mandatory (`dispute.correspondence` skill path — DSW.4) |
| **PII / retention** | Case data scoped per tenant; retention policy in host settings (DSW.6) |
| **Audit** | Full Nexus trace + artifact hash for every strategy/scenario output |
| **CVL (planned)** | L1 critic on argument maps; L2 critic on outbound drafts — see [`docs/project/architecture/CRITIC_VERIFICATION.md`](../../../architecture/CRITIC_VERIFICATION.md) |

---

## 5. Solution overview

```text
┌─────────────────────────────────────────────────────────────────┐
│  Tier-3  dispute_sim_application (HTTP / MCP)                   │
│  manifest · environment_profile · tool_wiring · HITL routes     │
└────────────────────────────┬────────────────────────────────────┘
                             │ Nexus graph / capability routing
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
 dispute_intake      dispute_analyst      dispute_strategist
 (organize corpus)   (arguments map)      (attack/defense lines)
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
                    dispute_scenario
              (process variants + correspondence review)
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
  Tier-0 RAG (case index)              Shadow workspace (drafts)
```

---

## 6. Agent roster

| Agent | Capability | Responsibility |
|-------|------------|----------------|
| `DisputeIntakeAgent` | `dispute.intake` | Validate uploads, classify doc types, build chronology, ingest to case RAG |
| `DisputeAnalystAgent` | `dispute.analyze` | Argument inventory, strength/weakness matrix, evidence gaps, party positions |
| `DisputeStrategistAgent` | `dispute.strategy` | Attack/defense lines, emphasis map, negotiation posture, risks to avoid |
| `DisputeScenarioAgent` | `dispute.scenario` | Court path variants, timeline/cost bands, correspondence pitfall review |

**Default entry:** `dispute.intake` (new case or new material batch).

**Full pipeline capability (target):** `dispute.pipeline` — graph: intake → analyze → strategy → scenario (DSW.2).

---

## 7. Case and material model

### 7.1 Case entity (metadata)

| Field | Purpose |
|-------|---------|
| `case_id` | Stable UUID / slug |
| `tenant_id` | Org isolation |
| `counterparty` | Subcontractor / client / supplier label |
| `dispute_type` | `payment` · `delay` · `quality` · `termination` · `other` |
| `jurisdiction` | Default `PL` |
| `status` | `intake` · `analysis` · `strategy` · `simulation` · `closed` |

### 7.2 Material types (intake taxonomy)

`contract` · `amendment` · `offer` · `email` · `invoice` · `settlement_report` · `expert_opinion` · `photo_evidence` · `procedural_doc` · `other`

### 7.3 Outputs (shadow artifacts)

| Artifact | Producer |
|----------|----------|
| `case_timeline.json` | intake |
| `argument_matrix.json` | analyst |
| `strategy_brief.md` | strategist |
| `scenario_report.md` | scenario |
| `correspondence_review.md` | scenario (draft review mode) |

---

## 8. Multi-agent pipeline (target graph)

```text
User: "New dispute with subcontractor X — attaching contract and emails"
  → dispute.intake (ingest + timeline)
  → dispute.analyze (matrix + gaps)
  → dispute.strategy (lines + emphasis)
  → dispute.scenario (variants + draft review if requested)
  → HITL checkpoint (if correspondence draft)
  → COMPLETED + artifact bundle
```

Orchestration: Nexus capability graph — **no** custom loop in Tier-2. Graph spec: DSW.2.

---

## 9. HITL and correspondence safety

| Trigger | Gate |
|---------|------|
| Draft email / demand letter / pre-litigation notice | L2 HITL — human approves or edits before export |
| Strategy brief marked `external_share` | L1 + optional L2 |
| Scenario with `binding_recommendation` flag | L2 mandatory |

Tier-3 host exposes standard Nexus HITL routes; agents emit `hitl_required` on correspondence steps.

---

## 10. Request flows

### 10.1 New case intake

```http
POST /v1/dispute_sim/run
{
  "capability": "dispute.intake",
  "input": "Organize dispute materials for ACME — contract and invoices attached",
  "metadata": {
    "case_id": "case-2024-acme",
    "source_paths": ["/data/disputes/acme/"]
  }
}
```

### 10.2 Analysis only (existing case)

`capability: dispute.analyze` + `metadata.case_id`

### 10.3 Full simulation

`capability: dispute.scenario` + `metadata.case_id` + optional `metadata.process_variants: ["settlement","injunction","trial"]`

### 10.4 Correspondence review

`capability: dispute.scenario` + `metadata.mode: correspondence_review` + draft text in `input`

---

## 11. Tier-0 dependencies

| Mechanism | Use in DSW |
|-----------|------------|
| `rag.ingest_document` / `rag.retrieve` | Case-scoped corpus |
| `workspace.write_file` (shadow) | Draft artifacts |
| `legal.*` skills (bundle) | Contract clause patterns (reuse, not duplicate) |
| Policy / HITL | Correspondence gate |
| Trace / observability | Audit trail per case |

**Integration profile:** `IntegrationProfile.legal_product()` (same family as `legal_application`).

---

## 12. Host layout

Standard Tier-3 product tree — see [`applications/USAGE.md`](../USAGE.md).

| Path | Role |
|------|------|
| `manifest.py` | Four-agent roster + `dispute_sim.product` environment |
| `host/environment_profile.py` | RAG on, web search off, harness + legal skill bundles |
| `host/wiring.py` | Registry assembly |
| `serving/fastapi_router.py` | `POST /v1/dispute_sim/run` |
| `mcp/server.py` | Cursor / IDE task transport |

**Default port:** `8025` (avoids LKW `8020` collision).

---

## 13. Relationship to existing `legal` agent

| | `legal` (Tier-2) | DSW (Tier-3 product) |
|---|------------------|----------------------|
| Scope | Single-contract review | Full dispute lifecycle |
| Host | `legal_application` | `dispute_sim_application` |
| Agents | 1 × `LegalAgent` | 4 × dispute_* agents |
| Overlap | Clause-level review skill | May *invoke* legal review as subgraph step (DSW.5) |

No merge — composition via Nexus graph when needed.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Hallucinated legal citations | RAG-only evidence mode + L1 critic |
| Overconfident win prediction | Scenario bands + explicit assumptions |
| Privileged material leakage | Tenant + case isolation; no cross-case retrieve |
| Unauthorized outbound send | Shadow + HITL; no send integration in v1 |
| Polish law drift | Versioned prompt packs + periodic eval dataset (DSW.7) |

---

## 15. Implementation waves (summary)

| ID | Title | Status |
|----|-------|--------|
| DSW.0 | Scaffold + architecture baseline | **Done** |
| DSW.1 | Intake UAEP — path validation + RAG ingest loop | Planned |
| DSW.2 | Nexus graph `dispute.pipeline` | Planned |
| DSW.3 | Analyst + strategist domain steps | Planned |
| DSW.4 | Scenario + correspondence review + HITL | Planned |
| DSW.5 | Optional subgraph to `legal.review` for clause drill-down | Planned |
| DSW.6 | Case persistence + retention policy | Planned |
| DSW.7 | Eval dataset (Polish dispute fixtures) | Planned |

Detail: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

---

## 16. Verification

```bash
uv run pytest agents/dispute_*/tests -q
uv run pytest applications/dispute_sim_application/tests -q
```

After domain steps:

```bash
uv run pytest -m gate -q
```

---

## 17. Architecture decisions

| ADR | Title |
|-----|-------|
| [ADR-DISPUTE_SIM-001](adr/ADR-DISPUTE_SIM-001.md) | Four-agent dispute simulation product split |

---

## 18. Runtime recovery (APP-EVOL-5)

| Scenario | Host action |
|----------|-------------|
| Host restart | `resume_scheduler` via `ReliabilityProfile.recovery_contract` |
| Task interrupted | `resume` with checkpoint + idempotency store |
| Graph node failure | `retry_node` via Nexus orchestration retries |
| Corrupt checkpoint | `replay_from_snapshot` using `environment_snapshot.v1` |

- **Checkpoint store:** SQLite task checkpoints (see `.env.example` / `BUILD_AND_DEPLOY.md`)
- **Scheduler:** `long_running_scheduler_enabled` for async and HITL paths
- **In-flight tasks on deploy:** drain via checkpoint + `resume_token`; do not abort without operator ack
