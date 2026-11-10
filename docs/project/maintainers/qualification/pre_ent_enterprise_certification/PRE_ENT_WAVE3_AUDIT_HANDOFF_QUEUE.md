# PRE-ENT Wave 3 — independent audit handoff queue

**Campaign:** `pre_ent_enterprise_certification`  
**AUDIT_HEAD (checkout for new sessions):** `085be5ca9b54a43a9a8929b1c3f2e955c7bfd6c0` (post-merge product tree; docs-only commits after this SHA do not change product paths)  
**Branch:** `development`  
**Date:** 2026-11-10  
**Publisher:** T-IGX-034-A Wave 3 routing — **no PASS**, **no closure**, **no push**

**Independence rule (all rows below):** Sessions that implemented or qualified these batches **must not** declare G6 PASS, IIAC PASS, FRZ promotion, stage `CLOSED`, or `ready_for_audit` self-promotion. Operator routes a **new chat** with role `independent-quality-auditor` per [`agent-env/rules/intergrax-independent-audit.md`](../../../../agent-env/rules/intergrax-independent-audit.md) and [`INDEPENDENT-REVIEW-HANDOFF.md`](../../../technical/devos/INDEPENDENT-REVIEW-HANDOFF.md).

**Default pytest prefix:**

```text
uv run --extra dev-ci python -m pytest -p no:xdist <path> -q
```

---

## Ready — schedule new independent auditor session @ AUDIT_HEAD

Each row is **engineering / qual complete enough** for read-only auditor replay on exact `AUDIT_HEAD`. Auditor captures exit codes and writes under `.tmp/session/pre-ent-wave3-audit/<slug>/`.

### 1 — T-IGX-034-B (BND-03B manifest edge)

| Field | Value |
| --- | --- |
| **Ticket Ref** | `governance/tickets/T-IGX-034-B.md` |
| **Lifecycle** | `quality_gate` — **not** closed |
| **Implementer must not PASS** | Yes — no FRZ-BND-03 global PASS, no `ready_for_audit` from implementer |

**Evidence paths**

| Kind | Path |
| --- | --- |
| Status | [`WAVE1_BND03B_STATUS.md`](WAVE1_BND03B_STATUS.md) |
| Wave 2 re-qual | [`WAVE2_SERIAL_QUAL_LOG.md`](WAVE2_SERIAL_QUAL_LOG.md) (#1 BND-03B) |
| Product | `intergrax/applications/_shared/product_manifest_registry.py`, `applications/shipped_product_manifest_catalog.py`, `intergrax/cli/doctor_health_app.py`, `intergrax/cli/doctor_diff_app.py` |
| Gate | `tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py` |

**Blocking pytest (auditor replay @ AUDIT_HEAD)**

```text
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py -q
```

**Last recorded:** 2 passed @ `f2682d651` (Wave 2 serial qual). **Residual for disposition:** doctor CLI still bootstraps catalog via `importlib` in handler scope (documented in status artifact — not gate weakening).

---

### 2 — T-IGX-034-E (ROADMAP-REPLAY-X mechanical gates — replay)

| Field | Value |
| --- | --- |
| **Ticket Ref** | `governance/tickets/T-IGX-034-E.md` |
| **Lifecycle** | `in_progress` — mechanical reconciliation only |
| **Implementer must not PASS** | Yes — no FRZ row promotion, no roadmap §5 publication |

**Evidence paths**

| Kind | Path |
| --- | --- |
| L3 prod-Δ SSOT | [`PRE_ENT_REPLAY_L3_PROD_DELTA.md`](PRE_ENT_REPLAY_L3_PROD_DELTA.md) |
| Wave 1 replay status | [`WAVE1_REPLAY_STATUS.md`](WAVE1_REPLAY_STATUS.md) |
| Qualification SSOT | `tests/qualification/roadmap_replay_x/_pre_ent_l3_prod_delta.py`, `tests/qualification/roadmap_replay_x/_prod_delta_disposition.py` |
| Gate | `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py` |

**Blocking pytest (auditor replay @ AUDIT_HEAD)**

```text
uv run --extra dev-ci python -m pytest -p no:xdist tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -q
```

**Last recorded:** 16 passed @ `f97d286b` ([`WAVE1_REPLAY_STATUS.md`](WAVE1_REPLAY_STATUS.md)). Auditor **must** re-run on `AUDIT_HEAD` after subsequent merges (incl. 034-B, 034-F docs).

---

### 3 — T-IGX-030-B (EBH-4 R1 — `RuntimeErrorCode` contract home) — serial qual, **not closed**

| Field | Value |
| --- | --- |
| **Ticket Ref** | `governance/tickets/T-IGX-030-B.md` |
| **Lifecycle** | `plan_approved` (lags git) — **not** `closed` |
| **Implementer must not PASS** | Yes — no TOOL-LIFE-X stage closure, no FRZ promotion |

**Evidence paths**

| Kind | Path |
| --- | --- |
| Wave 2 serial qual | [`WAVE2_SERIAL_QUAL_LOG.md`](WAVE2_SERIAL_QUAL_LOG.md) (#2) |
| Product | `intergrax/contracts/runtime_error_code.py`, `intergrax/runtime/nexus/errors/error_codes.py`, `intergrax/runtime/tools/tool_operation_recovery.py` |
| Gates | `tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py`, `tests/unit/architecture/test_runtime_error_code_contract_home.py` |
| Full test contract | Ticket §Test contract (wider batch optional for auditor discovery) |

**Auditor minimum pytest @ AUDIT_HEAD**

```text
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py -q
```

**Last recorded:** 8 passed @ `f2682d651` (Wave 2). Merge commit on tree: `31a6c3fe1` (Wave 0 presence).

---

### 4 — T-IGX-030-C (TOOL-LIFE-X C3/C4 — no-boundary `pool.submit` terminalization) — serial qual, **not closed**

| Field | Value |
| --- | --- |
| **Ticket Ref** | `governance/tickets/T-IGX-030-C.md` |
| **Lifecycle** | `plan_approved` — **not** `closed` |
| **Implementer must not PASS** | Yes — independent TOOL-LIFE-X re-audit is separate campaign stream |

**Evidence paths**

| Kind | Path |
| --- | --- |
| Wave 2 serial qual | [`WAVE2_SERIAL_QUAL_LOG.md`](WAVE2_SERIAL_QUAL_LOG.md) (#3) |
| Product | `intergrax/runtime/nexus/tools/invoker.py` (dep_boundary `None` branch) |
| Gate | `tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py` |
| L3 disposition row | [`PRE_ENT_REPLAY_L3_PROD_DELTA.md`](PRE_ENT_REPLAY_L3_PROD_DELTA.md) (batch T-IGX-030-C) |

**Auditor minimum pytest @ AUDIT_HEAD**

```text
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py -q
```

**Last recorded:** 6 passed @ `f2682d651`. Merge commit: `30f67c94a`.

---

### 5 — T-IGX-032-A (SKILL-X — KV authoritative pinning wiring) — serial qual, **not closed**

| Field | Value |
| --- | --- |
| **Ticket Ref** | `governance/tickets/T-IGX-032-A.md` |
| **Lifecycle** | `plan_approved` — **not** `closed` |
| **Implementer must not PASS** | Yes — PROD-Q P2-C owner path; **HIST-F-DUR-01** remains out of scope (see 034-C) |

**Evidence paths**

| Kind | Path |
| --- | --- |
| Wave 2 serial qual | [`WAVE2_SERIAL_QUAL_LOG.md`](WAVE2_SERIAL_QUAL_LOG.md) (#4) |
| DUR-01 note | Same file §DUR-01 / T-IGX-034-C — finding **unchanged** |
| Product | `intergrax/skills/skill_execution_pinning_wiring.py` |
| Unit gate | `tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py` |
| P2-C qual (auditor extended replay) | `tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py` |
| L3 disposition row | [`PRE_ENT_REPLAY_L3_PROD_DELTA.md`](PRE_ENT_REPLAY_L3_PROD_DELTA.md) (batch T-IGX-032-A) |

**Auditor minimum pytest @ AUDIT_HEAD**

```text
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py -q
```

**Recommended extended replay (ticket contract):**

```text
uv run --extra dev-ci python -m pytest -p no:xdist tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py -q
```

**Last recorded:** 9 passed (unit) @ `f2682d651`. Merge commit: `b6a056e7d`.

---

## BLOCKED — do not open independent audit session yet

| Blocker | Why blocked | SSOT / next gate |
| --- | --- | --- |
| **G-SKILL** | Operator must choose **G-SKILL-A** vs **G-SKILL-B** before SKILL-X §3.0.4 implementation scope | [`PRE_ENT_CAMPAIGN_PROGRESS.md`](PRE_ENT_CAMPAIGN_PROGRESS.md), [`WAVE0_PRESENCE_RECONCILIATION.md`](WAVE0_PRESENCE_RECONCILIATION.md) operator blockers |
| **OD-HIST04-A** | EVOL-X formal ratification pending | Wave 0 blocker table |
| **OD-HIST04-B** | IIAC material paths — no campaign edits | Wave 0 blocker table |
| **OD-HIST04-C** | TOOL-LIFE-X label vs open operator decision | Wave 0 blocker table |
| **F10 TES ladder (`496016c0`)** | In-tree F10 `TEST_EXECUTION_SHA` **`496016c0cf6cc5cf1f9a29d885677605dd0415d3`** ≠ governed execution pin; **4 identity paths**, **182 TREE_DRIFT / 0 SEMANTIC** — ladder advance blocked until policy allows controlled SHA | [`WAVE1_F10_PINNED_REPLAY.md`](WAVE1_F10_PINNED_REPLAY.md), [`WAVE1_F10_STATUS.md`](WAVE1_F10_STATUS.md), `governance/tickets/T-IGX-034-F.md` |
| **ENT Population E** | Enterprise audit rollup not run; REG defer per historical dossier | [`PRE_ENT_CAMPAIGN_PROGRESS.md`](PRE_ENT_CAMPAIGN_PROGRESS.md) |
| **Wave 3 sessions (this queue)** | Operator **D-PREENT-05** — five separate auditor chats; briefs @ [`PRE_ENT_WAVE3_AUDITOR_SESSION_BRIEFS.md`](PRE_ENT_WAVE3_AUDITOR_SESSION_BRIEFS.md) | Execution = new chats; orchestrator does not run audits |

**Related open Wave 1 (not in ready queue):** T-IGX-034-C (DUR-01), T-IGX-034-D/G/H — implement or qual incomplete; no auditor routing here.

---

## Campaign cross-links

- Progress snapshot: [`PRE_ENT_CAMPAIGN_PROGRESS.md`](PRE_ENT_CAMPAIGN_PROGRESS.md)  
- Wave 0 routing: [`WAVE0_PRESENCE_RECONCILIATION.md`](WAVE0_PRESENCE_RECONCILIATION.md)  
- Parent ticket: `governance/tickets/T-IGX-034-A.md`
