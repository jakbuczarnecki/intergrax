# PRE-ENT Wave 3 — copy-paste auditor session briefs (D-PREENT-05)

**Operator:** five **separate** chats, each `independent-quality-auditor`.  
**AUDIT_HEAD:** `085be5ca9b54a43a9a8929b1c3f2e955c7bfd6c0`  
**Queue SSOT:** [`PRE_ENT_WAVE3_AUDIT_HANDOFF_QUEUE.md`](PRE_ENT_WAVE3_AUDIT_HANDOFF_QUEUE.md)  
**Rules:** `@agent-env/rules/intergrax-independent-audit.md` — **no PASS** from implementer sessions.

Paste **one** block per new chat.

---

## Brief 1 — T-IGX-034-B (BND-03B)

```text
PRE-ENT Wave 3 independent audit — T-IGX-034-B (BND-03B manifest edge).

Role: independent-quality-auditor (new session; implementer must not PASS).

Checkout @ AUDIT_HEAD: 085be5ca9b54a43a9a8929b1c3f2e955c7bfd6c0 (branch development or detached).

Ticket: governance/tickets/T-IGX-034-B.md
Evidence: docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE1_BND03B_STATUS.md
Wave 2: docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE2_SERIAL_QUAL_LOG.md (#1)

Blocking pytest:
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py -q

Capture stdout + exit code under .tmp/session/pre-ent-wave3-audit/bnd-03b/
Do not declare G6 PASS, FRZ promotion, or ready_for_audit self-promotion.
Ref: governance/tickets/T-IGX-034-A.md
```

---

## Brief 2 — T-IGX-034-E (REPLAY mechanical gates)

```text
PRE-ENT Wave 3 independent audit — T-IGX-034-E (ROADMAP-REPLAY-X mechanical exit gates).

Role: independent-quality-auditor (new session).

Checkout @ AUDIT_HEAD: 085be5ca9b54a43a9a8929b1c3f2e955c7bfd6c0

Ticket: governance/tickets/T-IGX-034-E.md
Evidence: docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_REPLAY_L3_PROD_DELTA.md
       + docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE1_REPLAY_STATUS.md

Blocking pytest:
uv run --extra dev-ci python -m pytest -p no:xdist tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -q

Log: .tmp/session/pre-ent-wave3-audit/replay/
No FRZ promotion, no roadmap §5 publication, no PASS from this implementer campaign.
Ref: governance/tickets/T-IGX-034-A.md
```

---

## Brief 3 — T-IGX-030-B (EBH-4 R1 RuntimeErrorCode)

```text
PRE-ENT Wave 3 independent audit — T-IGX-030-B (EBH-4 R1 contract home).

Role: independent-quality-auditor (new session).

Checkout @ AUDIT_HEAD: 085be5ca9b54a43a9a8929b1c3f2e955c7bfd6c0

Ticket: governance/tickets/T-IGX-030-B.md
Wave 2 qual: docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE2_SERIAL_QUAL_LOG.md (#2)
Lifecycle: plan_approved (lags git) — not closed.

Minimum pytest:
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py -q

Log: .tmp/session/pre-ent-wave3-audit/030-b/
No TOOL-LIFE-X stage closure, no FRZ promotion.
Ref: governance/tickets/T-IGX-034-A.md
```

---

## Brief 4 — T-IGX-030-C (TOOL-LIFE-X no-boundary terminalization)

```text
PRE-ENT Wave 3 independent audit — T-IGX-030-C (pool.submit terminalization, dep_boundary None).

Role: independent-quality-auditor (new session).

Checkout @ AUDIT_HEAD: 085be5ca9b54a43a9a8929b1c3f2e955c7bfd6c0

Ticket: governance/tickets/T-IGX-030-C.md
Wave 2: docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE2_SERIAL_QUAL_LOG.md (#3)
L3 row: docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_REPLAY_L3_PROD_DELTA.md

Minimum pytest:
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py -q

Log: .tmp/session/pre-ent-wave3-audit/030-c/
Ref: governance/tickets/T-IGX-034-A.md
```

---

## Brief 5 — T-IGX-032-A (SKILL-X KV pinning precedence)

```text
PRE-ENT Wave 3 independent audit — T-IGX-032-A (KV authoritative pinning wiring).

Role: independent-quality-auditor (new session).

Checkout @ AUDIT_HEAD: 085be5ca9b54a43a9a8929b1c3f2e955c7bfd6c0

Ticket: governance/tickets/T-IGX-032-A.md
Wave 2: docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE2_SERIAL_QUAL_LOG.md (#4)
Note: HIST-F-DUR-01 out of scope here (T-IGX-034-C / D-PREENT-07 defer).

Minimum pytest:
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py -q

Optional extended:
uv run --extra dev-ci python -m pytest -p no:xdist tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py -q

Log: .tmp/session/pre-ent-wave3-audit/032-a/
Ref: governance/tickets/T-IGX-034-A.md
```
