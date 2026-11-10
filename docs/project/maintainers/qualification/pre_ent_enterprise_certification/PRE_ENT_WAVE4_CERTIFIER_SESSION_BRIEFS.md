# PRE-ENT Wave 4 — copy-paste certifier session briefs (handoff only)

**Operator:** five **separate** chats, each `certification-governance-agent` (distinct from implementer and from Wave 3 `independent-quality-auditor`).  
**This publisher turn:** handoff docs only — **no certifier sessions**, **no B6/G6 campaign CLOSED**, **no FRZ promotion**.

| Pin | SHA | Use |
| --- | --- | --- |
| **AUDIT_HEAD** (product + auditor replay) | `1e83ade3f5fb951a9eedd85d4dae89ead48955ee` | Checkout for blocking pytest and code review |
| **CERT_DOCS_HEAD** (campaign handoff SSOT) | `d00df592037404cd6c2b1bc95c00757544d551d8` | Read Wave 3 reports, queue, blocker matrix, these briefs |
| **IIAC @ AUDIT_HEAD** | exit **0** | [`PRE_ENT_POST_IIAC_GATE_EVIDENCE.md`](PRE_ENT_POST_IIAC_GATE_EVIDENCE.md) |

**Queue SSOT:** [`PRE_ENT_WAVE4_CERTIFICATION_HANDOFF_QUEUE.md`](PRE_ENT_WAVE4_CERTIFICATION_HANDOFF_QUEUE.md)  
**Wave 3 auditor evidence:** [`wave3_audits/WAVE3_AUDIT_CONSOLIDATED_REGISTER.md`](wave3_audits/WAVE3_AUDIT_CONSOLIDATED_REGISTER.md)  
**Blockers:** [`PRE_ENT_WAVE4_BLOCKER_MATRIX.md`](PRE_ENT_WAVE4_BLOCKER_MATRIX.md)  
**Closure path (Waves 5–6):** [`PRE_ENT_WAVE4_TO_6_CLOSURE_PATH.md`](PRE_ENT_WAVE4_TO_6_CLOSURE_PATH.md)

**Default pytest prefix:** `uv run --extra dev-ci python -m pytest -p no:xdist`

Paste **one** block per new chat.

---

## Brief 1 — T-IGX-034-B (BND-03B certifier)

```text
PRE-ENT Wave 4 certification — T-IGX-034-B (BND-03B manifest edge).

Role: certification-governance-agent (new session; not the Wave 3 auditor or implementer).

Product checkout @ AUDIT_HEAD: 1e83ade3f5fb951a9eedd85d4dae89ead48955ee
Handoff docs @ CERT_DOCS_HEAD: d00df592037404cd6c2b1bc95c00757544d551d8

Ticket: governance/tickets/T-IGX-034-B.md
Wave 3 auditor PASS: docs/project/maintainers/qualification/pre_ent_enterprise_certification/wave3_audits/WAVE3_AUDIT_T-IGX-034-B.md
Wave 1 status: docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE1_BND03B_STATUS.md

Blocking pytest (certifier replay @ AUDIT_HEAD):
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/architecture/test_bnd03b_product_manifest_registry_gate.py -q

Scope: ticket-scope certifier disposition only — accept or CONDITIONAL scoped on BND-03B deliverables per auditor findings (doctor CLI importlib residual is disposition, not auto FAIL unless operator routes).

Limits: no global FRZ-BND-03 PASS, no PRE-ENT campaign CLOSED, no roadmap §5 publication, no auto ticket lifecycle closed without operator sign-off.

Log: .tmp/session/pre-ent-wave4-cert/bnd-03b/
Ref: governance/tickets/T-IGX-034-A.md
```

---

## Brief 2 — T-IGX-034-E (REPLAY mechanical gates certifier)

```text
PRE-ENT Wave 4 certification — T-IGX-034-E (ROADMAP-REPLAY-X mechanical exit gates).

Role: certification-governance-agent (new session).

Product @ AUDIT_HEAD: 1e83ade3f5fb951a9eedd85d4dae89ead48955ee
Docs @ CERT_DOCS_HEAD: d00df592037404cd6c2b1bc95c00757544d551d8

Ticket: governance/tickets/T-IGX-034-E.md
Wave 3 auditor PASS: docs/project/maintainers/qualification/pre_ent_enterprise_certification/wave3_audits/WAVE3_AUDIT_T-IGX-034-E.md
L3 SSOT: docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_REPLAY_L3_PROD_DELTA.md

Blocking pytest:
uv run --extra dev-ci python -m pytest -p no:xdist tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -q

Scope: mechanical reconciliation + prod-Δ disposition honesty @ AUDIT_HEAD; governance lag on ticket Approved line is observation only unless certifier finds material contradiction.

Limits: no FRZ row promotion, no roadmap §5 ledger CLOSED, no G6/IIAC/ENT campaign closure.

Log: .tmp/session/pre-ent-wave4-cert/replay/
Ref: governance/tickets/T-IGX-034-A.md
```

---

## Brief 3 — T-IGX-030-B (EBH-4 R1 certifier)

```text
PRE-ENT Wave 4 certification — T-IGX-030-B (RuntimeErrorCode contract home).

Role: certification-governance-agent (new session).

Product @ AUDIT_HEAD: 1e83ade3f5fb951a9eedd85d4dae89ead48955ee
Docs @ CERT_DOCS_HEAD: d00df592037404cd6c2b1bc95c00757544d551d8

Ticket: governance/tickets/T-IGX-030-B.md
Wave 3 auditor PASS: docs/project/maintainers/qualification/pre_ent_enterprise_certification/wave3_audits/WAVE3_AUDIT_T-IGX-030-B.md
Wave 2 qual: docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE2_SERIAL_QUAL_LOG.md (#2)

Minimum pytest (auditor parity):
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py -q

Scope: EBH-4 R1 contract home + nexus encapsulation gate; wider ticket §Test contract blast radius is certifier discovery optional — not blocking minimum.

Limits: no TOOL-LIFE-X stage closure, no FRZ promotion, no campaign CLOSED.

Log: .tmp/session/pre-ent-wave4-cert/030-b/
Ref: governance/tickets/T-IGX-034-A.md
```

---

## Brief 4 — T-IGX-030-C (TOOL-LIFE-X no-boundary certifier)

```text
PRE-ENT Wave 4 certification — T-IGX-030-C (pool.submit terminalization, dep_boundary None).

Role: certification-governance-agent (new session).

Product @ AUDIT_HEAD: 1e83ade3f5fb951a9eedd85d4dae89ead48955ee
Docs @ CERT_DOCS_HEAD: d00df592037404cd6c2b1bc95c00757544d551d8

Ticket: governance/tickets/T-IGX-030-C.md
Wave 3 auditor PASS: docs/project/maintainers/qualification/pre_ent_enterprise_certification/wave3_audits/WAVE3_AUDIT_T-IGX-030-C.md
L3 row: docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_REPLAY_L3_PROD_DELTA.md

Minimum pytest:
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py -q

Scope: C3/C4 no-boundary terminalization only; OD-HIST04-C TOOL-LIFE-X label disposition remains campaign blocker (see PRE_ENT_WAVE4_BLOCKER_MATRIX.md) — certifier may ACCEPT SCOPED without parent TOOL-LIFE-X stage closure.

Limits: no global FRZ, no G6 PRE-ENT CLOSED.

Log: .tmp/session/pre-ent-wave4-cert/030-c/
Ref: governance/tickets/T-IGX-034-A.md
```

---

## Brief 5 — T-IGX-032-A (SKILL-X KV pinning certifier)

```text
PRE-ENT Wave 4 certification — T-IGX-032-A (KV authoritative pinning wiring).

Role: certification-governance-agent (new session).

Product @ AUDIT_HEAD: 1e83ade3f5fb951a9eedd85d4dae89ead48955ee
Docs @ CERT_DOCS_HEAD: d00df592037404cd6c2b1bc95c00757544d551d8

Ticket: governance/tickets/T-IGX-032-A.md
Wave 3 auditor PASS: docs/project/maintainers/qualification/pre_ent_enterprise_certification/wave3_audits/WAVE3_AUDIT_T-IGX-032-A.md

Minimum pytest (Wave 3 auditor parity):
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py -q

Optional extended (Wave 3 brief — not run by auditor; certifier may replay or defer):
uv run --extra dev-ci python -m pytest -p no:xdist tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py -q

Guidance: P2-C qual was **not** run @ Wave 3 — certifier chooses **replay** (recommended for full ticket contract) or **defer** with explicit scoped ACCEPT on unit gate only; HIST-F-DUR-01 remains T-IGX-034-C / blocker matrix, not 032-A scope.

Limits: no FRZ closure, no HIST-F-DUR-01 PASS, no G-SKILL §3.0.4 scope bleed.

Log: .tmp/session/pre-ent-wave4-cert/032-a/
Ref: governance/tickets/T-IGX-034-A.md
```
