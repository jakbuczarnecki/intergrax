# WAVE4 certifier — T-IGX-032-A (SKILL-X KV pinning precedence)

| Field | Value |
|-------|-------|
| Session slug | `032-a` |
| Role | `certification-governance-agent` |
| AUDIT_HEAD | `1e83ade3f5fb951a9eedd85d4dae89ead48955ee` |
| CERT_DOCS_HEAD (handoff SSOT read) | `d00df592037404cd6c2b1bc95c00757544d551d8` |
| Worktree | `D:\Projekty\_wt\intergrax\w4-032a` @ detached `1e83ade3` |
| Ticket | [`governance/tickets/T-IGX-032-A.md`](../../../../../../governance/tickets/T-IGX-032-A.md) |
| Ref (campaign) | `governance/tickets/T-IGX-034-A.md` |
| Wave 3 auditor | [`WAVE3_AUDIT_T-IGX-032-A.md`](../wave3_audits/WAVE3_AUDIT_T-IGX-032-A.md) — **PASS** |

## Blocking pytest (certifier replay)

```text
uv run --extra dev-ci python -m pytest -p no:xdist tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py -q
```

**Summary:** 9 passed in 5.45s, **exit 0** (parity with Wave 3 auditor @ same `AUDIT_HEAD`).  
**Log:** [`.tmp/session/pre-ent-wave4-cert/032-a/pytest_unit.log`](../../../../../../.tmp/session/pre-ent-wave4-cert/032-a/pytest_unit.log) (+ [`cert_env.txt`](../../../../../../.tmp/session/pre-ent-wave4-cert/032-a/cert_env.txt))

## Extended qual — P2-C (certifier judgment)

Wave 3 auditor ran **minimum gate only**; Brief 5 leaves P2-C qual **replay optional** or **defer**. Certifier **replayed** (recommended path for full ticket test contract).

```text
uv run --extra dev-ci python -m pytest -p no:xdist tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py -q
```

**Summary:** 7 passed in 75.17s, **exit 0**.  
**Log:** [`.tmp/session/pre-ent-wave4-cert/032-a/pytest_p2c_qual.log`](../../../../../../.tmp/session/pre-ent-wave4-cert/032-a/pytest_p2c_qual.log)

**Judgment:** P2-C replay **satisfies** the ticket's primary re-verification gate @ `AUDIT_HEAD`; no need to defer to unit-only scoped ACCEPT.

## Evidence reviewed

- Wave 3 independent audit report and consolidated register row for `032-a`.
- [`WAVE2_SERIAL_QUAL_LOG.md`](../WAVE2_SERIAL_QUAL_LOG.md) — row #4 pass @ prior campaign HEAD.
- [`PRE_ENT_WAVE4_BLOCKER_MATRIX.md`](../PRE_ENT_WAVE4_BLOCKER_MATRIX.md) — **G-SKILL** / **HIST-F-DUR-01** scoped-limit rows acknowledged.
- Product @ `AUDIT_HEAD`: `intergrax/skills/skill_execution_pinning_wiring.py` — resolver passes `document_store` to the leaf only when `kv_store is None` (KV authoritative when both backends coexist).
- Implementing commit on ancestry: `b6a056e7d` (`fix(skills): prefer KV when both pinning backends supplied (T-IGX-032-A)`).

## Scope findings (ticket only)

- Deliverables match ticket/plan: **one** resolver precedence change plus dedicated unit gate; leaf `wire_skill_execution_pinning_store`, host runtime, factories, and P2-C qual file remain untouched per ticket out-of-scope list.
- Unit gate covers KV-only, doc-only, both supplied → KV wired, production-mode durability validator interaction, and reference strict-composition store shape — consistent with alternative A2 in the ticket.
- P2-C qual green @ certifier replay — victim path (strict composition supplying both stores) no longer trips dual-backend `ValueError` at the leaf.
- **Residual (disposition elsewhere, not blocking this ticket):** **HIST-F-DUR-01** — literal / validator-delegated durable semantics on pin stores; Wave 2 notes and T-IGX-034-C ownership unchanged.
- **Campaign blockers (unchanged):** **G-SKILL** (SKILL-X §3.0.4 parent) — certifier acceptance is **KV pinning wiring only**, not full SKILL-X parent closure.
- **Governance lag (observation):** ticket lifecycle remains `plan_approved`; certifier does not advance lifecycle or FRZ.

## Not claimed

- G6 / IIAC / PRE-ENT campaign **CLOSED**
- FRZ promotion or roadmap §5 publication
- `ALL_PRE_ENT_PREDECESSORS_INDEPENDENTLY_CERTIFIED`
- PASS on **HIST-F-DUR-01** or full SKILL-X §3.0.4 scope
- Ticket lifecycle `closed` without operator sign-off

## Certifier disposition (T-IGX-032-A scope only)

**ACCEPT SCOPED**

Scoped to KV authoritative `resolve_skill_execution_pinning_wiring` precedence @ `AUDIT_HEAD`, with unit gate and **replayed** P2-C qual both green; **G-SKILL** parent and **HIST-F-DUR-01** explicitly **out of scope** for closure.
