# T-IGX-036-A — PDC-X P0: roadmap registration and campaign dossier (no product code)

## Scope

Batch **PDC-P0** only. Program SSOT amendment registering **`PDC-X`** (`PLATFORM-DOMAIN-COMPLETENESS-X`) as a mandatory stage between **`MCP-SERVER-X`** and **`ARCH-FREEZE`**:

- Roadmap §2.0.2 linkage row, §3 row, §3.2 stage definition, §5 ledger row, next-stage/position lines, and all canonical order chains (`… → MCP-SERVER-X → PDC-X → ARCH-FREEZE`).
- ARCH-FREEZE predecessor gate: requires **`PDC-X` = CLOSED** in the roadmap §3 row and §5 ledger. ARCH-FREEZE WBS gets a **note-only** amendment (task count stays **54**).
- ECAP baseline note: PDC-X uses incremental FRZ evidence and adds no new FRZ family. ENT-AUDIT-X handoff note.
- PDC-P0-10: EVOL-X §3 status cell reconciled with §5 and the EVOL-X SHA ladder (`B6_ACCEPTANCE_PUBLICATION_SHA` `1dd76933…`, `FORMAL_CLOSURE_PUBLICATION_SHA` `be77b3a5…`).
- Campaign dossier hub `docs/project/maintainers/qualification/platform_domain_completeness/`: index, master plan pointer, baseline & scope, SHA ladder, operator decision register, and draft ODs (`OD-PDC-IIAC-SCOPE`, `OD-PDC-SEQ-01`, `OD-PDC-FRZ-01`, `OD-PDC-ARCHF-01`, all PENDING).

**Baseline pin:** `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` (`origin/development`).

## Plan digest

Plan digest: c2a1753316d3964d03080bde4b60d5c8ede282a963859a22573c036568126acb

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-A.md`](plans/PLAN_T-IGX-036-A.md). It is the PDC-X master plan P0–P9, accepted by the Operator in the planning session on 2026-11-11.

## Approved

_Pending: the Operator, or a commissioned `operator-delegate` session per IIAC Amendment 01, writes the `Approved:` line. The implementer never writes it._

## Test contract

```text
uv run python scripts/governance/verify_arch_freeze_plan_traceability.py
uv run pytest tests/qualification/roadmap_replay_x tests/qualification/ebh_6/test_ebh_6_scope_reconciliation_integrity.py tests/qualification/ebh_7 tests/qualification/dist_scale_x tests/unit/governance/test_iiac_activation_gate.py -q
uv run python -m scripts.governance.igx_gov ticket
```

Expected: exit **0**. FRZ criterion count stays **225** in **26** families. ARCH-FREEZE WBS task count stays **54**. No checklist row status edits.

`verify_iiac_policy_activation.py --require-in-force` is recorded for information only. It is currently stale because of the MCP-SERVER-X P6 product delta (owned by MCP-SERVER-X; this batch neither causes nor fixes it).

## Out of scope

- Product code under `intergrax/`, `agents/`, `applications/`.
- IIAC protocol and IIAC scripts (material governance paths under ACT-N11; the protocol header contradiction is recorded only).
- MCP-SERVER-X dossier and product files.
- FRZ status promotion, and any new `FRZ-PDC-*` family.
- Signing OD-PDC-* decisions.
- PDC-P1..P9 work.
- Docker, vendor or production configuration.
- ARCH-FREEZE certification claims.

## Lifecycle state

plan_pending

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `pdc-x-p0` |
| Worktree | `<parent>/_wt/intergrax/pdc-x-p0` |
| HEAD @ P0 start | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` |
| ROADMAP_AMENDMENT_SHA | unset until merge |
