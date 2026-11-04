# ARCH-FREEZE B1 — independent audit handoff dossier

**Disposition target:** verify closed-world production inventory @ exact SHA without Cursor narrative.

| Field | Value |
| --- | --- |
| START_HEAD (wave 1) | `18a0f68fc6c8107e6344fdb4641ecec1945dae5e` |
| EXECUTION_SHA | `18a0f68fc6c8107e6344fdb4641ecec1945dae5e` |
| AUDIT_BASELINE_SHA (planning lock) | `967ef2f30c9c263fb9bb40d1b0fe77c98fc3919a` |
| Historical IC assessment HEAD (preserved) | `81829712acc19ebeadb4773467b9c5642ca975e3` |
| inventory_digest @ EXECUTION_SHA | `984e155d46529ddb345ac43a5c6493e747e8cc0fb56fc9c7408e1914a96d910c` |

## Reproduce inventory (T-B1-PROD-REACH-SCAN)

```bash
git checkout 18a0f68fc6c8107e6344fdb4641ecec1945dae5e
uv run python scripts/governance/arch_freeze_b1_closed_world_inventory.py \
  --output-dir .tmp/session/arch-freeze-b1-wave1
```

Artifacts (session-local, gitignored):

- `.tmp/session/arch-freeze-b1-wave1/arch_freeze_b1_inventory_summary.json`
- `.tmp/session/arch-freeze-b1-wave1/arch_freeze_b1_module_inventory.jsonl`
- `.tmp/session/arch-freeze-b1-wave1/ARCH_FREEZE_B1_INVENTORY_RUN.md`

## Mechanical counts @ EXECUTION_SHA

| Metric | Value |
| --- | ---: |
| Total modules inventoried (EVOL universe + supplementary `scripts/`) | 9024 |
| Mandatory tier (`intergrax/`, `agents/`, `applications/`) | 7928 |
| PRODUCTION_EFFECTIVE (all) | 8351 |
| PRODUCTION_EFFECTIVE (mandatory tier) | 7922 |
| LAB_ONLY | 673 |
| TEST_ONLY / DEAD_CODE / UNKNOWN module class | 0 |
| PROD-Q P0 raw candidates | 2991 |
| PROD-Q P0 unclassified candidates | 0 |
| EVOL module nodes | 8641 |
| Extractor `unknown:` nodes | 21 |
| Closed-world gate failures | CW-05, CW-08, CW-09, CW-10 |

## Findings (consolidated)

See [`ARCH_FREEZE_B1_FINDINGS_AND_IMPACT_DOSSIER.md`](./ARCH_FREEZE_B1_FINDINGS_AND_IMPACT_DOSSIER.md) and updated [`ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md`](./ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md).

**Not claimed:** ARCH-FREEZE CLOSED, L3 certifying PASS, independent audit PASS, CW gates PASS @ HEAD.

## Traceability

```bash
uv run python scripts/governance/verify_arch_freeze_plan_traceability.py
```

## B2 preparation

[`ARCH_FREEZE_B2_REMEDIATION_READY_DOSSIER.md`](./ARCH_FREEZE_B2_REMEDIATION_READY_DOSSIER.md)
