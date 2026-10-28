# EBH-5 — L9 auditor checklist (G-06)

| Field | Value |
| --- | --- |
| **Parent** | EBH-5 |
| **Mode** | Whole-stage replaceability certification (Protocol 03) |
| **Entry SHA** | **CERTIFICATION_HANDOFF_SHA** from [`EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md`](EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md) |
| **Forbidden verdicts** | Parent **CLOSED**, agent **AUDIT PASS**, global **FRZ PASS**, **EBH-6** closure |

## Preconditions

- [ ] Audit SHA = **CERTIFICATION_HANDOFF_SHA** (not wave/corrective SHA alone)
- [ ] `uv run pytest tests/qualification/ebh_5/ -q` green @ SHA
- [ ] `uv run pytest tests/qualification/config_x/test_config_x_final_reconciliation_gates.py -q` green @ SHA
- [ ] CONFIG-X active blockers empty @ SHA

## Evidence cross-check

- [ ] Inventory **69** vs code reachability sample
- [ ] Matrix **56**; RPL-H vs [`_behavioral_proof_registry.py`](../../../tests/qualification/ebh_5/_behavioral_proof_registry.py)
- [ ] Five QUALIFIED: composition proofs (not isolated port calls; OTLP **no** `_create_export_transport` patch)
- [ ] F01 negatives: invalid DISTRIBUTED_OTLP config; OTLP without endpoint fail-closed
- [ ] SQLite **38/38** closed-world registry
- [ ] Fixed substrate decisions vs DIST-SCALE scoped acceptance
- [ ] CONFIG-X **54/54** without revoking `fac9a43f…`

## Outcome options

- **ACCEPT** → operator may proceed toward OD-EBH5-02 / OD-EBH5-08 after SAC freeze
- **REJECT** with finding ids → consolidated remediation wave required
