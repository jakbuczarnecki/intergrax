# EBH-5 — L9 auditor checklist (G-06)

| Field | Value |
| --- | --- |
| **Parent** | EBH-5 |
| **Mode** | Whole-stage replaceability certification |
| **Forbidden verdicts** | Parent **CLOSED**, **INDEPENDENTLY ACCEPTED**, global **FRZ PASS**, **EBH-6** closure |

## Preconditions

- [ ] Audit SHA = git commit under review
- [ ] `uv run pytest tests/qualification/ebh_5/ -q` green @ SHA
- [ ] CONFIG-X active blockers empty @ SHA

## Evidence cross-check

- [ ] Inventory vs code reachability sample
- [ ] Matrix RPL-H vs behavioral proof index
- [ ] SQLite annex vs fixed substrate decisions
- [ ] CONFIG-X 54 reconciliation without revoking `fac9a43f…`

## Outcome options

- **ACCEPT** implementation wave → operator may publish **READY FOR AUDIT** / proceed toward closure
- **REJECT** with finding ids → remediation wave required
