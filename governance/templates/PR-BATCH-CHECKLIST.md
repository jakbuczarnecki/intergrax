# PR batch checklist (copy into PR description)

- [ ] Ticket: `governance/tickets/T-IGX-___-_.md` lifecycle ≥ `quality_gate`
- [ ] Every commit touching product paths includes `Ref: governance/tickets/T-IGX-___-_.md`
- [ ] `uv run python -m scripts.governance.igx_gov ticket` log pasted in ticket **Checkpoint**
- [ ] No G6 PASS or certification claims from implementer session
- [ ] Tier boundaries respected (`AGENTS.md`)
- [ ] CI smoke / required checks green on PR
