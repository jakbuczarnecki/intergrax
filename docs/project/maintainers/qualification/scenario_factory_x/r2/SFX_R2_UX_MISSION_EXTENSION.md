# SFX R2 — Studio / UX Mission Extension (PROPOSED)

Extends [`SFX_UX_DESIGN.md`](../SFX_UX_DESIGN.md) (11-step flow preserved).

## New surfaces

### Mission Dashboard

- Mode selector: Manual / Assisted / Bounded autonomous (disabled until grant).
- Pinned `goal_digest`, acceptance checklist, budget meters (iterations, cost, time).
- State projection badge (derived, not FSM authority).

### Iteration timeline

- `CandidateIteration` cards: hypothesis, SEF job ref, verdict summary, cost.
- Champion marker; regression diff vs previous champion.

### Candidate comparison

- Pareto-style table: business pass rate, safety, runtime, cost, human touches (G-2 / AUTO-08).

### Controls

- Pause / resume mission, request intervention, revoke grant (approver only).
- POC presentation view: demo link, limitations, explicit `SANDBOX / NOT PRODUCTION`.

## Parity (PC-1..6)

CLI: `sfx mission status|pause|resume` (PROPOSED). MCP: same op names and schemas as API. Offline CLI: read-only mission views; **no** L2 driver.

## Future customer self-service (concept only)

Restricted templates, synthetic data, quotas — documented in commercial roadmap; **not** authorized for current scope.