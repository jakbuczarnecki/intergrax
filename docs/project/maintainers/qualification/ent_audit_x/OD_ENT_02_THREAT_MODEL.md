# OD-ENT-02 — Threat model (compensating path)

**Scope:** Option B compensating control when live GitHub branch protection evidence is not yet available.  
**Status:** DRAFT — Session A package; Session D acceptance required.  
**Residual:** **OPEN-BP-01** — org-level enforce on git host remains a hardening backlog item.

## Assets

- `development` branch integrity
- Enterprise audit CI gates (`ent-audit-fast`, `ent-audit-pr-gate`)
- Governance commit Ref + ticket SSOT

## Threats

1. **Direct push bypass** — contributor pushes to `development` without PR checks (host not enforcing protection).
2. **Fork PR without required checks** — merge path that skips enterprise-audit contexts.
3. **Local-only qualification** — agent claims PASS without CI run IDs @ pinned SHA.
4. **Self-merge narrative** — implementer session publishes B6/IIAC PASS without independent chat.
5. **Stale SHA audit** — audit replay @ old SHA while product HEAD advanced (material delta).
6. **Policy activation bypass** — `IN_FORCE` without Session B digest (N-08).

## Controls (material)

- Pre-push hooks + `uv run python -m scripts.governance.igx_gov ticket`
- Documented CI workflow jobs on `development`
- `verify_od_ent_02_compensating_control.py` + IIAC negative tests
- Independent auditor session + manifest + role separation validators

## Residual risk

- Compensating controls are **not** claimed equivalent to live branch protection without evidence (N-06).
- OPEN-BP-01 remains until LIVE disposition or explicit ARCH-FREEZE gate acceptance.
