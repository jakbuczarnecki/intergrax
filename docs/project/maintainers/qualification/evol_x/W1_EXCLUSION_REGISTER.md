# EVOL-X W1 — Exclusion register (skeleton)

Each row requires **predicate + evidence** before W2 CW-04 accepts exclusion.

| ID | Pattern | Predicate | Evidence (W1) | Status |
|----|---------|-----------|---------------|--------|
| X-01 | `build/**`, `dist/**`, `.tmp/**`, `.venv/**`, `__pycache__/**` | GENERATED_UNTRACKED | Not in `git ls-files` | ACCEPTED |
| X-02 | `applications/build/**` | GENERATED_UNTRACKED | `pyproject.toml` packaging comment | ACCEPTED |
| X-03 | `docs/**` (non-authority) | HISTORICAL_DOC | No inbound from classified code paths — per-file proof W2 | OPEN |
| X-04 | `docs/audit_results/**` | HISTORICAL_DOC | Policy | ACCEPTED |
| X-05 | `exports/`, untracked proof sandboxes | GENERATED_UNTRACKED | `git status` untracked | ACCEPTED |
| X-06 | `platform_proofs/**` (tracked) | IN_SCOPE | Must classify in inventory — **not** auto-excluded | OPEN W2 |
| X-07 | `intergrax/legacy`, `proofs`, `proof_data` | IN_SCOPE | COMPAT-X roots include proofs — classify | OPEN W2 |

**Rule:** Unresolved critical omission → `DEPENDS_UNKNOWN` or inventory node — **never** silent PASS via conservative selection alone (see CW strengthening doc).

## W2-05 proof status (2026-11-01, OD-EVOL-09)

| ID | Status | Proof (CW-04, machine-checked on HEAD) |
|----|--------|-------------------------------------------|
| X-03 | ACCEPTED | k_to_x_edges == 0 (no classified module imports a docs/** path) |
| X-06 | IN_SCOPE_CLASSIFIED | platform_proofs/ is a discovery root; every tracked .py is a classified module (owner EVOL-X-PP) |
| X-07 | IN_SCOPE_CLASSIFIED | intergrax/legacy, proofs, proof_data under intergrax/ root; every tracked .py classified |

**GENERATED_UNTRACKED semantics (X-01/02/05):** these predicates can only exclude paths absent from git ls-files. Tracked files matching those globs (e.g. platform_proofs/.../scripts/build/*.py) are NOT excluded and are classified.
