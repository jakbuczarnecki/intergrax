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
