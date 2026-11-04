# Evidence and certification (DCP)

## Manifest flags

[`governance/GOVERNANCE-MANIFEST.yaml`](../../../../governance/GOVERNANCE-MANIFEST.yaml) keeps `global_certification.*` and phase `certification_ready: false` until external audit.

DCP implementation does **not** imply G6 PASS or FRZ closure.

## SHA ladder (delivery artifacts)

| Artifact | Integrity |
|----------|-----------|
| Plan attached to ticket | `Plan digest: <sha256>` via `plan_digest.py` |
| Policy registry | Git history + optional future TRUSTED_POLICY_PIN (GOV-06 deferred) |
| Ticket checkpoint | Paste of `igx_gov ticket` stdout in ticket file |

Session diagnostics belong under `.tmp/session/` only ([session artifacts rule](../../../../.cursor/rules/intergrax-session-artifacts.mdc)).

## Qualification

FRZ-* statuses remain authoritative in `docs/project/maintainers/qualification/` — referenced, not modified, by DCP.
