# Evidence and certification (DCP)

## Ready for daily use vs certified

| Claim | Meaning |
|-------|---------|
| **Ready for operator/agent use** | W2–W7 technical baseline on `development`; `igx_gov doctor` + governance-fast CI green; hooks documented. |
| **DCP program certified** | Requires independent G6 PASS + operator signoff; `global_certification.dcp_program_certified` stays **false** until then. |

Implementers MUST NOT set `g6_independent_audit_pass` or `dcp_program_certified` true. GOV-08 may read `ready_for_audit` while certification flags remain false.

## Manifest flags

[`governance/GOVERNANCE-MANIFEST.yaml`](../../../../governance/GOVERNANCE-MANIFEST.yaml) keeps `global_certification.*` and phase `certification_ready: false` until external audit.

DCP implementation does **not** imply G6 PASS or FRZ closure.

## SHA ladder (delivery artifacts)

| Artifact | Integrity |
|----------|-----------|
| Plan attached to ticket | `Plan digest: <sha256>` via `plan_digest.py` |
| Policy registry | `validate_policy_registry.py` + optional GOV-06 `TRUSTED_POLICY_PIN.yaml` (ADVISORY CI job) |
| Ticket checkpoint | Paste of `igx_gov ticket` stdout in ticket file |

Session diagnostics belong under `.tmp/session/` only ([session artifacts rule](../../../../agent-env/rules/intergrax-session-artifacts.md)).

## Qualification

FRZ-* statuses remain authoritative in `docs/project/maintainers/qualification/` — referenced, not modified, by DCP.
