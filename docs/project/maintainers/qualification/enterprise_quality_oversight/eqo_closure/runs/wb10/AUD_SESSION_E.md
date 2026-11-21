# EQO-W3-WB10-AUD — session E verdict (wave AUD-2)

**Verdict (WB10): `FAIL`**
**Auditor session:** independent session E (not the EXE agent; no EXE/DIS artefact in this WB authored by this session).
**Date:** 2026-11-21
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469` (source read at the pin, read-only)
**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Verdict vocabulary

| Verdict | Meaning |
|---------|---------|
| `PASS` | Evidence at the pin supports the stated disposition. |
| `FAIL` | Source at the pin confirms the defect or contradicts the stated disposition. The finding stays OPEN. |
| `BLOCKED` | The pytest slice does not produce evidence to decide, or an owner/operator decision is a prerequisite. |

Aggregate rule: any `FAIL` gives `FAIL`; otherwise any `BLOCKED` gives `BLOCKED`; otherwise `PASS`.

## Packet checks

| Check | Result |
|-------|--------|
| TRAIN_PIN in manifest / DES_PREREG / packet = `28c27cb8…` | YES (object exists locally) |
| `pytest_command_sha256` recomputed from `pytest_command` (UTF-8) | MATCH `e70afb0e…` (manifest = DES_PREREG) |
| Log vs manifest summary | MATCH: 6 failed, 66 passed, 1 skipped in 80.56s |
| Log records command, node ids or its own hash | NO for the command and hash. The log does name the six failing node ids and the failing assertions. |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |

**What the slice executes:** the TRACE-X P6 adversarial bundle and closed-world gates, the STATE-X P0 baseline and the ROADMAP-REPLAY-X mechanical exit gates. This is the first WB where the EXE failures are product evidence rather than environment noise. Six gates are red at the pin:

- `test_txp6_q02_restart_resume_closed_world_parity`, `test_txp6_q08_no_unclear_restart_resume_paths`, `test_txp6_q11_negative_sensitivity_unregistered_restart_path`: the G-class (unclear) restart/resume path `intergrax/platform/enterprise_audit/impact_map.py` is unclassified. The file exists at the pin, and the classifier has no prefix for it.
- `test_replay_x_prod_delta_empty_since_dup_x`: product files changed since DUP-X are outside `REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS`; the log lists well over a hundred `intergrax/` paths.
- `test_replay_x_closure_counters_computed` (`assert 3 == 0`) and `test_replay_x_b6_r2_reconciliation_zero` (a `ecap_counter... failures:0!=1` reconciliation violation).

The skip is the TRACE-X P6 adversarial bundle (`...adversarial_bundle.py:69`, session manifest missing). The STATE-X P0 baseline passed.

## Per-finding verdicts

| Finding | Interim DIS | AUD verdict | Rationale (evidence at pin) |
|---------|-------------|-------------|-----------------------------|
| EQOA-A2-019 | BLOCKED | **FAIL** | The certified TRACE-X-P6 restart/resume surface does not hold at the pin. The P6 closed-world gates q02, q08 and q11 fail on an unclassified path, so the certified claim of zero unclear restart/resume paths is contradicted. No TRACE-X requalification record covers the change (the slice does not produce one). Route: owner classifies the path and runs the TRACE-X requalification on a new pin. |
| EQOA-A2-030 | BLOCKED | **FAIL** | The acceptance of ROADMAP-REPLAY-X depends on gates that are red at the pin: product delta since DUP-X is not empty, closure counters are not zero (3 != 0) and the B6 R2 reconciliation is not zero. The accepted state ("CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)", asserted by `test_replay_x_operator_closure_recorded`, which passes) therefore does not describe the pin. The second half of the hypothesis (acceptance rests on auditor log inspection of artefacts absent from the repo) is not decided here: DUP-X and the EBH-7 GHA run are not in the slice. |
| EQOA-A5-012 | BLOCKED | **FAIL** | Hypothesis supported by execution at the pin. The dossier says the STATE-X and TRACE-X closed-world gates were not rerun after the 2026-10-30 replay while 274 product files changed; this session did not re-derive the 274 count, but the prod-delta gate itself lists well over a hundred `intergrax/` paths outside its scoped list. Run now, the TRACE-X closed-world gates fail (q02, q08, q11) and the STATE-X P0 baseline passes. The drift is therefore real and has already broken a gate. The "new MCP ingress" and "new durable stores" part is consistent with the listed paths (`intergrax/applications/_shared/mcp_*`, `fastapi_mcp.py`, durable store modules) but the log does not show whether each is classified. |

## Rationale for the aggregate

`FAIL`: three findings are contradicted by gates that fail at the pin, not only by source reading. No finding in this WB is decidable as PASS.

## Required follow-up (no product fix here)

1. Classify `intergrax/platform/enterprise_audit/impact_map.py` for TRACE-X P6, and replace prefix-wide mappings with file-level classification (see WB07 A2-049).
2. Refresh `REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS` or run the ROADMAP-REPLAY-X requalification for the product delta, and clear the 3 closure counters and the B6 R2 reconciliation violation.
3. Re-run the TRACE-X P6 bundle with `TRACE_X_P6_PASS1=1` so the adversarial bundle is not skipped.
4. Replay DUP-X and cite the EBH-7 GHA run record, or record the limitation (A2-030, second half).
5. Both gates may need a new pin: the verdict binds to `28c27cb`, and `development` has moved.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
