# EQO-W3-WB04-AUD — session E verdict (wave AUD-2)

**Verdict (WB04): `BLOCKED`**
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
| `pytest_command_sha256` recomputed from `pytest_command` (UTF-8) | MATCH `8b732ccd…` (manifest = DES_PREREG) |
| Log vs manifest summary | MATCH: 1 failed, 12 passed in 26.70s |
| Log records command, node ids or its own hash | NO (summary lines only; log hash not verifiable) |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |

**What the slice executes:** the two CTRL-X files (`test_ctrl_x_qualification_batch.py`, `test_ctrl_x_r1_security_composition.py`) and the MCP-SERVER-X P5 batch wrapper. The 12 passes are CTRL-X tests. The single failure is `test_mcp_server_x_p5_consolidated_batch_passes`: the wrapper runs `pytest.main` per target and the first target, `tests/unit/applications/test_mcp_trust_a_principal_binding.py`, returned `exit=2`. Exit 2 is "interrupted", typically a collection error, and the log does not record the cause. At the pin the target file exists and every `intergrax` module it imports exists. The file imports `fastmcp`, `uvicorn` and `mcp.client.*`, which come from the optional `mcp` extra in `pyproject.toml`. A missing extra in the EXE environment is a plausible cause (WB06 shows the EXE environment lacked `langchain_core`). That is an inference. Because the wrapper stops at the first failed target, none of the other 11 P5 targets ran.

## Per-finding verdicts

| Finding | Interim DIS | AUD verdict | Rationale (evidence at pin) |
|---------|-------------|-------------|-----------------------------|
| EQOA-A2-047 | BLOCKED | **BLOCKED** | The hypothesis is documented at the pin. `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md:239` records `FRZ-CTL-01` as `PASS` (accepted HEAD `318a4d83…`, proof replay 44/44) without mentioning the 15 failures. `CTRL_X_ENTERPRISE_CONTROL_PLANE_RECERTIFICATION.md:385,413,587` records 15 pre-existing Security test failures deferred to QUAL-X (`test_ctrl_x_r2_security_baseline.py`, present at the pin). That file is not in the slice, so there is no execution evidence for the 15 failures at the pin, and the slice's CTRL-X tests cannot refute or confirm them. Closure needs a replay of the baseline file or an owner/operator decision on how `FRZ-CTL-01` states the deferral. |
| EQOA-A7-013 | BLOCKED | **BLOCKED** | The test that would exercise MCP exposure security did not run (see above). `PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md:84` (EX-D6-04) records the original gap as "MCP exposure security not exercised because fastmcp was absent". The EXE run likely repeated that condition. The hypothesis ("no later source audits it") is neither confirmed nor refuted. Route: replay the P5 batch with the `mcp` extra installed and attach a log that records the per-target exit codes. |
| EQOA-A5-015 | BLOCKED | **BLOCKED** | No ENT-AUDIT FULL run is in the slice, and the packet cites no run after T-IGX-041-M. The pytest slice cannot decide it. Closure needs a post-041-M ENT-AUDIT FULL run record or a scoped acceptance. |

## Rationale for the aggregate

`BLOCKED`: no hypothesis is contradicted by source, but none can be decided. The only test aimed at the MCP exposure path died at collection, and the other two findings are not covered by the slice. EXE exit 1 is an environment or collection problem, not a product verdict.

## Required follow-up (no product fix here)

1. Re-run the slice with the `mcp` extra present. Record the cause of the `exit=2` and the exit code of each P5 target.
2. A2-047: run `test_ctrl_x_r2_security_baseline.py` at the pin, or get an operator decision on the `FRZ-CTL-01` deferral wording.
3. A5-015: produce an ENT-AUDIT FULL run record after T-IGX-041-M, or record a scoped acceptance.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
