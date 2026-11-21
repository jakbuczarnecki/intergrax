# EQO-W3-WB08-AUD — session E verdict (wave AUD-2)

**Verdict (WB08): `BLOCKED`**
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
| TRAIN_PIN in DES_PREREG / packet = `28c27cb8…` | YES (object exists locally). WB08 has no `RUN_MANIFEST.json`. |
| `pytest_command_sha256` recomputed from `pytest_command` (UTF-8) | MATCH `536729a6…` (DES_PREREG) |
| Log vs manifest summary | NOT APPLICABLE: EXE `NOT_RUN`; there is no log and no manifest |
| Log records command, node ids or its own hash | NOT APPLICABLE |
| Selected test files exist at the pin | YES, all four (`test_mem_xint6_cross_layer_e2e_certification.py`, `test_d2_context_fidelity.py`, `test_d2_compaction_roundtrip.py`, `test_xd_02_agent_context.py`) |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |

**What the slice executes:** nothing. WB08 stopped after DES (preregistration). The packet leaves the choice to the auditor: accept a DES-only scope or schedule EXE first. A DES-only scope cannot produce a verdict, because the two hypotheses are claims about behaviour "at TRAIN_PIN" and nothing was run.

## Scope

WB08 owns no Campaign-A finding row (`finding_ids: []`; "0 of 51 in the B-1 table"). It carries two obligations from T-IGX-043-A/B.

## Per-obligation verdicts

| Obligation | Interim DIS | AUD verdict | Rationale (evidence at pin) |
|------------|-------------|-------------|-----------------------------|
| OBLIGATION-043-AB-CONTEXT (production_mode and context-engine requirements hold on cross-layer memory and compaction paths) | none (EXE not run) | **BLOCKED** | No execution evidence. The prerequisite is also unmet at the pin: `governance/tickets/T-IGX-043-A.md` is the FC-1 `production_mode` batch, and its PR #680 is not merged into the pin (see WB01, `nexus_host_execution.py:57` still builds the router without `production_mode`). T-IGX-043-B (FC-2 production wiring) is approved but it covers capacity, queue worker and memory stores, not the compaction round trip. The obligation cannot hold in full at a pin that lacks FC-1. |
| OBLIGATION-043-AB-UAEP-CE (canonical context assembly and the CE-01 fail-closed guard apply on UAEP-adjacent routes exercised by XD/D2) | none (EXE not run) | **BLOCKED** | Same: the XD/D2 tests were not run, so no claim about the routes can be supported or contradicted. |

## Rationale for the aggregate

`BLOCKED`: the slice was never executed. Source reading alone cannot confirm cross-layer context and memory behaviour, and the FC-1 prerequisite is absent on the pin. Nothing contradicts the obligations, so this is not `FAIL`.

## Required follow-up (no product fix here)

1. Run the preregistered command once, with the optional extras that the four test files need, and attach the log.
2. Re-bind the obligations to a pin that contains T-IGX-043-A (#680), or state that the obligation is evaluated before FC-1 and cannot close on this pin.
3. Record the obligations against T-IGX-043-A/B in the ticket so they are not left without a finding row.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
