# EQO-W3-WB01-AUD — session E verdict (wave AUD-2)

**Verdict (WB01): `FAIL`**
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
| TRAIN_PIN in manifest / packet = `28c27cb8…` | YES (object exists locally) |
| `pytest_command_sha256` | NOT VERIFIABLE: WB01 has no `DES_PREREG.json` and the manifest carries no hash, so nothing was preregistered |
| Log vs manifest summary | MATCH: 1 failed, 23 passed, 1 skipped |
| Log records command, node ids or its own hash | NO (summary lines only; log hash not verifiable) |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |
| Findings in scope | The packet lists none; the five IDs come from `GOVERNANCE_EVIDENCE.md` and `EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl` (WB01 probes) |

**What the slice executes:** only `tests/unit/governance/test_iiac_activation_gate.py`. It tests IIAC policy activation, not any of the five findings. The one failure is `test_canonical_policy_is_in_force_and_verifies` (`IIAC policy activation: FAIL`). That matches the known `NOT_IN_FORCE` state on the train (EQOA-A0-007, X-03) and is not decided here. The skip is a symlink test (`symlink creation is unavailable`, a Windows limit).

## Per-finding verdicts

| Finding | Interim DIS | AUD verdict | Rationale (evidence at pin) |
|---------|-------------|-------------|-----------------------------|
| EQOA-A1-021 | ACCEPTED_SCOPED_LIMITATION | **FAIL** | Hypothesis confirmed. `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md:1939` still says "Independently audited establishment of the enterprise-freeze control system" and cites only commit `324ad07d`. A search of `docs/` and `governance/` at the pin finds that SHA only in the roadmap, the FRZ checklist (EBH-2F-R1 row) and `ROADMAP_REPLAY_X_HISTORICAL_STAGE_MATRIX.md:28`, which rates it `RULE-ACCEPTED-CHILD`. No audit report exists. The disposition names no scope, owner or reopen condition. Route: owner relabels L1939 or cites the audit record (DOC_PATCH). |
| EQOA-A4-044 | BLOCKED | **FAIL** | Defect reproduced by reading source. `intergrax/runtime/execution/nexus_host_execution.py:57-60` builds `AgentRouter(nexus_loop.registry, event_bus=nexus_loop.event_bus)` with no `production_mode`, so it takes the default `False` (`agent_router.py:49`). `NexusLoop` passes `production_mode=production_mode` to its own router (`nexus_loop.py:361-364`). Host-task routing therefore ignores `production_mode`. T-IGX-043-A / #680 is not on the pin, so the fix is not present. The slice has no routing test. Route: land #680 (T-IGX-043-A), then replay. |
| EQOA-A4-015 | BLOCKED | **BLOCKED** | Relaxed construction exists at the pin: `agents/organization_worker/organization_worker_agent.py:114` hard-codes `production_mode=False`, `agents/boundary_demo/boundary_demo_agent.py:87` uses the lab config, and `agents/external_contractor_adapter/external_contractor_adapter_agent.py:113` uses `harness_production_mode()`. Whether any production host reaches these paths is the open question, and no test in the slice answers it. It also depends on #680. |
| EQOA-A5-003 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | No gate-weakening review for the GOV-X1 rewrite was found under `docs/`, `governance/` or `tests/` at the pin. The hits are the findings register and unrelated rows. Proving a negative needs the owner (PDC-X D6 / ARCH-FREEZE) to produce the review or record the limitation with scope and reopen condition. LOW severity. |
| EQOA-A7-030 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | `CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md` (:211 and :216) records "re-audit authorised after correctives" and "audit obligation stays attached". It names no obligation owner, target SHA or date for DS-3, SSOT-D2, SSOT-D3, F-03 or F-04. Closure needs an owner/operator assignment. The slice does not touch it. |

## Rationale for the aggregate

`FAIL` because two findings (A1-021, A4-044) are confirmed by source at the pin and the slice gives them no coverage. The other three need owner or operator input. The EXE result (1 failed, exit 1) concerns IIAC activation and says nothing about the five hypotheses.

## Required follow-up (no product fix here)

1. A4-044 and A4-015: land T-IGX-043-A / #680 on the train, then replay host-task routing and the three Tier-2 `RuntimeContext` builders.
2. A1-021: owner relabels `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md:1939` as a documentation publication, or cites the audit record.
3. A5-003: owner supplies the GOV-X1 gate-weakening review or a scoped acceptance record.
4. A7-030: operator assigns an owner, target SHA and date for each attached audit obligation.
5. IIAC activation (X-03): handled in the RB-2 track, not by this WB.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
