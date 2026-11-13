# DELEGATION-PDC-P6-REPAIR — Operator commission for ticket T-IGX-036-T (integration repairs)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-P6-REPAIR |
| **Stage** | PDC-X P6 |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `DELEGATION-PDC-STANDING.md` |
| **Date** | 2026-11-13 |

## Commission (transcription)

1. All P5 batches are merged on `development`. The implementer reported that the integration repairs found in the lanes (for example `tzdata`, D3 capability flags, D2 freshness, D8 composition wiring) need product changes outside every lane write set, and that standing authorization 4 covers only plans M and S. The implementer asked for consent to a separate ticket **T-IGX-036-T**. The Operator answered in chat, in English rendering: **"Consent to ticket T, go on"** (original wording in Polish: "Zgoda na ticket T, jedź dalej").

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <this file>@<commit sha>` in **T-IGX-036-T** only, after confirming: the plan digest matches; the plan matches batch B-P6-R in `PDC_IMPLEMENTATION_WBS.md` (write set equal); the plan adds no new registry, scheduler, engine, trust, policy or audit authority; no gate is weakened; the plan states that only known red classes (IIAC activation) are tolerated at merge.
- **Write-set amendments of T-IGX-036-T** may be re-signed by a delegate when all of these hold: each added path is needed to close a finding of an XD run or of a lane report; the path is not under `intergrax/contracts/` or `intergrax/applications/_shared/` unless the plan already names that path; no gate is weakened (a gate may be strengthened or given one classification entry); no new authority and no new functional scope. Anything else returns to the Operator.
- The implementer merges the batch PR under `DELEGATION-PDC-STANDING.md` authorization 3.

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X roadmap closure; IIAC activation lines other than as stated in `DELEGATION-PDC-STANDING.md`; lifting C10/C15; edits to hotspot paths not named in the plan.

## Separation

The implementer session and its implementer subagents never write `Approved:` lines. A delegate session does not implement the ticket it signs.

## Amendment 1 — write set of B-P6-R (T-IGX-036-T) from the XD and ECAP findings

- **Basis:** the standing Operator authorization 4 in `DELEGATION-PDC-STANDING.md` (write-set amendments that close findings of the same batch) and the delegation scope above. No new Operator decision is claimed.
- **Findings source:** the B-P6 report of T-IGX-036-S (PR 514, merged as `fbad6a8de`): XD findings 1-8, ECAP-F1 and ECAP-F2.
- **Extension** (thirteen paths, none under `intergrax/contracts/` or `intergrax/applications/_shared/`): `intergrax/runtime/nexus/tools/tool_runtime.py`, `intergrax/runtime/nexus/tools/plan_context_invocation.py`, `intergrax/runtime/task/worker_bootstrap.py`, `intergrax/runtime/persistence/sqlite_opens.py`, `intergrax/runtime/persistence/sqlite_composition.py`, `intergrax/runtime/nexus/artifacts/context_reference.py`, `intergrax/runtime/nexus/artifacts/spill.py`, `intergrax/memory/default_memory_control_plane.py`, `intergrax/runtime/agent_governance/pipeline.py`, `intergrax/llm_adapters/contracts/capabilities.py`, `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json`, `intergrax/platform/enterprise_audit/data/enterprise_audit_check_manifest.v1.json`, `tests/qualification/ebh_5/_sqlite_touchpoint_registry.py`.
- **Not extended, returned to the Operator:** XD finding 9 and 10 (new functional scope; finding 10 also lies under `intergrax/contracts/`).
- **Effect on the signature:** the plan digest of T-IGX-036-T changed, so the earlier `Approved:` line no longer applies. The ticket returns to `plan_pending`.
- **Delegation:** the delegate session may re-sign **T-IGX-036-T only**, under the same checks as above, additionally confirming that the plan's write set equals the amended WBS write set and that the plan adds no new registry, scheduler, engine, trust, policy or audit authority and weakens no gate.
- **Not delegated** stays as listed above.

## Amendment 2 — XD stand-in cases and wave of B-P6-R (T-IGX-036-T)

- **Basis:** the standing Operator authorization 4 in `DELEGATION-PDC-STANDING.md` and the delegation scope above. The Operator decision "A separate ticket" (chat, original wording in Polish: "Osobny ticket") split XD findings 9 and 10 out of this batch into T-IGX-036-U; see `DELEGATION-PDC-P6-MODEL.md`.
- **Extension** (five test files, none under a hotspot): `tests/qualification/pdc_x/xd/test_xd_02_agent_context.py`, `test_xd_03_agent_governance.py`, `test_xd_09_artifacts_context.py`, `test_xd_10_artifacts_security.py`, `test_xd_12_evol_ent_audit.py`. They hold the strict `xfail` cases of XD findings 1-8, which would fail the suite once the repairs land; only the marker is removed.
- **WBS:** B-P6-R moves from wave W4 to W5 and depends on B-P6, because it now writes files under the write set of B-P6 (T-IGX-036-S, already merged). No other batch changes.
- **Effect on the signature:** the plan digest of T-IGX-036-T changed, so the earlier `Approved:` line no longer applies. The ticket returns to `plan_pending`.
- **Delegation:** the delegate session may re-sign **T-IGX-036-T only**, under the same checks as above, additionally confirming that the added paths are tests only and that no assertion is weakened.
- **Not delegated** stays as listed above.

## Amendment 3 — import cycle and launcher stub (T-IGX-036-T)

- **Basis:** the standing Operator authorization 4 in `DELEGATION-PDC-STANDING.md` and the delegation scope above; each added path closes a finding of the implementer report for items 1-7 of the same batch (PR 517).
- **Extension** (two paths, neither under a hotspot): `intergrax/runtime/observability/export_boundary.py` (remove two redundant module-level imports that are already imported under `TYPE_CHECKING`, closing the second import cycle of item 5) and `tests/qualification/pdc_x/d4/test_d4_no_product_vocabulary.py` (stub path of the launcher follows item 4; no assertion weakened).
- **Effect on the signature:** the plan digest of T-IGX-036-T changed, so the earlier `Approved:` line no longer applies. The ticket returns to `plan_pending`.
- **Delegation:** the delegate session may re-sign **T-IGX-036-T only**, under the same checks as above, additionally confirming that the two added paths are the only change and that no assertion is weakened.
- **Not delegated** stays as listed above.
