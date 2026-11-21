# EQO-W3-WB02-AUD — session E verdict (wave AUD-1)

**Verdict (WB02): `FAIL`**
**Auditor session:** independent session E (not the EXE agent; no EXE/DIS artefact in this WB authored by this session).
**Date:** 2026-11-21
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469` (source read at the pin, read-only)
**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)
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
| `pytest_command_sha256` recomputed from `pytest_command` (UTF-8) | MATCH `3047a03f…` (manifest = DES_PREREG) |
| Log vs manifest summary | MATCH: 66 passed, 1 skipped |
| Log records command, node ids or its own hash | NO (summary lines only; log hash not verifiable) |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |

**What the slice executes:** the 66 passes are 1 tool-invoker test, 1 SEC-ENT middleware test and the rest TENANT-X P0 fast carrier/classifier gates. The only skip is the whole `test_vectorstore_cross_tenant_isolation.py` module (module-level `importorskip("qdrant_client")`), so 3 vectorstore isolation tests produced no execution evidence.

## Per-finding verdicts

| Finding | Interim DIS | AUD verdict | Rationale (evidence at pin) |
|---------|-------------|-------------|-----------------------------|
| EQOA-A4-006 | ACCEPTED_SCOPED_LIMITATION | **FAIL** | Hypothesis confirmed. `scripts/maintenance/check_runtime_event_tenant_propagation.py:15` still lists `intergrax/agents/uaep.py`, which does not exist at the pin. The script is referenced only in `scripts/ci/script_paths.py:211`. No test in the slice invokes it, and no CI/test path does either. The slice (tool invoker, middleware, TENANT-X P0 fast) does not exercise UAEP event tenant propagation, so "limitation" is unsupported. Route: owner re-binds the gate path (as in T-IGX-041-B), then replay. |
| EQOA-A4-027 | NOT_VERIFIED | **FAIL** | Defect reproduced by reading source. `applications/research_application/serving/fastapi_router.py:46-54` takes `tenant_id = body.tenant_id` when no authenticated principal is present. `applications/research_application/serving/schemas.py:9` defaults `tenant_id: str = "research-tenant"`. A missing tenant silently becomes a fixed default, which is the FRZ-TEN-10 failure mode while FRZ-TEN-10 is recorded PASS. No test in the slice covers this. Write set needs OD-EQO-08 (serving-router ticket). |
| EQOA-A7-009 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | Label drift confirmed: `EBH_7_GAP_REGISTER.md:5` says `STRUCTURAL — N/A with evidence (not E2E)`. `EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md:141` (G09) says `PASS` with the note "Structural N/A evidence only — not platform E2E proof". No pytest in the slice replays E2E for the five `unknown_role` surfaces. Closure needs the owner/operator accepted-limitation record (owner, scope, reopen condition; EDER-D-0646 `ACCEPTED_AS_STATED_RECORD_REQUIRED`) plus label standardisation. |
| EQOA-A7-028 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | EBH-7 is `CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)` (OD-EBH7-08); the production-delta residual is not replayed at the pin (manifest blocker `EBH7_TENANT_E2E_NOT_REPLAYED`). The slice cannot decide it; it needs an EBH-7 replay or an operator disposition. |
| EQOA-A7-031 | BLOCKED | **BLOCKED** | Agree with EXE. The skips are `pytest.skip("... run in tenant_platform_e2e module" / "... dedicated fixture test")` in `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py:27,29`. No routed-proof replay was executed. |
| EQOA-A8-006 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | Gap confirmed: 24 of 73 `docs/project/architecture/*.md` contain no occurrence of "tenant". The finding is LOW and NON_BLOCKING_WITH_EVIDENCE. The DIS honestly says pytest cannot close it. Closure needs an owner doc patch (tenant/security section or pointer to the TENANT-X record) or an acceptance record. |

## Rationale for the aggregate

`FAIL` because two findings (A4-006, A4-027) are contradicted by source at the pin and the slice gives them no coverage. The remaining four are not decidable by this slice. EXE `GREEN` (exit 0) reflects only that the selected tests pass. It does not close any hypothesis in `DES_PREREG.json`.

## Required follow-up (no product fix here)

1. A4-006: owner updates the gate path and re-runs the propagation audit, then replay on a new pin.
2. A4-027: operator decision OD-EQO-08 (serving-router write set) or a recorded FRZ-TEN-10 scope exception for reference Tier-3 apps.
3. A7-009, A7-028, A7-031: operator/owner EBH-7 disposition and label standardisation, or replay of the tenant E2E routed proofs.
4. A8-006: owner doc patch or acceptance record.
5. Vectorstore isolation: execute under the `dev-ci-rag` extra (rag-guard profile) and attach the log.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
