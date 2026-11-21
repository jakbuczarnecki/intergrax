# EQO-W3-WB03-AUD — session E verdict (wave AUD-1)

**Verdict (WB03): `FAIL`**
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
| `pytest_command_sha256` recomputed from `pytest_command` (UTF-8) | MATCH `dcab8127…` (manifest = DES_PREREG) |
| Log vs manifest summary | MATCH: 39 passed |
| Log records command, node ids or its own hash | NO (summary lines only; log hash not verifiable) |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |

**Disposition quality:** all four DIS rows carry the same text ("band-C pytest slice passes on TRAIN_PIN; dossier hypothesis not fully falsified by this probe alone"). The DES hypotheses are defect claims about default or no-op behaviour. A green slice does not falsify them unless it drives the default path. `ACCEPTED_SCOPED_LIMITATION` also names no scope, owner or reopen condition.

## Per-finding verdicts

| Finding | Interim DIS | AUD verdict | Rationale (evidence at pin) |
|---------|-------------|-------------|-----------------------------|
| EQOA-A4-026 | ACCEPTED_SCOPED_LIMITATION | **FAIL** | Hypothesis confirmed. `intergrax/runtime/interactions/verification/factory.py:37` has `mode = InboundVerifierMode.NONE` as default. Lines 52-56 map an unset or invalid mode to `NONE` (`except ValueError: resolved_mode = InboundVerifierMode.NONE`). `verification/contract.py:21` documents the no-op verifier as the "default for laboratory paths". The slice's only verification test (`test_interaction_intake_service.py:162`, "verification failed") uses an injected verifier. No test drives the default or invalid-mode path. |
| EQOA-A4-031 | ACCEPTED_SCOPED_LIMITATION | **FAIL** | Hypothesis confirmed. `intergrax/runtime/nexus/uaep/uaep_executor.py:193-194` takes `context_engine` and `llm_adapter` as optional, defaulting to `None`. Line 386 guards canonical assembly with `if self._context_engine is not None and self._llm_adapter is not None:` and has no else/raise, so the UAEP path continues without assembly. The slice's CE-01 test (`test_mem_xint6_cross_layer_e2e_certification.py:477`) covers `enforce_context_engine_when_provider_sources_active(state)` on `RuntimeState`. That is a different function, not the UAEP path. |
| EQOA-A4-037 | ACCEPTED_SCOPED_LIMITATION | **FAIL** | Hypothesis confirmed. `intergrax/runtime/nexus/orchestration/human_response.py:91-92` has `if human_store is None: return`. `nexus_loop.py:246,317` makes `human_decision_store` optional (default `None`). A search of `intergrax/`, `applications/` and `agents/` found no production construction site that passes `human_decision_store` into the loop. The only matches are the unrelated tool-registry binding. The slice's governance tests inject a `SQLiteHumanDecisionStore` explicitly (`test_nexus_loop_governance.py:129-203`), which covers the configured path and not the production default. |
| EQOA-A3-027 | ACCEPTED_SCOPED_LIMITATION | **BLOCKED** | FRZ-PRD-07 is `OPEN` in `ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md:503` and requires replay of the D4 and P2-N02 nodes at the pin to decide the strict-host defect. The slice's D1 lifecycle test file (20 tests) has no strict-path assertion (no `strict` token). The PASS candidate cannot be confirmed or refuted from this packet. |

## Rationale for the aggregate

`FAIL` because three findings (A4-026, A4-031, A4-037) describe silent no-op or skipped-guard behaviour that is still present in source at the pin. The passing slice exercises only the explicitly configured paths, so the interim `ACCEPTED_SCOPED_LIMITATION` is not supported by any evidence in the packet. A3-027 needs a strict-host replay.

## Required follow-up (no product fix here)

1. A4-026: owner decision. Either fail closed on unset or invalid verifier mode in strict hosts, or record a scoped lab-only limitation with an owner and reopen condition. Add a default-path negative test.
2. A4-031: owner decision. Either apply the CE-01 fail-closed guard on the UAEP path, or record the limitation. Add a UAEP-path negative test.
3. A4-037: owner decision. Wire a production human decision store or fail closed, with a production-construction test. Track via T-IGX-043-E and the #680 probe.
4. A3-027: replay D4 and P2-N02 nodes at the pin with strict fixtures.
5. After the pin moves, run `EQO-W3-WB03-DRV`. Do not open a new campaign.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
