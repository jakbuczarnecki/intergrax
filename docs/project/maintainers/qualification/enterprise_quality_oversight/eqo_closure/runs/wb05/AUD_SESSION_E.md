# EQO-W3-WB05-AUD — session E verdict (wave AUD-1)

**Verdict (WB05): `BLOCKED`**
**Auditor session:** independent session E (not the EXE agent; no EXE artefact in this WB authored by this session).
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
| TRAIN_PIN in manifest / packet = `28c27cb8…` | YES (object exists locally) |
| `pytest_command_sha256` | NOT PRESENT. WB05 has no `DES_PREREG.json` and no command hash, so there is no preregistration to verify the command against. |
| Log vs manifest summary | MATCH: 14 passed |
| DIS/EVD artefact | NOT PRESENT. The packet lists none. The only disposition is `disposition_note` in the manifest: "does not close dossier hypotheses". |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |

**What the slice executes:** static gates only: BND-03 (Tier-0 upward edges), BND-03b (product manifest registry), RW-BND-05 (provider switch closed list), Tier-0 context import boundary, EBH-5 RPL-G regression. The log has no command or node ids.

## Per-finding verdicts

| Finding | Dossier class | AUD verdict | Rationale (evidence at pin) |
|---------|---------------|-------------|-----------------------------|
| EQOA-A2-009 | CONFIRMED_DEFECT | **PASS** (residual R1) | The original defect (gate scans only `intergrax/`) no longer holds. `test_rw_bnd05_provider_switch_gate.py:23` scans `intergrax/runtime`, `intergrax/applications` and `applications`, against a closed `CLOSED_PROVIDER_ROOTS` list classified ROOT/CONST/IMPL/DEMO_PROOF/SHARED_DEMO. That is the owner's recommended "explicit sanctioned-exception list". `agents/` has 0 files importing `intergrax.integrations._shared` or `.providers`. **R1:** `agents/` is not a scan root, so it is clean but has no regression guard. The literal claim "leakage = 0" must be reworded to "leakage limited to the closed list". |
| EQOA-A4-020 | ARCHITECTURE_RISK | **PASS** (residual R2) | The finding recommended classifying the provider-import files into sanctioned composition, constant-only and bypass, with a gate allowlist. The RW-BND-05 closed list does this for the Tier-1 runtime and Tier-3 roots, and the gate passes at the pin. **R2:** the list must stay owner-controlled. Adding a root is an owner decision, not a test edit. |
| EQOA-A7-021 | UNVERIFIED_HYPOTHESIS | **PASS** | Not reproducible at the pin. A source scan of `intergrax/` for `from/import agents` or `applications` found no static upward import. The BND-03 gate (Tier-0 must not import agents/applications) and EBH-5 `test_ebh5_rpl_g_tier0_no_agents_import_applications` pass. This covers static imports only; dynamic loading is under A4-007. |
| EQOA-A4-007 | UNVERIFIED_HYPOTHESIS | **BLOCKED** | Dynamic loaders exist at the pin: `intergrax/agents/readiness/scoreboard.py:86` (`importlib.import_module(import_path)`), `intergrax/agents/authoring/__init__.py:36-38` (`importlib.import_module(module_path)`) and `intergrax/applications/_shared/migration_wiring.py`. The slice has only static gates, which cannot see config-driven loaders. The manifest says the same. There is no adversarial runtime proof that the seams cannot be driven to import Tier-2/3 outside the sanctioned registries. |
| EQOA-A3-011 | UNVERIFIED_HYPOTHESIS | **BLOCKED** | 4 `hasattr(` hits remain in `intergrax/rag/vectorstore` (for example `hybrid/lexical_hybrid.py:64,116` and `soak/prod_slo.py:328`) against FRZ-TYP-04 "reflection-based dispatch = 0". The slice has no vectorstore test (the context boundary test is unrelated) and nothing classifies the probes as semantic or non-semantic. |

## Rationale for the aggregate

`BLOCKED`: three findings (A2-009, A4-020, A7-021) have supporting static-gate evidence at the pin. A4-007 and A3-011 cannot be decided from this slice. The WB also has no preregistration (DES_PREREG, command hash) and no DIS artefact, so the checklist item "verify pytest log hash / command" cannot be completed and the findings were never formally dispositioned. No evidence contradicts a stated disposition, so this is not `FAIL`.

## Required follow-up (no product fix here)

1. Create `DES_PREREG.json` and a DIS record for WB05 that cites this verdict. The integrator or EXE owner does this in a later wave, not this session.
2. A4-007: add an adversarial runtime proof for the loader seams, or an owner decision that sanctions the registries as the only loadable roots.
3. A3-011: classify the 4 `hasattr` probes (semantic or non-semantic) and record the result.
4. R1: add `agents` to the gate scan roots, or record why it is excluded.
5. R2: record the closed list as owner-controlled in the gate documentation.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
