# EQO-W3-WB09-AUD — session E verdict (wave AUD-2)

**Verdict (WB09): `FAIL`**
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
| `pytest_command_sha256` recomputed from `pytest_command` (UTF-8) | MATCH `a9361ab1…` (manifest = DES_PREREG) |
| Log vs manifest summary | MATCH: 1 failed, 65 passed in 11.30s |
| Log records command, node ids or its own hash | NO (summary lines only; log hash not verifiable) |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |

**What the slice executes:** SKILL-X batch C (versioning, provenance, recovery), SKILL host lifecycle batch A, the MCP-SERVER-X P5 wrapper and the PROD-Q strict-serving wiring gate. The 65 passes include the PROD-Q gate. The single failure is the same P5 wrapper as in WB04: its first target, `test_mcp_trust_a_principal_binding.py`, returned `exit=2` (likely a missing `mcp` extra; the log does not say). The slice's skills tests never reference `disable_skill`, `remove_skill` or `upgrade_skill`.

## Per-finding verdicts

| Finding | Interim DIS | AUD verdict | Rationale (evidence at pin) |
|---------|-------------|-------------|-----------------------------|
| EQOA-A2-027 | BLOCKED | **FAIL** | Stale scope note confirmed in source. `intergrax/skills/host_lifecycle.py` defines `disable_skill` (:112), `remove_skill` (:149) and `upgrade_skill` (:184). `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md:487` still carries the SKILL-X scope note that disable/remove/upgrade are unsupported and untested (`CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md:149,189`, F-04). Acceptance stays bound to `552e3798`, not to the changed code, and the slice does not exercise those methods. The PROD-Q half of the hypothesis is not reproduced: the strict-serving wiring gate passes at the pin, so that regression is no longer present, but that does not close the stale scope note or the requalification of the changed lifecycle surface. Route: owner updates the scope note and requalifies, or states the surface is outside the accepted scope. |
| EQOA-A4-040 | BLOCKED | **FAIL** | Defect reproduced by reading source. Direct `uvicorn.run` serving entrypoints at the pin: `applications/attestation_demo/host/main.py:19`, `applications/intergrax_assistant_application/host/main.py:19`, `applications/lab_application/host/main.py:22`, `applications/poc_template_application/host/main.py:19`, `applications/research_application/host/main.py:98` and `applications/governed_contractor_application/host/package_b_strict_production_serving_command.py:159`. Only `applications/local_workspace_application` imports the hosting lifecycle engine (`intergrax.hosting`). "Most Tier-3 entrypoints" holds. The strict-serving gate that passes covers the governed contractor package-B command, not the other five. The P5 wrapper failure is not related. Write set needs an operator decision on the hosting-engine adoption scope for reference apps. |

## Rationale for the aggregate

`FAIL` because both findings are confirmed by source at the pin. The P5 failure (EXE exit 1) is an environment or collection problem shared with WB04 and does not bear on these two findings.

## Required follow-up (no product fix here)

1. A2-027: owner updates the SKILL-X scope note (roadmap L487) and runs a targeted replay of `host_lifecycle.py` disable/remove/upgrade and the pinning wiring, or records them as outside scope.
2. A4-040: operator decides whether reference Tier-3 apps must start through the hosting lifecycle engine or are excepted with scope and reopen condition.
3. Re-run the P5 wrapper with the `mcp` extra (shared with WB04) so the slice has no unrelated red.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
