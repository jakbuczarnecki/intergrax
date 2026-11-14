# T-IGX-040-A — PDC-X batch B-P6-J: MCP-RC handoff repairs, F-49 (`openai` catalog row) and F-41 (impact map for the Tier-3 `applications/` root)

## Scope

Two product repairs requested by the MCP-RC session (handoff `MCP_SERVER_X_RC_PDC_X_HANDOFF_F49_F41.md` on `mcp-rc`, Operator decision `OD-MCP-RC-05`, a one-time exception to the freeze of product paths): (1) the single `openai` catalog row dispatches on `integration_category` like the `ollama` row (T-IGX-036-X), with pinned tests and test-only pins for the ollama wrapper (F-50); (2) a rule for the root `applications/` in the ENT-AUDIT impact map and its generated SSOT, in a second typed table so that the existing p6_impact and xd_12 tests stay unchanged. The other shared-slug rows (`hf`, `vllm`, `llama_cpp`) were surveyed and have no defect; they are not changed. Product code: yes (`intergrax/integrations/providers/embedding_provider/openai/register.py`, `intergrax/platform/enterprise_audit/impact_map.py` and its SSOT JSON). Plan: `PLAN_T-IGX-040-A.md`.

## Plan digest

Plan digest: 007451efdb18b21831c06112a38f884ac4b127fe1bcf3c3121009deadb774da7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-040-A.md`](plans/PLAN_T-IGX-040-A.md).

## Approved

Approved: operator-delegate @ 2026-11-14T11:10:18+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-RC-HANDOFF.md@23575650a9e8072cfc5296d505cc90fef80ea84b

Delegate checks (commission of `DELEGATION-PDC-RC-HANDOFF.md`, all passed before this line was written):

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above.
- The plan write set equals the write set of batch B-P6-J in `PDC_IMPLEMENTATION_WBS.md` (five paths, identical). No write path lies under `intergrax/contracts/`, `intergrax/applications/_shared/`, `intergrax/integrations/registry/`, `scripts/governance/` or `.github/`, and none of the 23 MCP-RC paths of the handoff (`MCP_SERVER_X_RC_PDC_X_HANDOFF_F49_F41.md` at `c2fa6ebb9`, read without checking out `mcp-rc`) is in it.
- The plan keeps `test_xd_12_evol_ent_audit.py` and `test_p6_impact_map_coverage.py` unedited (new files only; the Tier-3 root rule goes into a second typed table), narrows no gate, keeps one catalog row per slug, adds no registry, scheduler, engine, trust, policy or audit authority, and excludes certification (Protocol item 7, Out of scope); it is a repair plan, not a certification verdict.

Merge gate: `OD-PDC-SEQ-01` (Option A, product merges wait for the MCP-SERVER-X P8 publication) governs, and it is satisfied. Evidence: the roadmap on `development` shows MCP-SERVER-X as `CLOSED - INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) - PUBLISHED (SCOPED)`; its final closure report is merged (PR 480, merge `c11184164af3072d0a48102a5e8d62adfec53695`, an ancestor of `development`); P8 audit `PASS_SCOPED`, certification `CERTIFIED_PASS_SCOPED`, terminal verification `P8_TERMINAL_VERIFIED`. That is the condition of authorization 3 of `DELEGATION-PDC-STANDING.md`, under which the implementer merges this batch pull request (green CI, only the known IIAC activation red class, head equals the verified SHA, normal merge for conflicts) and decision 8 of `DELEGATION-PDC-WAVE2.md`. Earlier product merges of PDC-X (for example PR 537) are consistent with this but are not the basis. `OD-MCP-RC-05` is recorded on `mcp-rc` only; it is a scheduling request of the MCP-RC campaign (merge before its rebind, both repairs in one pull request, then product paths frozen again) and adds no gate and no authority. Write-set amendments are not delegated and return to the Operator.

Merge order note: the F-41 root rule for `applications/` is also what T-IGX-036-Z needs for `applications/local_workspace_application/tests/test_run_artifact_metadata.py`, so the orchestrator merges B-P6-J before B-P6-I (the WBS lists J after I).

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_ecapfix tests/qualification/pdc_x/p6_impact tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py tests/qualification/pdc_x/reg_p tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py tests/unit/integrations/test_p2_002_b2_embedding_provider_catalog.py tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py tests/qualification/compat_x/test_compat_x_plugin_provider_r4_gates.py -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

Plus the generator run and the throwaway MCP-RC tree check described in the plan (Deliverable 6).

## Out of scope

- `scripts/governance/`, the IIAC activation record, `.github/`, `intergrax/contracts/`, `intergrax/applications/_shared/`, the registry package `intergrax/integrations/registry/`, FRZ files, P8 originals and the 23 MCP-RC paths (byte-for-byte equal to `b0e19bf61`).
- Single-write registration (F-52), F-51 behaviour changes, hardening F-43 and F-44, other providers and categories, any change of the 85-check manifest, changes to the impact-map generator script.
- The EBH-7 re-pin, IIAC activation (PDC-P7-00), MCP-RC activation and rebind, OD signatures, certification.

## Lifecycle state

plan_approved
