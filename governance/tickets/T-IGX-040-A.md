# T-IGX-040-A — PDC-X batch B-P6-J: MCP-RC handoff repairs, F-49 (`openai` catalog row) and F-41 (impact map for the Tier-3 `applications/` root)

## Scope

Two product repairs requested by the MCP-RC session (handoff `MCP_SERVER_X_RC_PDC_X_HANDOFF_F49_F41.md` on `mcp-rc`, Operator decision `OD-MCP-RC-05`, a one-time exception to the freeze of product paths): (1) the single `openai` catalog row dispatches on `integration_category` like the `ollama` row (T-IGX-036-X), with pinned tests and test-only pins for the ollama wrapper (F-50); (2) a rule for the root `applications/` in the ENT-AUDIT impact map and its generated SSOT, in a second typed table so that the existing p6_impact and xd_12 tests stay unchanged. The other shared-slug rows (`hf`, `vllm`, `llama_cpp`) were surveyed and have no defect; they are not changed. Product code: yes (`intergrax/integrations/providers/embedding_provider/openai/register.py`, `intergrax/platform/enterprise_audit/impact_map.py` and its SSOT JSON). Plan: `PLAN_T-IGX-040-A.md`.

## Plan digest

Plan digest: 007451efdb18b21831c06112a38f884ac4b127fe1bcf3c3121009deadb774da7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-040-A.md`](plans/PLAN_T-IGX-040-A.md).

## Approved

(pending; the plan is signed by the Operator or by a delegate session whose commission covers a request that originates in the MCP-RC campaign; the signer also states which merge gate governs, see the plan, Protocol item 5; the implementer never writes the approval line)

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

plan_pending
