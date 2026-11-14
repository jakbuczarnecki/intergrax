# Handoff to PDC-X: two product repairs on `development` before the MCP-RC activation re-pin (F-49, F-41)

| Field | Value |
| --- | --- |
| **From** | MCP-RC orchestrator (campaign `T-IGX-038-A..F`, branch `mcp-rc`) |
| **To** | PDC-X owner and ENT-AUDIT owner |
| **Operator decision** | `OD-MCP-RC-05` (2026-11-14): "Napraw przed rebindem" for F-49 and "Reguła przed rebindem" for F-41 (English renderings: "Fix before the rebind", "Rule before the rebind"). A one-time exception to the freeze of product paths on `development`. |
| **Why it matters** | The MCP-RC activation of the integrated baseline is waiting. Each product-path change on `development` after the audited tree needs one more audit addendum and a re-pin, so both repairs must land together, then `development` must be frozen again. MCP-RC does not edit these files (outside its ticket write sets). |
| **Status** | Request. Nothing here is implemented, audited or approved by MCP-RC. |

Sources (claims to check, not truth): independent audit `PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_2.md` (F-41) and `_ADDENDUM_3.md` (F-49, F-50), certifier `PDC_P7_00_CERTIFIER_PREFLIGHT_ADDENDUM_2.md`. All on branch `mcp-rc`.

## Repair 1: F-49, the `openai` catalog row has the defect that T-IGX-036-X fixed for `ollama`

**Finding.** The `openai` slug has one catalog row owned by `managed_retrieval/openai` (factory `create_openai_managed_retrieval`). `embedding_provider/openai/register.py` only augments that row with the embedding category and contract specs (`augment_integration_contract_specs("openai", ...)`) and leaves the owning factory in charge of both categories. Result at `development`: `resolve(EMBEDDING_PROVIDER, "openai")` raises `IntegrationConfigurationError` without a key and `TypeError ... returned OpenAIManagedRetrievalIntegration, expected EmbeddingProvider` with a placeholder key (reproduced by the certifier, `iiac_evidence/PDC-X/P7-00/certifier/addendum_2/logs/openai_sibling_probe_certifier_at_bb78a7432.json`).

**Requested change** (the same shape as `f209febbe`, T-IGX-036-X, PR #537): in `intergrax/integrations/providers/embedding_provider/openai/register.py` wrap the factory of the single `openai` row so that `resolve` dispatches on `integration_category`: the embedding category builds `create_openai_embedding_provider_integration` (already in `embedding_provider/openai/bundle.py`), every other category keeps the owning factory with unchanged keyword arguments; register the wrapped entry with `override=True` after the augmentation, as in `ollama/register.py`.

**Acceptance.**
- Pinned tests next to `tests/qualification/pdc_x/p6_ecapfix/` mirroring the ollama ones: embedding category resolves to the embedding provider integration; the managed_retrieval category is unchanged; the embedding registration without the owning row still fails closed; a catalog with both contributions keeps one `openai` row.
- Add the pins the auditor found missing for the ollama wrapper too (F-50, test-only, optional but cheap): `enabled` default `False` and pass-through, keyword arguments reach the owning factory. Check the survivors of the auditor's mutation sample (`iiac_evidence/PDC-X/P7-00/addendum_3/`).
- No new import beyond the openai embedding bundle; no credential, endpoint, tenant or network handling added (these are the audit criteria; F-51 and F-52 list the known soft spots of the pattern: `enabled` goes through `bool()`, other keywords are dropped in the embedding branch, and the two-step registration has a narrow window; consider writing the wrapped entry in one write).
- Catalog revision generation and digest unchanged (the digest excludes factories); the 201 (slug, category) comparison of the auditor shows exactly the intended single change for `openai`.
- Any other shared-slug row with the same pattern (the auditor checked `openai`; check `hf`, `vllm`, `llama_cpp` and the other `embedding_provider/*/register.py`) gets the same treatment or an explicit statement why not.

## Repair 2: F-41, the ENT-AUDIT impact map has no rule for the Tier-3 `applications/` root

**Finding.** `intergrax/platform/enterprise_audit/impact_map.py` and the shipped SSOT `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json` (PR #525, T-IGX-036-V) have prefixes only for `intergrax/*`. The eight modules `applications/<app>/mcp/server.py` (apps: `dispute_sim_application`, `governed_contractor_application`, `intergrax_assistant_application`, `lab_application`, `legal_application`, `local_workspace_application`, `poc_template_application`, `research_application`) select only the invariant floor. On a tree where they are in the PDC-touched range (any tree that contains the MCP-RC change), `tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py::test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check` and `tests/qualification/pdc_x/p6_impact/test_p6_impact_map_coverage.py::test_every_pdc_touched_product_module_maps_to_at_least_one_check` fail ("8 PDC-touched product modules map to no check"). They pass on `development` alone only because those files are unchanged there.

**Requested change.** Add a rule for the `applications` root (at least `applications/<app>/mcp`, preferably the root) mapping to existing manifest check ids that fit Tier-3 application code. The existing `intergrax/applications` rule is a starting point (boundary, public contract, typing and security-audit-trail checks); choose from the 85-check manifest, do not invent ids, and regenerate the shipped JSON with the generator so `test_shipped_ssot_equals_generator_output` holds.

**Acceptance.**
- Both tests above pass on a tree that contains the MCP-RC change (test locally on `origin/mcp-rc`; run both tests on `development` alone too).
- The tests are not narrowed, skipped or xfailed (a narrowed test would weaken a gate; the auditor says so explicitly).
- The auditor's probes still hold: no check is dropped or rerouted for any tracked file; every id exists in the manifest; entrypoint files exist.
- Optional hardening the auditor suggests (F-43, F-44) is not required for the rebind.

## What happens after you land both

1. Tell the MCP-RC orchestrator that both are merged on `development` and that product paths are frozen again. A product-path commit after that voids the pin again.
2. MCP-RC then merges `development` into `mcp-rc`, the independent auditor writes addendum 4 for every product path changed since `0fc8badd8` (the `openai` register, the impact map and its JSON, any shared-slug siblings), the certifier adds a preflight addendum, the activation certifier re-pins the record, then the PR and the five CI contexts. Expect two to three hours of session time.
3. Please also say whether F-35 (ten COMPAT-X event-evolution gates), F-09 (pytest does not exit after `test_xd_03_agent_governance.py`) and F-50 have owners; they are not required for the rebind but they stay open in the audit record.

## Not requested

No change to `scripts/governance/`, the activation record, `.github/`, MCP-RC paths (`applications/*/mcp/server.py` and `intergrax/applications/_shared/mcp_*`, which are audited and must stay byte-identical to `b0e19bf61`), FRZ files or P8 originals. No ruleset change.
