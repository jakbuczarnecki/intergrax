# PLAN T-IGX-036-I — PDC-P5-REG batch B-REG-T: stale-gate repair (wave B, tests only)

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-REG-T**). Diagnosis evidence: `PDC_REG_DIAGNOSIS.md`. Requirements: PDC-D3-R024, PDC-D4-R020, PDC-D6-R028.

This batch changes tests only, plus one optional doc. It touches no product path (`intergrax/`, `agents/`, `applications/`), so IIAC ACT-N11 is unaffected and the batch may merge before MCP-SERVER-X P8 (WBS gate "No product path").

## Deliverables

1. **ebh_5 (PDC-D3-R024), 4 tests.** These tests predate the EBH-7 terminal reclassification (`f913944ff`, `d4937889f`). Update three files under `tests/qualification/ebh_5/`:
   - `test_ebh_5_g06_enterprise_integrity_gates.py`: expect qualified-at-head = registry minus INTERNAL / FIXED_SUBSTRATE ids (48).
   - `test_ebh_5_proof_integrity_gates.py`: compare against `QUALIFIED_CONCERN_IDS - internal - fixed_substrate`. Replace the stale `== 5` with the registry/matrix row count and rename the test accordingly.
   - `test_ebh_5_rpl_f_integrity_gates.py`: require `rpl_f == disp.gate`; require PASS plus proof nodes only for the active RPL-F set; require `N/A — WITH EVIDENCE` with a rationale otherwise.

   The registry keeps the 8 reclassified ids, because ebh_7 depends on them. Optionally, refresh the QUALIFIED summary row and the stale gate name in `docs/project/maintainers/qualification/EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX.md`.
2. **GR-12 (PDC-D4-R020), 4 tests.** `tests/unit/applications/task_control_product_host_test_support.py` must pass `agent_contract_authority` to `build_harness_host_runtime` for STRICT PRODUCT. Prefer a `RosterAgentContractAuthority` built from the test registry projection, which is closest to production (`governed_contractor_application/host/factory.py`). The lab-compat materialization is acceptable if no projection helper exists. This mirrors the production requirement introduced in `49911bb3a`.
3. **GR-10 (PDC-D6-R028), 4 tests.** Update the AST/text gates to the structure after `716ed746f` and `6700ff756`, without weakening any invariant:
   - `test_gr10_a3_r1_uaep_checkpoint_persistence_resume_qualification.py`: replace the blanket text ban with an AST check that `agent_checkpoint_store` reaches only `run_acp_session` and never `UAEPExecutor`.
   - `test_gr10_gates.py` and `test_gr10_r5_inference_inner_governance_qualification.py`: accept the `self._pre_model_gate(adapter=..., ...)` call made before `_invoke`, and assert that `build_structured_inference_pre_model_gate` calls `enforce_pre_model_before_structured_inference`.
   - `test_gr10_r6_inference_governance_evidence_qualification.py`: point the gate at `intergrax/runtime/governance/governed_inference_composition.py`.
4. **Read-only diagnosis of the other pre-existing red qualification tests** seen in the PDC-P0 validation (`tests/qualification/ebh_7`, `dist_scale_x`, `roadmap_replay_x`). The result is a record in `PDC_REG_DIAGNOSIS.md` (verdict and introducing change for each test). **No fixes** in this ticket. Any work they need becomes new requirement rows in a later batch.
5. **Requirements matrix:** record the replay evidence for the three requirement rows (the suites are green after the fix).

## Acceptance

- `tests/qualification/ebh_5` is fully green.
- `tests/qualification/governance` is fully green.
- No change under `intergrax/`, `agents/` or `applications/`.
- No invariant is weakened: every replaced gate asserts the same guarantee against the current structure.
- `tests/qualification/ebh_7` has no new failures compared with its pre-change state.

## Test contract

```text
uv run pytest tests/qualification/ebh_5 tests/qualification/governance tests/qualification/ebh_7 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product code; the `ollama` regression (batch B-REG-P, gated); fixes for ebh_7, dist_scale_x or roadmap_replay_x; FRZ changes.
