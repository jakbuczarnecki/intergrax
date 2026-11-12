# PLAN T-IGX-036-P — PDC-P5 batch B-D6: D6 Runtime Security

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-D6**: fail-closed signing and poisoning defaults, single tool-input defense path, durable audit trail, secret gate, plugin provenance). Decision record: ADR-PDC-06 (Accepted) and OD-PDC-SCOPE-07 = A. Requirements: PDC-D2-R008, PDC-D6-R004, R006, R007, R008, R009, R013, R014, R015, R016, R020, R021, R023.

This batch changes product code. It is implemented in an isolated worktree and **does not merge before MCP-SERVER-X P8 is published** (OD-PDC-SEQ-01). A merge invalidates the IIAC activation (ACT-N11), so PDC-P7-00 re-activates it.

## Deliverables

1. **Signing fails closed (R008).** The literal fallback `"harness-dev-signing-key"` in `critical_action_signing_wiring.py` is removed. A missing signing secret raises a typed configuration error outside lab profiles. Lab profiles keep an explicit, named opt-in; there is no environment-check fallback inside product code.
2. **Poisoning defense on by default (R014, PDC-D2-R008).** `retrieval_poisoning_defense_enabled` defaults to enabled when the profile is absent in `tools/providers/rag/service.py` and the security assembly; disabling requires an explicit profile value. Test: a missing profile still enforces the filter.
3. **One tool-input defense path (R013).** `assert_tool_input_safe` is wired into the canonical tool invocation middleware, or removed with evidence. No dead security claims remain.
4. **Durable audit trail (R020).** `SecurityAuditTrail` persists through the canonical evidence and persistence plane (append-only, tenant-scoped), wired in `security_audit_trail_wiring.py`. No security-owned store. Test: the trail survives a restart.
5. **Secrets (R006, R007).** Credentials are resolved only at the use site. A static raw-secret gate plus a caplog and trace leak test prove that raw secrets do not reach logs, traces, subprocess argv or evidence.
6. **Plugin provenance (R016, OD-PDC-SCOPE-07 = A).** Plugin entry-point admission in `core/plugins/admission.py` enforces provenance by reusing the agent package trust mechanism (`AgentPackageTrustPolicy`, Ed25519 verifier). Unsigned plugins are rejected in strict profiles. No second trust or signature authority.
7. **Adversarial coverage (R004, R009, R015).** Tests for child delegation widening denied, a revoked credential mid-retry denied (the cases the WBS names under `xd_06`), and tenant data not sent to an unapproved provider. Fixes are made only where a test proves a gap, inside the write set.
8. **Docs (R021, R023).** A consolidated security architecture satellite under `docs/project/maintainers/architecture/satellites/` covering the D1-D8 surfaces, plus the encryption and residency responsibility matrix. It references GOVERNED_EXECUTION, INTEGRATIONS credentials, sandbox and the EE-B3 threat model, and repairs the broken `IDENTITY_TRUST.md` link. It is a satellite, not a new policy authority.

Design constraints: No new policy engine, identity authority, trust authority or audit store (FRZ-OWN). Dev fallbacks are not kept behind an environment check. MCP exposure security (PDC-D6-R017) belongs to MCP-SERVER-X P8, not here. Dependencies B-D2 and B-D5 are implemented as unmerged draft PRs (#489, #488); this batch is cut from `development` and does not import code from them. Wiring `xd_06` and `xd_10` cases are written as `test_xd06_*` and `test_xd10_*` modules under `tests/qualification/pdc_x/d6/` and re-pointed by PDC-P6-01. D2 added a provider-egress sensitivity policy in `intergrax/context/policy/egress.py`; this batch must not duplicate it, and any overlap with R015 is reconciled at integration (PDC-P6). Shared hotspot: this is the only wave 3a batch writing under `intergrax/applications/_shared/`, and it writes only the four files named in the write set.

## Write set

`intergrax/runtime/security/`, `intergrax/core/security/`, `intergrax/core/catalog_bootstrap.py`, `intergrax/core/plugins/admission.py`, `intergrax/runtime/execution_evidence/attestor.py`, `intergrax/tools/providers/rag/service.py`, `intergrax/applications/_shared/critical_action_signing_wiring.py`, `intergrax/applications/_shared/security_assembly_resolver.py`, `intergrax/applications/_shared/security_audit_trail_wiring.py`, `intergrax/applications/_shared/application_security_wiring.py`, `docs/project/maintainers/architecture/satellites/`, `tests/qualification/pdc_x/d6/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator, who is the single integration owner for the shared hotspots (`intergrax/applications/_shared/`, `intergrax/contracts/`).

## Protocol

1. Reuse the canonical mechanism named in ADR-PDC-06.
2. Write the tests first, under `tests/qualification/pdc_x/d6/`.
3. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files.
4. Commit with `Ref: governance/tickets/T-IGX-036-P.md`, push to a branch, open a PR against `development`, and **do not merge**.
5. Record the implementation SHA; the publication SHA is recorded after the merge.
6. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- Every requirement row of the batch has a passing test (or, for R021 and R023, the published satellite) at the implementation SHA.
- Layer boundaries hold (`intergrax/` does not import `agents/` or `applications/`).
- No new registry, scheduler, engine, trust or audit authority beside the existing owner.
- Existing suites for the touched packages stay green (no weakened gate).
- Test log under `.tmp/proof/pdc-x-p5/B-D6/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d6 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

The implementer adds one more bounded `uv run pytest` over the existing unit suites of the touched packages and records it in the checkpoint.

## Out of scope

Other batches (B-D8 and B-D4 follow in sequence); MCP exposure security (B-MCP); the `ollama` regression (B-REG-P); FRZ changes; OD signatures; merges; IIAC activation.
