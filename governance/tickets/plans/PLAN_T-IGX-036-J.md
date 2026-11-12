# PLAN T-IGX-036-J — PDC-P5 batch B-D1: D1 Agent Lifecycle

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-D1**: agent upgrade/retire, skill lifecycle, discovery owner, adversarial lifecycle). Decision record: ADR-PDC-01 (Accepted). Requirements: PDC-D1-R003, R004, R007, R011, R012, R013, R014, R017, R018, R020.

This batch changes product code. It is implemented in an isolated worktree and **does not merge before MCP-SERVER-X P8 is published** (OD-PDC-SEQ-01). A merge invalidates the IIAC activation (ACT-N11), so PDC-P7-00 re-activates it.

## Deliverables

1. **Upgrade (R011).** Add `upgrade_agent(...)` to `AgentPlatformAdminService` and the agent manager command facade: install the candidate into the existing slot, `mark_verified` (trust and attestation), run a compatibility check of the new `AgentContract` version against the bound configuration (COMPAT-X version policy), then `promote_verified_to_active`. A failed check leaves the previous revision `INSTALLED_ACTIVE` and records a typed rejection.
2. **Retire (R014).** Add `retire_agent(...)`: move to `RETIRED`, remove routing eligibility, refuse new activations, keep installation and evidence records. Executions already running finish under their pinned revision.
3. **Transition evidence (R017).** Every lifecycle transition emits one canonical event through the existing evidence path.
4. **Skills (R018, OD-PDC-SCOPE-01 = A).** Add `disable_skill`, `remove_skill` and `upgrade_skill` to `SkillHostLifecycleService`, extending the Skills manifest model.
5. **Discovery owner (R020).** One discovery owner, enforced by a gate; no parallel loader.
6. **Adversarial coverage (R003, R004, R007, R012, R013).** Tests for tampered or unsigned packages, a no-shadow-registry gate, cross-tenant install/bind/activate denied, rollback after a failed activation, and revoke during execution failing closed.

Design constraints: No new registry, manager or lifecycle authority (FRZ-OWN-04). Retire is not implemented in the runtime step loop (ADR-AGENT-003). Nexus is not the agent and HarnessKernel does not plan (ADR-AGENT-001..002).

## Write set

`intergrax/agent_distribution/`, `intergrax/skills/`, `intergrax/runtime/registry/`, `intergrax/core/plugins/discovery.py`, `intergrax/rag/bootstrap/entry_point_load.py`, `intergrax/runtime/vendor_knowledge/contribution_catalog.py`, `intergrax/contracts/agent_lifecycle_state.py`, `tests/qualification/pdc_x/d1/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator, who is the single integration owner for the shared hotspots (`intergrax/applications/_shared/`, `intergrax/contracts/`).

## Protocol

1. Reuse the canonical mechanism named in ADR-PDC-01.
2. Write the tests first, under `tests/qualification/pdc_x/d1/`.
3. One `uv run pytest` per verification step; ruff and pyright on touched files.
4. Commit with `Ref: governance/tickets/T-IGX-036-J.md`, push to a branch, open a PR against `development`, and **do not merge**.
5. Record the implementation SHA; the publication SHA is recorded after the merge.
6. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- Every requirement row of the batch has a passing test at the implementation SHA.
- Layer boundaries hold (`intergrax/` does not import `agents/` or `applications/`).
- No new registry, scheduler or authority beside the existing owner.
- Existing suites for the touched packages stay green (no weakened gate).
- Test log under `.tmp/proof/pdc-x-p5/B-D1/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d1 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

The implementer adds one more bounded `uv run pytest` over the existing unit suites of the touched packages and records it in the checkpoint.

## Out of scope

Other batches; the `ollama` regression (B-REG-P); FRZ changes; OD signatures; merges; IIAC activation.
