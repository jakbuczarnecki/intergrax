# PLAN T-IGX-036-R — PDC-P5 batch B-D4: D4 Host Runtime

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-D4**: generic lifecycle suite with crash and restart proof, lifecycle metrics, digest-preserving restart, stable public API, TL-FIX-C). Decision record: ADR-PDC-04 (Accepted) and OD-PDC-SCOPE-02 = A. Requirements: PDC-D4-R003, R004, R005, R006, R007, R009, R010, R014, R015.

This batch changes product code. It is implemented in an isolated worktree and **does not merge before MCP-SERVER-X P8 is published** (OD-PDC-SEQ-01). A merge invalidates the IIAC activation (ACT-N11), so PDC-P7-00 re-activates it.

## Deliverables

1. **Generic APP-HOST lifecycle suite (R003, R004, R005, R006).** A qualification suite over the real `HostedApplicationEngine`: startup failure rolls back in reverse order, a degraded component is reported (readiness and degraded states), shutdown drains in-flight background work, and a second instance is rejected by the instance guard. Fixes are made only where a test proves a gap, inside the write set.
2. **Crash and restart proof (R007, APP-HOST-5E).** An OS-subprocess test that kills a hosted application and proves a supervised restart through the existing `HostedApplicationSupervisor`. No new supervisor. Process-based tests use bounded timeouts and are cleaned up so they cannot leave stray processes.
3. **Lifecycle metrics (R009, APP-HOST-3D).** Start, ready, degraded, drain duration, restart count and exit class are emitted through the canonical observability and event port. Hosting owns no metrics backend.
4. **Digest-preserving restart (R010, APP-HOST-5D).** Restart preserves and verifies the configuration and profile digest for every hosted application, not only the LKW path. A digest mismatch on restart is rejected.
5. **Stable public API (R014, APP-HOST-9F).** The public hosting API is declared as a manifest covered by the COMPAT-X version policy, with a gate that fails on an undeclared public symbol or change.
6. **No product vocabulary in the generic profile (R015, TL-FIX-C).** LKW-specific fields are removed from the generic `HostDeploymentProfile` and kept on the product side (`lkw_hybrid_daemon_wiring.py`), with a test that fails if product vocabulary reappears in the generic profile. Existing callers keep working through the COMPAT-X shim policy.

Design constraints: No hosting-owned metrics store and no second composition root (FRZ-OWN). `ApplicationEnvironmentProfile` authority is preserved. APP-HOST-6 (InteractionProfile), APP-HOST-7 (OS service adapters) and APP-HOST-9E (host plugins) are authorized exclusions (OD-PDC-SCOPE-02 = A) and are not implemented. Dependencies B-D7 and B-D8 are unmerged draft PRs (#475 and the B-D8 PR); this batch is cut from `development` and does not import code from them. The `xd_05` case named in the WBS (drain with background work) is written as a `test_xd05_*` module under `tests/qualification/pdc_x/d4/` against the existing background execution, and re-pointed by PDC-P6-01 on the integrated baseline. This batch writes under the shared hotspot `intergrax/applications/_shared/` only in the two files named in the write set, and it is the only wave 3c batch. A public contract break stops the batch and goes to the orchestrator.

## Write set

`intergrax/hosting/`, `intergrax/applications/_shared/environment_wiring.py`, `intergrax/applications/contracts/environment_profile/sub_profiles.py`, `intergrax/applications/_shared/lkw_hybrid_daemon_wiring.py`, `tests/qualification/pdc_x/d4/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator, who is the single integration owner for the shared hotspots (`intergrax/applications/_shared/`, `intergrax/contracts/`).

## Protocol

1. Reuse the canonical mechanism named in ADR-PDC-04.
2. Write the tests first, under `tests/qualification/pdc_x/d4/`.
3. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files. Process-based tests carry hard timeouts.
4. Commit with `Ref: governance/tickets/T-IGX-036-R.md`, push to a branch, open a PR against `development`, and **do not merge**.
5. Record the implementation SHA; the publication SHA is recorded after the merge.
6. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- Every requirement row of the batch (9 rows) has a passing test at the implementation SHA.
- Layer boundaries hold (`intergrax/` does not import `agents/` or `applications/`).
- No new registry, supervisor, metrics backend, composition root or authority beside the existing owner.
- Existing suites for the touched packages stay green (no weakened gate); COMPAT-X gates stay green.
- Test log under `.tmp/proof/pdc-x-p5/B-D4/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d4 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

The implementer adds one more bounded `uv run pytest` over the existing unit suites of the touched packages and records it in the checkpoint.

## Out of scope

Other batches; APP-HOST-6, -7 and -9E (OD-PDC-SCOPE-02 = A); the `ollama` regression (B-REG-P); FRZ changes; OD signatures; merges; IIAC activation.
