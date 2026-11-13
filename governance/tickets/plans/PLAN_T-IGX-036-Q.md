# PLAN T-IGX-036-Q — PDC-P5 batch B-D8: D8 Artifact Lifecycle

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-D8**: artifact taxonomy, mandatory digest, governed read and download port, durable store, retention, spill, dedupe). Decision record: ADR-PDC-08 (Accepted) and OD-PDC-SCOPE-04 = A. Requirements: PDC-D6-R022, PDC-D8-R001, R002, R003, R004, R005, R006, R007, R008, R009, R010, R012, R013, R014, R016, R017, R018.

This batch changes product code. It is implemented in an isolated worktree and **does not merge before MCP-SERVER-X P8 is published** (OD-PDC-SEQ-01). A merge invalidates the IIAC activation (ACT-N11), so PDC-P7-00 re-activates it.

## Deliverables

1. **Taxonomy and owners (R001).** Four classes with four owners and no global storage engine: business artifacts stay with Collaborative Work, execution and audit evidence with the evidence plane, context artifacts with UCL, and temporary tool output with the tool runtime (spill-to-reference). A gate asserts the taxonomy and the one-owner-per-class rule.
2. **Shared minimal reference contract (R002, R004).** In `intergrax/contracts/artifact_ref.py` the content digest becomes mandatory on every durable artifact reference, next to producer identity, execution lineage, tenant and workspace scope and a retention class. A digest mismatch is rejected. Artifact lineage is reconstructable from the references. Additive migration: existing callers keep working through a defined default or an explicit typed error, not a silent empty digest.
3. **Versioning (R003).** Run artifact versions are append-only.
4. **Governed read port (R005, R006, R016).** One governed read port in front of every store: authorization, tenant check and an audit event per operation. Stores stay storage-only and contain no security logic. A cross-tenant read and an unauthorized read are denied, and artifact operations appear in evidence. The port reuses the existing governance and authorization owners; it adds no policy engine.
5. **Durable execution artifact store (R007, R014).** The in-memory `ArtifactStore` gains a durable implementation through the canonical persistence plane. Artifacts survive a restart and recovery after a restart is proved on a real SQLite store.
6. **Retention and legal hold (R008, R009, PDC-D6-R022).** Retention class and TTL are enforced by a sweep admitted as background work through canonical execution (no new scheduler). Audit evidence and artifacts under legal hold are never deletable by TTL or compaction; deletion of evidence is refused.
7. **Governed download (R010, OD-PDC-SCOPE-04 = A).** Governed secure download is the governed read port plus a short-lived provider URL; no presigned URL is issued outside the governed port.
8. **Spill and context references (R012, R013).** Large tool output spills to an artifact reference. The round trip returns the original content, and a context reference to an artifact still resolves after compaction.
9. **Dedupe and aliases (R017, R018).** `applications/contracts/application_artifacts.py` becomes a COMPAT-X shim over `contracts/task_artifacts.py` (COMPAT-X R5 policy, no public contract break), guarded by a single-definition gate. The trace alias is renamed so only one `ArtifactRef` remains and no ambiguous alias exists.
10. **Docs.** The feature doc pair `docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md` and `docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md` (a feature doc coordinating the domain owners, not a new domain). Registration in the runtime hub and in `docs/project/capabilities/README.md` is outside the write set and belongs to the orchestrator; if the docs-pair checker fails without it, the batch reports that and does not edit the hubs.
11. **Closing the gaps found by the first implementation (Amendment 1, Operator decision 2026-11-13).** (a) The closed-world gate in `tests/qualification/state_x/_state_x_explicit_mechanism_classifications.py` gets a classification entry for `intergrax/runtime/nexus/artifacts/sqlite_artifact_store.py`, modeled on the existing in-memory artifact store entry; the gate is not weakened and no other entry changes. (b) R010: `intergrax/tools/providers/storage/service.py` (`storage_presigned_url`) no longer signs an arbitrary key for an arbitrary caller; presigned URLs are issued only through the governed read port (tenant check, authorization and audit event), and the strict-xfail gate from the first implementation flips to a passing test. (c) R018: `intergrax/runtime/replay/models.py` and `intergrax/runtime/replay/replay_engine.py` stop defining a separate `ArtifactRef`; they use the single reference contract or a clearly renamed replay type, with the old import path kept working through the COMPAT-X shim policy, and the strict-xfail gate flips to a passing test. Hub registration of the feature docs in `docs/project/capabilities/README.md` and the runtime hub, and the composition-root wiring (tool runtime calling `ToolOutputSpiller`, the read port and retention sweep registration, the SQLite opener), remain outside this write set; the orchestrator handles the docs registration and PDC-P6 handles the wiring.

Design constraints: No global artifact storage engine and no security logic inside stores (FRZ-OWN). Dependency B-D6 is an unmerged draft PR (#494); this batch is cut from `development` and does not import code from it, and the governed read port reuses the existing authorization owners. The `xd_09` and `xd_10` cases named in the WBS are written as `test_xd09_*` and `test_xd10_*` modules under `tests/qualification/pdc_x/d8/` and re-pointed by PDC-P6-01. This batch writes under the shared hotspots `intergrax/contracts/` and `intergrax/applications/_shared/` only in the files named in the write set, and it is the only wave 3b batch. B-D4 follows it. Public contract changes follow the COMPAT-X shim policy; a break of a public symbol stops the batch and goes to the orchestrator.

## Write set

`intergrax/contracts/artifact_ref.py`, `intergrax/contracts/task_artifacts.py`, `intergrax/applications/contracts/application_artifacts.py`, `intergrax/runtime/nexus/artifacts/`, `intergrax/runtime/nexus/tracing/trace_models.py`, `intergrax/collaborative_work/`, `intergrax/runtime/execution/run_artifact_bundle_composition.py`, `intergrax/runtime/nexus/orchestration/run_artifact_bundle_builder.py`, `intergrax/applications/_shared/run_artifact_bundle_builder.py`, `docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md`, `docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md`, `tests/qualification/state_x/_state_x_explicit_mechanism_classifications.py`, `intergrax/tools/providers/storage/service.py`, `intergrax/runtime/replay/models.py`, `intergrax/runtime/replay/replay_engine.py`, `tests/qualification/pdc_x/d8/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator, who is the single integration owner for the shared hotspots (`intergrax/applications/_shared/`, `intergrax/contracts/`).

## Protocol

1. Reuse the canonical mechanism named in ADR-PDC-08.
2. Write the tests first, under `tests/qualification/pdc_x/d8/`.
3. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files.
4. Commit with `Ref: governance/tickets/T-IGX-036-Q.md`, push to a branch, open a PR against `development`, and **do not merge**.
5. Record the implementation SHA; the publication SHA is recorded after the merge.
6. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- Every requirement row of the batch (17 rows) has a passing test, or for the feature doc the published pair, at the implementation SHA.
- Layer boundaries hold (`intergrax/` does not import `agents/` or `applications/`).
- No global storage engine, no new registry, scheduler or authority beside the existing owner.
- Existing suites for the touched packages stay green (no weakened gate); COMPAT-X gates stay green for the shim.
- Test log under `.tmp/proof/pdc-x-p5/B-D8/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d8 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

The implementer adds one more bounded `uv run pytest` over the existing unit suites of the touched packages and records it in the checkpoint.

## Out of scope

Other batches (B-D4 follows); the `ollama` regression (B-REG-P); FRZ changes; OD signatures; merges; IIAC activation; hub registration of the feature docs.
