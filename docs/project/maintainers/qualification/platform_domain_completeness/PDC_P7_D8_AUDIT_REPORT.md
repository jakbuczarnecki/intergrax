# PDC-P7 IA-D8 - Independent audit of D8 Artifact Lifecycle

| Field | Value |
| --- | --- |
| Stage | PDC-X / D8 (PDC-P7, independent auditor IA-D8) |
| Role | INDEPENDENT AUDITOR. Not the implementer (B-D8 / T-IGX-036-Q, B-P6-R repairs), not the certifier. No product, test, script, governance or ticket file was edited; docs and evidence only. |
| Auditor session | `PDC-P7-IA-D8-INDEPENDENT-AUDITOR-pdc-x-ia-d8-04b4f7` (branch `claude/pdc-x-ia-d8-04b4f7`) |
| Audited SHA (`D8_AUDITED_SHA`) | `c08efe46a74a20595468d8d8bf06eab2dca5f573` |
| Audit execution SHA | `3b371ef9ddee209eb707f184337708303a956023` (descends from the audited SHA; the delta since it is docs plus `governance/tickets/T-IGX-038-A..F.md` only, no path under `intergrax/ agents/ applications/ scripts/ tests/`) |
| Policy | `IIAC-2026-11-05.1`, OD-PDC-IIAC-SCOPE (Option A): verdicts PASS / PASS_SCOPED / PARTIAL / FAIL |
| Rows | 19 `PDC-D8-R*` rows; `PDC-D6-R022` (B-D8 scope, D6-owned) cross-referenced only |
| **Overall verdict** | **PARTIAL** (IIAC manifest value `CONDITIONAL`, see limitation LIM-verdict-mapping) |
| Findings | 1 HIGH, 6 MEDIUM, 9 LOW, 4 INFO (20) |

## 1. Summary

The mechanisms B-D8 set out to build are real and well tested at the library level: the taxonomy and class gate, `DurableArtifactRef` with a mandatory digest, append-only versions, the SQLite store with restart recovery and tamper detection, the single governed port (tenant check, existing governance owner, one audit event per outcome, short-lived download), the retention sweep with evidence and legal-hold protection, spill with a self-verifying reference, and the contract de-duplication. All 134 B-D8, XD-09 and XD-10 tests pass at the audited tree, and the adversarial probes of the tenant, digest and evidence-delete paths held.

The audit does not close the domain, for one reason that repeats across rows: the mechanisms are not reachable in any shipped composition. No product code gives the runtime an `artifact_store`, nothing admits the retention sweep, nothing writes `AUDIT_EVIDENCE`, the one runtime gateway for execution artifacts raises `TypeError`, and the reference chain written by spill has no reader. In addition, raw `storage.*` tools still bypass the governed port, the port's workspace scope is optional, and the sweep can delete a newer retained version. These are repairable inside the D8 owner paths plus one composition decision (F-D8-01, F-D8-02).

Known, already classified reds were not re-reported (EBH-7 identity, five `roadmap_replay_x` tests, `dist_scale_x` and Docker under the C10/C15 freeze). The `pdc_x` pytest exit hang (R-05) did not occur in the three D8-scope runs.

## 2. Method

1. Read OD-PDC-IIAC-SCOPE, the 19 D8 rows of `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`, the D8 rows of the ownership matrix, ADR-PDC-08, the B-D8 plan in the WBS, ticket `T-IGX-036-Q`, and the capability pair `ARTIFACT_LIFECYCLE`. The matrix rows still show the pre-P5 state (`NOT_CERTIFIED`, `OPEN`); they were used as the claim, not as evidence.
2. Read the product code of every owner path: `contracts/artifact_ref.py`, `contracts/task_artifacts.py`, `applications/contracts/application_artifacts.py`, `runtime/nexus/artifacts/*` (all modules), `collaborative_work/content_storage.py`, `runtime/execution/run_artifact_bundle_composition.py` and the bundle builder, `tools/providers/storage/service.py`, `runtime/execution/agent_runtime_context.py` (gateway and worker composition), `runtime/task/worker_bootstrap.py`, `runtime/nexus/tools/plan_context_invocation.py`, and the persistence composition.
3. Ran the cited tests, one pytest process at a time (`logs/`): D8 directory plus XD-09 plus XD-10 (134 passed, 132.9 s), the Collaborative Work content storage files for R015 (52 passed), the adjacent partial-results test file (1 failed, F-D8-19), and `scripts/qualification/pdc_x_requirements.py --check` (OK).
4. Ran adversarial probes in four scripts (`iiac_evidence/PDC-X/D8/adversarial_scripts/`, logs under `logs/`): bypass via the raw storage tools, read after expiry, workspace omission, sweep starvation, tampered row, naive datetimes, negative limit, permissive governance, hostile source, late-version evidence delete, legal-hold overwrite, retention-class typos, digest newline, cross-tenant lineage, the execution gateway. Reproduction commands are in each finding.
5. Re-verified the findings of the P7-00 dossier and addenda routed to IA-D8 (section 5), and traced production callers with `git grep` over `intergrax/ agents/ applications/` to separate "unit-true" from "reachable in a shipped composition".

Verdict basis per row: PASS = the claim holds and, where the row is a runtime behaviour, it is reachable in the composition the platform ships; PASS_SCOPED = holds except for an Operator-authorized exclusion; PARTIAL = the mechanism holds but a reachability, bypass or enforcement defect keeps the row open; FAIL = the mechanism does not hold.

## 3. Per-row verdicts

| Row | Claim (short) | Verdict | Evidence and findings |
| --- | --- | --- | --- |
| PDC-D8-R001 | Taxonomy, one owner per class, no global engine | **PASS** | `taxonomy.py` plus `assert_taxonomy_consistent`; stores refuse `business` and `context` (`require_owned_class`); `test_d8_taxonomy_gate.py` green. INFO F-D8-17 (d). |
| PDC-D8-R002 | Mandatory digest on durable refs | **PASS** | `DurableArtifactRef.content_digest` has no default and is validated; `Artifact` derives or verifies it; mismatch tests green. LOW F-D8-11 (trailing newline). v1 `ArtifactRef.checksum` stays optional as the row allows. |
| PDC-D8-R003 | Versioning; run artifacts append-only | **PASS** | Both stores append (`put` versions, `get_version`, `list_versions`); SQLite primary key plus `BEGIN IMMEDIATE`. Delete removes all versions by design (see F-D8-05). |
| PDC-D8-R004 | Producer identity and execution lineage | **PARTIAL** | Contract and `reconstruct_artifact_lineage` hold (tests, probe). The runtime gateway cannot run (F-D8-07); spill lineage is thin and `durable_refs`/`to_durable_ref` have no product caller (F-D8-02); ancestors are not tenant-checked (F-D8-13); bundle builder (F-D8-15). |
| PDC-D8-R005 | Tenant/workspace scope | **PARTIAL** | Tenant isolation holds (request versus caller check before any store access, stored-ref re-check, tenant-keyed SQL; XD-10 and probes). Workspace scope is optional (F-D8-04) and the raw storage tools are unscoped (F-D8-03). |
| PDC-D8-R006 | Governed read/authorization port | **PARTIAL** | Port is real: capability grant through the governance boundary, denial and approval paths audited. Open: raw storage tools bypass it (F-D8-03), decision value ignored (F-D8-08), no shipped host builds it (F-D8-02). |
| PDC-D8-R007 | Durable execution artifact store | **PARTIAL** | `SQLiteArtifactStore` is durable, WAL, tamper-checked and part of the persistence bundle. Nothing in product writes an execution-evidence artifact (F-D8-07) or hands the store to the runtime (F-D8-02). |
| PDC-D8-R008 | Retention/TTL enforcement | **PARTIAL** | Sweep, schedule, protection and `resolve_expiry` work in tests. Not enforced in any shipped composition and no trigger (F-D8-01 HIGH); read ignores expiry (F-D8-06); unaudited and version-unsafe delete (F-D8-05); robustness (F-D8-09, F-D8-10). |
| PDC-D8-R009 | Deletion, legal hold, evidence non-deletable | **PARTIAL** | Store invariants verified, including an evidence version followed by an expiring version (probe P11 refused); Collaborative Work has no delete API. But nothing produces `AUDIT_EVIDENCE` (F-D8-01), `storage.delete` can remove Collaborative Work content (F-D8-03), holds are overwritable (F-D8-14). |
| PDC-D8-R010 | Governed secure download | **PARTIAL** | Port plus TTL-clamped provider URL after authorization, tenant check and audit; `storage.presigned_url` fails closed (code and gate). The Collaborative Work source that carries the object key has no product caller and the port is not composed (F-D8-02). |
| PDC-D8-R011 | Partial outputs owned by `PartialResultContract` | **PASS** | `contracts/partial_result_contract.py` and `applications/_shared/partial_results_wiring.py` exist; none of the three partial-result modules mentions artifacts. The adjacent red test is not this claim (F-D8-19). |
| PDC-D8-R012 | Spill of large tool output to artifact refs | **PARTIAL** | Spill, credential-literal scrub, ephemeral TTL, self-verifying token and round trip verified (XD-09 green). Inert in shipped hosts and no resolver in product (F-D8-02). |
| PDC-D8-R013 | Durable context references to artifacts | **PARTIAL** | Reference fragment is mandatory and survives the real `apply_fragment_compaction`; cross-tenant token denied. `reference_fragment` and `ArtifactReferenceResolver` have no product caller (F-D8-02). |
| PDC-D8-R014 | Recovery after restart | **PASS** | A new `SQLiteArtifactStore` over the same file returns versions, holds, expiry, producer and lineage; a tampered row is rejected on load (probe P8). LOW F-D8-09 (one bad row aborts a sweep). |
| PDC-D8-R015 | Cross-provider portability of content storage | **PASS_SCOPED** | Replay: 52 passed (content storage plus two architecture gates); the delta since the EBH-5 pin is cosmetic plus one read-only helper. Scope: provider evidence is a fake object storage; real vendor integration is a declared exception (C10/C15 freeze). The full Collaborative Work repository suites were not replayed (limitation). INFO F-D8-17 (c). |
| PDC-D8-R016 | Auditability of artifact operations | **PARTIAL** | One `GovernanceAuditEvent` per outcome for the listed outcomes (tests; events carry actor, tenant, execution identity). Gaps: backend failures unaudited (F-D8-12), sweep may run unaudited (F-D8-05), no durable audit sink composed (F-D8-01, F-D8-02). The row's named owner (`runtime_inspection/adapters/artifact_read.py`) is unchanged by B-D8. |
| PDC-D8-R017 | Single definition of task artifact contracts | **PASS** | `application_artifacts.py` re-exports `task_artifacts` and defines no class; single-definition gate green. INFO F-D8-17 (a): the ownership matrix says "removed". |
| PDC-D8-R018 | One `ArtifactRef` | **PASS** | Trace and replay aliases are unbound and resolve through module `__getattr__` shims; `ReplayArtifactRef` and `TraceArtifactRef` are the only names; alias gate green. INFO F-D8-17 (e) residual annotation. |
| PDC-D8-R019 | Compaction never deletes evidence artifacts | **PASS** | Compaction operates on context fragments only; XD-09 runs the real `apply_fragment_compaction` and shows the artifact and its reference intact; the store refuses evidence deletion. Depends on F-D8-01 for evidence ever being classified. |
| PDC-D6-R022 | TTL deletion honored (D6-owned, B-D8 batch) | n/a (IA-D6) | Mechanism test green; the enforcement gap is F-D8-01. |

Tally: PASS 8 (R001, R002, R003, R011, R014, R017, R018, R019), PASS_SCOPED 1 (R015), PARTIAL 10 (R004-R010, R012, R013, R016), FAIL 0.

## 4. Findings

Severity: HIGH = a requirement's enforcement is absent in the shipped product; MEDIUM = a bypass, data-loss or evidence gap with a plausible trigger; LOW = hardening or edge case; INFO = record-keeping. Repairs belong to a later ticket of the owning domain, done by a different session; the auditor fixed nothing.

### F-D8-01 - HIGH - Retention and TTL are not enforced by any shipped composition; evidence protection has no producer

- **Rows:** PDC-D8-R008, PDC-D8-R009, PDC-D6-R022 - **Verification:** CODE_READ + GREP (reproducible); unit mechanism re-run green (134 passed) - **Dossier lead:** F-04 (D8 instance)
- **Evidence:** The sweep exists only as a handler: `register_retention_sweep` (retention.py:219) is reached only through `compose_worker_artifact_services` (agent_runtime_context.py:478), which `build_nexus_task_execution_registry` calls only when its optional `artifact_store` argument is given. The only shipped caller, `applications/_shared/queue_worker_wiring.py:86` (`create_nexus_celery_worker_app`), passes none. `ARTIFACT_RETENTION_SWEEP_TASK_NAME` is referenced only inside retention.py: nothing in product code builds or submits a sweep task (compare `memory_retention_sweep_task_request` for the memory plane). `ArtifactRetentionPolicy.retain_hours` has one consumer, the optional `policy` argument of `resolve_expiry`, which no product caller supplies; `delete_on_task_complete` and `archive_to_object_store` have none (grep over intergrax/ agents/ applications/). `ArtifactRetentionClass.AUDIT_EVIDENCE` is assigned by no product code (only enum, `resolve_expiry` and `protection_reason` mention it), so the 'audit evidence is never deletable' invariant protects nothing that the platform produces. The B-D8 unit tests exercise the sweep with a harness-built store, registry and clock.
- **Reproduction:** `git grep -n 'ARTIFACT_RETENTION_SWEEP_TASK_NAME\|AUDIT_EVIDENCE\|retain_hours\|delete_on_task_complete' -- intergrax agents applications ':!*tests*'; git grep -n 'artifact_store' -- applications intergrax/applications (no assignment of RuntimeContext.artifact_store or worker artifact_store).`
- **Impact:** TTL expiry (FRZ-SEC-09) and the evidence/legal-hold deletion guarantees hold for the library and for hosts that wire it by hand; a default host never deletes an expired artifact, never admits the sweep, and never classifies anything as audit evidence.
- **Suggested repair scope:** Orchestrator decision whether host composition is a PDC-X claim (as for dossier F-04). If yes: (a) composition root passes the persistence bundle's `artifact_store`, a governance port and a durable audit recorder into the worker bootstrap and into `RuntimeContext`; (b) add a sweep task-request builder and an admission/schedule trigger next to the sweep; (c) give `ArtifactRetentionPolicy` a real consumer or retire the dead fields; (d) give evidence artifacts a producer that sets AUDIT_EVIDENCE.
- **Owner paths:** `intergrax/runtime/task/worker_bootstrap.py`, `intergrax/applications/_shared/queue_worker_wiring.py`, `intergrax/runtime/nexus/artifacts/retention.py`, `intergrax/runtime/execution/agent_runtime_context.py`, `intergrax/contracts/task_artifacts.py`

### F-D8-02 - MEDIUM - The D8 runtime (durable store, governed port, spill, reference chain) is dormant in shipped hosts, and the spill round trip has no resolver in product code

- **Rows:** PDC-D8-R006, PDC-D8-R007, PDC-D8-R010, PDC-D8-R012, PDC-D8-R013, PDC-D8-R004 - **Verification:** CODE_READ + GREP (reproducible) - **Dossier lead:** F-04 (D8 instance)
- **Evidence:** No product code assigns `RuntimeContext.artifact_store` (field default None, agent_runtime_context.py:126), so `_spill_large_tool_output` (plan_context_invocation.py:65) is inert. `GovernedArtifactReadPort` is constructed only in `compose_worker_artifact_services`; `ArtifactReferenceResolver`, `reference_fragment`, `REFERENCE_TOOL_NAME` (no tool of that name is registered) and `CollaborativeWorkContentSource` have no non-test caller; no read source is registered for the `context` or `business` class in any composition. If a host does set a store, a tool output over 16 KiB is replaced in the model-visible message by a 512-char preview plus `artifact://...` marker (plan_context_invocation.py:77) that nothing in the runtime can turn back into content. The spill also records only run_id (no task_id, trace_id, workspace_id; `agent_id` may be empty string), so the lineage promised for R004 is thin on the one product path that writes artifacts. `RunArtifactBundle.durable_refs` and `ArtifactRef.to_durable_ref` have no product caller.
- **Reproduction:** `git grep -n 'ArtifactReferenceResolver\|reference_fragment(\|CollaborativeWorkContentSource\|GovernedArtifactReadPort(\|durable_refs(\|\.artifact_store' -- intergrax agents applications ':!*tests*'.`
- **Impact:** Rows R006/R007/R010/R012/R013 close at library level only; enabling spill alone would lose tool output for the agent.
- **Suggested repair scope:** Same composition decision as F-D8-01; additionally a registered `artifact.reference` read tool (through the governed port) and a reference-fragment injection point must ship together with spill, or spill must stay off.
- **Owner paths:** `intergrax/runtime/nexus/tools/plan_context_invocation.py`, `intergrax/runtime/nexus/artifacts/context_reference.py`, `intergrax/runtime/nexus/artifacts/spill.py`, `intergrax/runtime/execution/agent_runtime_context.py`

### F-D8-03 - MEDIUM - Raw object-storage tools (`storage.get/put/delete/exists`) bypass the governed artifact port and the tenant check

- **Rows:** PDC-D8-R005, PDC-D8-R006, PDC-D8-R009 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** new
- **Evidence:** ADR-PDC-08 closed `storage.presigned_url` (fails closed) but its siblings in tools/providers/storage/service.py take any key and act on the same `ObjectStorage` that backs Collaborative Work content (key = `intergrax/collaborative-work/artifact-content/v1/<sha256(tenant NUL workspace)>/sha256/<digest>`, derivable from public ids plus a content digest). They carry no tenant or prefix binding, no artifact authorization and no GovernanceAuditEvent; `get`/`exists` are risk LOW, `put`/`delete` HIGH. Probe P4 (ia_d8_probes_1.py): content stored for tenant 'tenant-victim' is returned by `storage_get`, reported by `storage_exists` and removed by `storage_delete` with no check.
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probes_1.py  (probe P4_storage_tools_bypass; log logs/probes_1.log)`
- **Impact:** An agent holding the storage capability can read, overwrite or hard-delete business artifact content outside the governed port; delete also defeats the 'Collaborative Work has no hard-delete API' claim (R009). Exploitation needs the content digest, so the cross-tenant read risk is bounded; the missing governance is not.
- **Suggested repair scope:** Bind storage tool keys to the caller tenant (prefix) and refuse the Collaborative Work content prefix, or route the tools through the governed port; add a gate like the presign gate that fails if an ungoverned tool can address the content prefix.
- **Owner paths:** `intergrax/tools/providers/storage/service.py`, `intergrax/tools/providers/storage/bundle.py`, `intergrax/collaborative_work/content_storage.py`

### F-D8-04 - MEDIUM - Governed read: workspace scope is optional and the capability is tenant-wide

- **Rows:** PDC-D8-R005, PDC-D8-R006 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** F-01 (confirmed)
- **Evidence:** governed_access.py:283 checks the stored workspace only `if request.workspace_id is not None`; authorization is the tenant-level capability `artifact.read`, with no run or principal binding. Probe P6: an artifact stored in 'ws-1' is read with no workspace_id, denied only for a wrong one. Same for download, legal hold and delete requests. Re-verified from dossier lead F-01.
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probes_1.py  (probe P6_workspace_scope_optional)`
- **Impact:** Any holder of `artifact.read` in a tenant reads any execution-evidence or tool-output artifact of that tenant by id (markers carry the id).
- **Suggested repair scope:** Derive the workspace (or a run/principal binding) from `ArtifactAccessContext` or make it mandatory in `ArtifactReadRequest` for classes that store it; add a negative test.
- **Owner paths:** `intergrax/runtime/nexus/artifacts/governed_access.py`, `intergrax/runtime/nexus/artifacts/sources.py`

### F-D8-05 - MEDIUM - Retention sweep can delete without any audit record and deletes versions written after the expiry selection

- **Rows:** PDC-D8-R008, PDC-D8-R009, PDC-D8-R016 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** F-02 (confirmed, both parts)
- **Evidence:** `ArtifactRetentionSweeper(audit=None)` deletes and records nothing (`_record` returns when no auditor; worker_bootstrap passes `artifact_audit=None` by default). `sweep()` selects expired latest versions, then `delete_artifact` removes EVERY version: probe F02 (in-memory and SQLite) puts a non-expiring v2 after selection and the delete leaves 0 versions. Re-verified from dossier lead F-02.
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probes_2.py  (probes F02_toctou..., F02_sweep_without_audit...)`
- **Impact:** Destructive background deletion with optional evidence; a newer retained version can be destroyed by a sweep that selected an older expired one.
- **Suggested repair scope:** Refuse to construct the sweeper without an audit recorder on product profiles; delete conditionally on the version/expiry observed (compare-and-delete inside the store transaction).
- **Owner paths:** `intergrax/runtime/nexus/artifacts/retention.py`, `intergrax/runtime/nexus/artifacts/sqlite_artifact_store.py`, `intergrax/runtime/nexus/artifacts/in_memory_artifact_store.py`

### F-D8-06 - MEDIUM - Expired artifacts stay readable and downloadable until a sweep runs

- **Rows:** PDC-D8-R008 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** new
- **Evidence:** `GovernedArtifactReadPort.read`/`issue_download_url` and `ExecutionArtifactSource` never consult `expires_at_utc`. Probe P5: an artifact whose TTL ended 30 days earlier is read successfully.
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probes_1.py  (probe P5_read_after_expiry_unswept)`
- **Impact:** With F-D8-01 (no scheduled sweep) the TTL is neither enforced at rest nor at read; ephemeral tool output outlives its retention class.
- **Suggested repair scope:** Treat an expired, unprotected artifact as not found in the source or the port (clock already injected) and test it.
- **Owner paths:** `intergrax/runtime/nexus/artifacts/sources.py`, `intergrax/runtime/nexus/artifacts/governed_access.py`

### F-D8-07 - MEDIUM - The runtime gateway for execution artifacts (`RuntimeContext.create_artifact`) cannot work: it omits the required tenant_id

- **Rows:** PDC-D8-R004, PDC-D8-R007 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** new
- **Evidence:** agent_runtime_context.py:235 builds `Artifact(...)` without `tenant_id` (a required field since commit 4f7517050); calling it raises `TypeError: Artifact.__init__() missing 1 required positional argument: 'tenant_id'` (probe logs/probe_create_artifact.log). It has no caller anywhere in intergrax/, agents/, applications/ or tests/. It also sets no producer, lineage, retention class or expiry. Pre-existing (e8064d72e), not introduced by B-D8, but it is the documented gateway of the D8 execution-evidence class; the only product code that writes `Artifact` objects is the spill path.
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probe_create_artifact.py`
- **Impact:** No execution-evidence artifact of class `execution_evidence` can be produced by the runtime; R004 producer/lineage and R007 durability have no real writer.
- **Suggested repair scope:** Fix the gateway (tenant from `state`, producer and lineage from the run identity, retention class) and add a test that calls it against a SQLite store; or remove it and name the real writer.
- **Owner paths:** `intergrax/runtime/execution/agent_runtime_context.py`, `intergrax/runtime/nexus/artifacts/models.py`

### F-D8-08 - LOW - GovernedArtifactReadPort._guard ignores the authorization decision value

- **Rows:** PDC-D8-R006 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** F-10 (confirmed)
- **Evidence:** `authorize_tool` result is discarded (governed_access.py `_guard`); fail-closed only because the real boundary raises. Probe F10: a governance port returning a DENY decision without raising lets the read succeed. Re-verified from dossier lead F-10.
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probes_2.py  (probe F10_permissive_governance_returning_deny)`
- **Impact:** Any other `AgentRuntimeGovernancePort` implementation that returns instead of raising grants access.
- **Suggested repair scope:** Check the returned decision in `_guard` and deny on anything but ALLOW.
- **Owner paths:** `intergrax/runtime/nexus/artifacts/governed_access.py`

### F-D8-09 - LOW - Sweep robustness: starvation by protected rows, one poisoned row aborts the sweep, naive/aware datetimes, negative limit

- **Rows:** PDC-D8-R008, PDC-D8-R014 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** F-28 (confirmed, part new)
- **Evidence:** P7: with limit=5 and five expired legal-hold artifacts that sort first, `sweep` examines only those, deletes nothing and a second sweep repeats it, so an unprotected expired artifact is never reached. P8: a tampered SQLite row makes `list_expired` raise `ArtifactDigestMismatchError`; the sweep aborts for the whole tenant. F28: a naive `expires_at_utc` makes the in-memory `list_expired` raise TypeError; a negative `limit` slices off the last candidate in memory and means unlimited in SQLite. Re-verified from dossier lead F-28.
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probes_1.py (P7, P8) and ia_d8_probes_2.py (F28_*)`
- **Impact:** Availability and accuracy of retention under hold-heavy or corrupted data; no data disclosure.
- **Suggested repair scope:** Skip protected rows in the candidate query (or page past them), catch and report per-artifact load failures, normalize datetimes at `Artifact` construction, validate `limit` >= 1.
- **Owner paths:** `intergrax/runtime/nexus/artifacts/retention.py`, `intergrax/runtime/nexus/artifacts/sqlite_artifact_store.py`, `intergrax/runtime/nexus/artifacts/in_memory_artifact_store.py`, `intergrax/runtime/nexus/artifacts/models.py`

### F-D8-10 - LOW - A v1 `ArtifactRef` retention class that is not an exact enum value silently becomes `default`

- **Rows:** PDC-D8-R008, PDC-D8-R009 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** new
- **Evidence:** `_retention_class_from_str` (artifact_ref.py:222) falls back to DEFAULT. Probe P3: 'audit_evidence' stays, but 'AUDIT_EVIDENCE', 'audit-evidence' and 'evidence' become 'default' in the durable reference (fail-open for the one class that must never be deletable).
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probes_1.py  (probe P3_retention_class_downgrade)`
- **Impact:** A mislabelled evidence artifact loses its deletion protection without any error.
- **Suggested repair scope:** Raise a typed error for an unknown retention class during migration.
- **Owner paths:** `intergrax/contracts/artifact_ref.py`

### F-D8-11 - LOW - Digest validators accept a trailing newline

- **Rows:** PDC-D8-R002 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** new (related to dossier F-31)
- **Evidence:** `_DIGEST_RE = ^sha256:[0-9a-f]{64}$` with `re.match` accepts `sha256:<hex>\n` (probe P2: True). Same pattern in task_artifacts `_SHA256_HEX_RE`. A reference carrying such a value validates but can never verify its content.
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probes_1.py  (probe P2_digest_trailing_newline)`
- **Impact:** Malformed digests enter durable references; mismatch is only detected at verification time.
- **Suggested repair scope:** Use `fullmatch` or `\Z` in both validators; add a negative test.
- **Owner paths:** `intergrax/contracts/artifact_ref.py`, `intergrax/contracts/task_artifacts.py`

### F-D8-12 - LOW - Backend failures during an artifact operation leave no outcome event

- **Rows:** PDC-D8-R016 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** F-28 (confirmed)
- **Evidence:** `_guard` audits a fixed exception list (not found, denied, deletion refused, digest mismatch, download unavailable). Probe P10: a source raising RuntimeError produces only the authorization event; no `artifact.read.outcome` event. Same for sqlite or validation errors. Re-verified from dossier lead F-28.
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probes_1.py  (probe P10_unhandled_error_not_audited)`
- **Impact:** An authorized operation that fails unexpectedly is not in the evidence trail (R016: one event per operation outcome).
- **Suggested repair scope:** Add a catch-all that records a DENY/ERROR outcome event and re-raises.
- **Owner paths:** `intergrax/runtime/nexus/artifacts/governed_access.py`

### F-D8-13 - LOW - `reconstruct_artifact_lineage` does not check tenant or workspace of ancestors

- **Rows:** PDC-D8-R004, PDC-D8-R005 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** new
- **Evidence:** Probe P9: a child of tenant-A whose parent reference belongs to tenant-OTHER reconstructs as ['tenant-OTHER/p', 'tenant-A/c']. Parents are matched by id and digest only.
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probes_1.py  (probe P9_lineage_cross_tenant_parent)`
- **Impact:** A pool mixing tenants can chain across tenants; callers must pre-filter. No store or port feeds this function today.
- **Suggested repair scope:** Require identical `scope.tenant_id` along the chain and raise `ArtifactLineageError` otherwise.
- **Owner paths:** `intergrax/contracts/artifact_ref.py`

### F-D8-14 - LOW - Legal hold: SQLite `INSERT OR REPLACE` overwrites the holder and reason; the in-memory store drops them; release needs no ownership

- **Rows:** PDC-D8-R009 - **Verification:** PROBE_CONFIRMED - **Dossier lead:** F-28 (confirmed)
- **Evidence:** Probes P13/P14: a second `place_legal_hold` replaces ('legal-counsel','litigation') by ('other-agent','x'); `InMemoryArtifactStore._holds` keeps only (tenant, artifact). `release_legal_hold` removes any hold for any caller with `artifact.legal_hold`.
- **Reproduction:** `PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probes_1.py  (probes P13, P14)`
- **Impact:** The legal-hold trail is lossy and overwritable by a second holder.
- **Suggested repair scope:** Keep holds append-only (hold id per holder) and release only the caller's hold, or record holder changes as events.
- **Owner paths:** `intergrax/runtime/nexus/artifacts/sqlite_artifact_store.py`, `intergrax/runtime/nexus/artifacts/in_memory_artifact_store.py`

### F-D8-15 - LOW - Run artifact bundle builder follows file symlinks under the sandbox root, and bundle entry types still allow an empty sha256

- **Rows:** PDC-D8-R002, PDC-D8-R004 - **Verification:** CODE_READ, NOT_REPRODUCED - **Dossier lead:** F-27 (code confirmed, not reproduced)
- **Evidence:** `_sandbox_artifact_refs` uses `rglob('*')`, `is_file()` and `read_bytes()` (all follow symlinks) and publishes the sha256 of the target. NOT reproduced: this host cannot create symlinks (WinError 1314); finding stands on code reading. `ApplicationArtifactRef/WorkspaceArtifactRef/SandboxArtifactRef.sha256` default to '' and are validated only for format; the digest becomes mandatory only in `RunArtifactBundle.durable_refs`, which has no product caller (F-D8-02). From dossier lead F-27.
- **Reproduction:** `Code reading of intergrax/runtime/nexus/orchestration/run_artifact_bundle_builder.py (_sandbox_artifact_refs); probe F27 in ia_d8_probes_2.py logs the symlink limitation.`
- **Impact:** Hash oracle for a file outside the sandbox (read pre-dates the batch); bundle metadata can carry no digest.
- **Suggested repair scope:** Skip symlinks and verify `resolve()` stays under the root; make `sha256` required on new bundle entries or project through `durable_refs` at publication.
- **Owner paths:** `intergrax/runtime/nexus/orchestration/run_artifact_bundle_builder.py`, `intergrax/contracts/task_artifacts.py`

### F-D8-16 - LOW - Test strength: no hostile-source or shipped-composition test; two tenant checks unpinned

- **Rows:** PDC-D8-R005, PDC-D8-R006, PDC-D8-R008 - **Verification:** TEST_RUN + PROBE - **Dossier lead:** F-29 (confirmed)
- **Evidence:** No D8/XD test uses a source that returns another tenant's reference (grep `tenant_mismatch` in tests/qualification/pdc_x: all assertions are request-tenant mismatches); the stored-tenant re-check in `read` is therefore unpinned (probe_hostile_source shows it works today). No test composes a default host and asserts that a sweep is registered or the port exists; the P6-R wiring tests assert helper behaviour. Re-verified from dossier lead F-29.
- **Reproduction:** `grep -rn 'tenant_mismatch' tests/qualification/pdc_x ; PYTHONPATH=. uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/adversarial_scripts/ia_d8_probe_hostile_source.py`
- **Impact:** Defence-in-depth checks and the wiring claims can regress unnoticed.
- **Suggested repair scope:** Add a hostile-source test and a composition test that fails when the shipped worker bootstrap registers no artifact sweep.
- **Owner paths:** `tests/qualification/pdc_x/d8/test_d8_governed_port.py`, `tests/qualification/pdc_x/p6_repair/test_p6r_d8_worker_bootstrap.py`

### F-D8-17 - INFO - Documentation and ownership inconsistencies, residual alias use

- **Rows:** PDC-D8-R001, PDC-D8-R015, PDC-D8-R017, PDC-D8-R018 - **Verification:** CODE_READ - **Dossier lead:** new
- **Evidence:** (a) PDC_OWNERSHIP_MATRIX row says `application_artifacts.py` 'is removed'; ADR says remove 'under COMPAT-X R5 shim policy'; the code keeps a re-export shim (consistent with the architecture doc, not with the matrix). (b) plan/ARTIFACT_LIFECYCLE.md 'Open items' still lists hub registration and an artifact store opener as open; both exist (capabilities/README.md:79, sqlite_opens.py:87). (c) The R015 matrix row cites `CURRENT_HEAD_DELTA` '0 files (llm_adapters)', unrelated to content_storage; the real delta since EBH-5 (33ecf19cc) is one file, cosmetic plus the read-only helper `artifact_content_object_key`. (d) taxonomy.py names `tool_runtime.py` as owner of `tool_output` while the spill lives in nexus/artifacts/spill.py. (e) `runtime/observability/emitter.py:39` still imports `ArtifactRef` from the bridge under TYPE_CHECKING (resolves only through the module `__getattr__` shim; invisible to type checkers).
- **Reproduction:** `Reading the cited files.`
- **Impact:** Reviewers can read the wrong closure state.
- **Suggested repair scope:** Reconcile the ownership matrix and plan doc at P8; switch the emitter annotation to `TraceArtifactRef`.
- **Owner paths:** `docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md`, `intergrax/runtime/observability/emitter.py`, `intergrax/runtime/nexus/artifacts/taxonomy.py`

### F-D8-18 - INFO - The B-D8 implementation commit carries an AI-attribution trailer

- **Rows:** PDC-D8-R001 - **Verification:** CODE_READ - **Dossier lead:** new
- **Evidence:** `git show 92ee26bf3` ends with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; the repo rule `intergrax-no-ai-attribution` forbids it. The commit is merged history.
- **Reproduction:** `git show --format=%B -s 92ee26bf3`
- **Impact:** Process only; no product effect.
- **Suggested repair scope:** None required for history; noted for CERT.

### F-D8-19 - INFO - Adjacent partial-results test is red (not D8 evidence)

- **Rows:** PDC-D8-R011 - **Verification:** TEST_RUN - **Dossier lead:** new
- **Evidence:** tests/unit/runtime/long_running/test_partial_results.py::test_build_task_progress_view_aggregates_checkpoints fails (`human_request_expires_at` is None; the fixture hard-codes 2026-05-27, before the audit date, cause not investigated). The R011 claim (artifacts do not own partial results) is structural and verified separately.
- **Reproduction:** `uv run pytest tests/unit/runtime/long_running/test_partial_results.py -q  (logs/pytest_r011_partial_results_adjacent.log)`
- **Impact:** None on D8.
- **Suggested repair scope:** Owner of long_running tests.
- **Owner paths:** `tests/unit/runtime/long_running/test_partial_results.py`

### F-D8-20 - INFO - Known classified reds and the R-05 exit hang were not reproduced in D8 runs

- **Rows:** n/a - **Verification:** TEST_RUN - **Dossier lead:** F-09 (not reproduced for D8 paths)
- **Evidence:** The three D8-scope pytest commands exited on their own (exit 0). `xd_03` (the dossier F-09 trigger) was not run. EBH-7 identity, roadmap_replay_x and dist_scale_x/Docker were not run and are not re-reported.
- **Reproduction:** `logs/pytest_d8_xd09_xd10.log (EXIT=0)`
- **Impact:** None.
- **Suggested repair scope:** None.

## 5. Dossier leads routed to IA-D8 (re-verified at the audited SHA)

| Lead | Dossier claim | Result at `c08efe46a` | Finding |
| --- | --- | --- | --- |
| F-01 | Workspace scoping of governed read optional | CONFIRMED (probe P6) | F-D8-04 |
| F-02 | Sweep audit optional; delete removes versions written after selection | CONFIRMED, both parts (probes F02, in-memory and SQLite) | F-D8-05 |
| F-04 | Hardening capabilities without production caller (D8 instance: artifact services) | CONFIRMED for D8 (grep) | F-D8-01, F-D8-02 |
| F-10 | `_guard` ignores the authorization decision | CONFIRMED (probe F10) | F-D8-08 |
| F-27 | Bundle builder follows symlinks and publishes a digest | Code read confirms; NOT reproduced (host cannot create symlinks) | F-D8-15 |
| F-28 | Store and retention edge cases | CONFIRMED: naive/aware TypeError, tampered row, unaudited errors, hold overwrite, negative limit | F-D8-09, F-D8-12, F-D8-14 |
| F-29 | Redundant tenant checks unpinned | CONFIRMED by grep (no hostile-source test); the check itself works (probe) | F-D8-16 |
| F-31 | Presign tool always raises; stricter sha256 validators | CONFIRMED as designed; validators still accept a trailing newline | F-D8-11 |
| F-09 | pytest does not exit after `xd_03` | Not reproduced for D8 paths; `xd_03` not run | F-D8-20 |

Addenda 1-4 contain no D8-routed finding (they list D8 only as part of the per-domain audits to come).

## 6. Test and probe matrix (summary)

| Run | Command (one process at a time) | Result |
| --- | --- | --- |
| D8, XD-09, XD-10 | `uv run pytest tests/qualification/pdc_x/d8 tests/qualification/pdc_x/xd/test_xd_09_artifacts_context.py tests/qualification/pdc_x/xd/test_xd_10_artifacts_security.py -q` | 134 passed, 132.9 s, exit 0 |
| R015 replay | `uv run pytest tests/unit/collaborative_work/test_artifact_content_storage.py tests/unit/collaborative_work/test_artifact_content_storage_architecture_gates.py tests/unit/collaborative_work/test_artifact_execution_lineage_architecture_gates.py -q` | 52 passed |
| R011 adjacent | `uv run pytest tests/unit/runtime/long_running/test_partial_results.py -q` | 1 passed, 1 failed (F-D8-19) |
| Requirements schema | `uv run python scripts/qualification/pdc_x_requirements.py --check` | OK, 172 rows |
| Probes | four scripts under `iiac_evidence/PDC-X/D8/adversarial_scripts/` | logs under `logs/`; results cited per finding |

The first background launch of the D8 run (flag `-p no:cacheprovider`) aborted on an unknown config option and was repeated without it; only the clean run is recorded. Probe P1 in `ia_d8_probes_1.py` hit an import error (wrong class name) and is superseded by `ia_d8_probe_create_artifact.py`.

## 7. Limitations (what this audit did not cover)

- Product behaviour was probed with in-memory and SQLite stores, a fake object storage and the real in-memory governance pipeline. No real object storage, vendor sandbox, PostgreSQL or Docker was used (C10/C15 freeze, declared exception); R015 and R010 are therefore contract-level evidence.
- The full Collaborative Work repository qualification suites (`repository_qualification_suite`, SQLite and PostgreSQL repositories) were not replayed; only the content storage files named above were. R003's business-artifact side rests on the earlier EBH-5 certification and on reading `content_storage.py`.
- No mutation testing was run; test strength rests on reading the D8 tests, grep for hostile-source cases and probes (F-D8-16).
- F-D8-15 (symlinks) is code-read only. The run artifact bundle builder was not exercised end to end.
- Runtime inspection (`runtime_inspection/adapters/artifact_read.py`) was read but not run; whether governance audit events reach inspection views was not traced beyond the recorder sink (R016).
- Concurrency of SQLite writers across processes was not stress-tested; `BEGIN IMMEDIATE` and the primary key were read, not exercised.
- Not run: EBH-7 identity gate, `roadmap_replay_x`, `dist_scale_x`, `xd_03` and the other XD files, CI contexts, `igx_gov ticket`, the full `tests/unit` tree, live providers. Other domains' audit reports were not read (independence).
- Whether host composition is a PDC-X claim is an orchestrator decision (also open as dossier F-04). If it is not, F-D8-01 and F-D8-02 become a scope statement and the affected rows could become PASS_SCOPED only through an Operator-authorized exclusion; the auditor does not grant one.
- No certification, FRZ promotion, closure or production-readiness claim is made. This verdict is void for any path changed after `c08efe46a` that touches the D8 owner paths.

## 8. Evidence

`docs/project/maintainers/qualification/iiac_evidence/PDC-X/D8/`: `iiac_audit_session_manifest.PDC-X.d8.json`, `iiac_audit_d8_findings.json`, `audit_scope.json`, `test_matrix.txt`, `EVIDENCE_MANIFEST.txt` (sha256 of every file), `logs/`, `adversarial_scripts/`. Manifest verifiers `verify_iiac_audit_manifest.py` and `verify_iiac_role_separation.py` outputs are recorded in `logs/`.
