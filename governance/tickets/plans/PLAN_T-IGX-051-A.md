# PLAN T-IGX-051-A — T-AF2-RW-TENANT: tenant binding (sandbox sessions, object keys, tool context, artifact read, observation tenant, install CAS, MCP tenant, unscoped memory)

Draft revision 2, 2026-11-17 (revision 1: digest `77e5d3db0922e3bed45a1821a0a7a6de2161c60119f1e68992af130e6ffc32d2`, kept in `rev1/`). Nothing was run and nothing was changed outside `.tmp`. No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this plan; labels such as "DS-EXC", "PDC register", "audit" are quoted with their source. Evidence notation `path@H:line`; `H` = `05699b633cdb1318c215f6529e449420efc1382c` (`origin/arch-freeze/rw-train`, the merge of PT-WA, PR #642). Of the 14 product paths of the write set only `intergrax/applications/_shared/harness_host_runtime.py` differs between `H` and `origin/development` `2727a2c2364a97492202b931375772a400279a9b` (PT-WA; checked with `git diff --name-only`). Defects are described by ID and title only. `FACT_CHECK.md` has the claim-by-claim table (rows F-01..F-77 are code facts whose line numbers a script resolved at `H`; F-78..F-102 are scope, register and plan facts resolved by the same script; F-103..F-112 are counts and register facts read by hand; U-01..U-10 are the items not verified).

Source of scope: `RW_PLAN.md` **v18** Appendix A1 block `T-AF2-RW-TENANT (T-IGX-051-A)` (19 product firm paths, 32 firm in total, 42 conditional; section a row T20; revision 1 was written against v13, 14 / 22 / 0, and the orchestrator ruling Q-TEN-01 of v14 amended A1), its negative-test rows NT-ROW-065..074 (section d), its regression row (section e) and its lane and overlaps (sections b, c). Plan of record section E (DS-EXC dispositions) and section G (the RW train row) give the intent. This plan does not widen A1. Where the code shows that A1 is too narrow, section 4 lists the remaining requests (GAP-10 to GAP-12) and the question; they are not part of the write set until the RW-PLAN is amended and the plan digest is recomputed.

## Revision history

| Revision | Digest | Change |
|---|---|---|
| 1 | `77e5d3db0922e3bed45a1821a0a7a6de2161c60119f1e68992af130e6ffc32d2` | first draft against RW-PLAN v13 (14 / 22 / 0) |
| 2 | recorded in the ticket | reconciled with RW-PLAN v18 (19 / 32 / 42) and the step-0 dry run of the orchestrator at the train head `881428fb231f34215a67364aa5788a4f2f986e83` (record `dry-run/DRY_RUN_T-IGX-051-A.md`, label "DRY_RUN_FINDINGS" quoted from it): DR-TEN-01 write set; DR-TEN-02 new Part I (TENANT-X carrier re-pin, request GAP-10); DR-TEN-03 closed refusal conditions of Parts A and H; DR-TEN-04 dependency on PT-WD (Q-TEN-14); DR-TEN-05 mechanical lint commit L0; DR-TEN-06 test markers and the `Implementer session:` rule |

**Serialization (RW-PLAN section c, lane 7).** This ticket starts only after PT-WA and PT-WB and after these have merged into `arch-freeze/rw-train`: PT-WF (policy_wiring.py; plan Z tenant wiring), PT-FC1 (harness_host_runtime.py), T-AF2-RW-SUPPLY (admin_service.py), PT-FC3 (environment_wiring.py), PT-FC2 (memory_wiring.py, memory_provider_admission.py) and T-AF2-RW-TRUST (the shared `tests/unit/rag/*` glob). It runs in parallel only with T-AF2-RW-TYP, and PT-T01 (lane 8) starts after it. Every shared file is read at the train head at that time (step 0), not at `H`; line numbers in this plan are `H` evidence and will move. At the dry-run head `T` = `881428fb2` none of the 74 v18 paths differs from `H` (dry-run record); the predecessors named above had not merged then.

## 1. Purpose and findings

One product-path ticket of the RW train (DP-11 single integrated wave). Binding intent (plan of record section E, quoted): per-tenant sandbox sessions; tenant-bound object keys; tenant in the tool context; artifact read bound to principal and run; `tenant_id` on the observation; CAS install; MCP without harness auth refuses a client-asserted tenant; cross-tenant read, write and install rejected; client-asserted tenant refused.

| Finding | Source label | What this ticket does | What it cannot close alone |
|---|---|---|---|
| EX-D4-04 (PDC-D4-R018) | DS-EXC, REMEDIATION (tenant) | Part A: the hosted sandbox session is opened under the host's own tenant and a per-host task label, never the fixed `harness`/`bootstrap` labels; a strict host without a tenant refuses; blank labels are refused | tenant-scoped sessions inside the backend contract (the register names `SandboxHostBackend`; A1 has no contract path, GAP-05); provider-side isolation is PROD-Q |
| EX-D8-03 (PDC-D8-R005/R006) | DS-EXC, REMEDIATION (tenant) | Part B: `ToolWiringContext` carries the tenant; raw `storage.*` tools bind object keys to it and emit an audit event per call | an Operator decision on the storage tool contract (the register: tenant on the wiring context, or retire the raw tools; Q-TEN-04); the audit port (Q-TEN-05) |
| EX-L-02 (PDC-D8-R005/R006) | DS-EXC, REMEDIATION (tenant) | Part B: the object-touching `workspace.*` tools (export, import) use the same tenant-bound keys and audit | legacy unscoped keys become unreachable by the tools (no migration, Q-TEN-04) |
| EX-D8-04 (PDC-D8-R005/R006) | DS-EXC, REMEDIATION (tenant) | Part C: a governed artifact read or download is bound to the run of the access context; a read by another run of the same tenant is refused with an audited reason code | principal binding is an Operator decision (Q-TEN-07); cross-run reads need an explicit typed grant that no caller has today |
| EX-D5-05 (PDC-D5-R015) | DS-EXC, REMEDIATION (tenant) | Part D: `tenant_id` on the base observation model with a non-blank rule; the strict host without a tenant refuses at policy wiring | the literal "required" field breaks three caller files, the SUPPLY-owned registry subclass and persisted records, all conditional in A1 (GAP-04; Q-TEN-02); the pinned CTRL-X test is firm (GAP-03); the strict-host refusal at the registry is PT-WF Part JA |
| EX-L-09 (G-12 part) | DS-EXC, REMEDIATION | Part E: `install_agent` into an occupied slot is refused; two concurrent installs into one slot: exactly one wins (in process) | a cross-process "expect absent" compare needs the store files (GAP-02); an expected-revision token on the request needs `admin_models.py` (GAP-01); the G-09 part is PT-WF (DP-23) |
| R-P8-01 | PDC register, MCP final closure limitation 5 | Part F: a typed composition guard and a `fails_closed` refusal path so that a host that fails closed never admits an asserted tenant or user | the three shared MCP call sites are firm in A1 (GAP-06) but the application facades that would pass `fails_closed` are not (GAP-11); the register row closes only after the B-1 check X-A8-022 (NOT_REPRODUCED at B) and an independent audit |
| F-D2-11 (tenant part) | PDC debt register | Part G: reindex hooks are tenant-scoped; a hook registered for one tenant never runs for another | the ingest forwarder passes the tenant; callers of the freshness assessment outside A1 do not yet (GAP-07) |
| EX-D2-06 (PDC-D2-R002) | DS-EXC, OPERATOR DECISION (STRICT forbids unscoped memory) | Part H (class C): a strict host composing any memory capability without a tenant refuses composition | the Operator decision itself (Q-TEN-10); step 0 confirms which capabilities are unscoped at the train head |

Related items named in the RW-PLAN row but owned elsewhere, not edited here: FRZ-TEN-10 (moved to PT-WF Part JA by dedupe D-04; this ticket keeps only the MCP client-asserted tenant), FRZ-TEN-01 and the G-09 part of EX-L-09 (PT-WF, DP-23), FRZ-CFG-02, FRZ-CFG-06 and FRZ-SEC-04 (owner PT-FC3; this ticket is a dependent, RW-PLAN section g). The FRZ rows stay unpromoted; no row is claimed.

## 2. Design rules applied to every part

1. **Fail closed, no default that widens authority.** A missing tenant, audit sink or posture input refuses composition or the call; it never falls back to a shared namespace. Lab and unit constructions pass an explicit value (`intergrax-env-vendor-qualification`: missing mandatory configuration fails closed).
2. **One strictness notion.** The "strict host" notion is `wiring_fails_closed(env)` of `intergrax/applications/_shared/wiring_posture.py`, created by PT-FC2 (FC-2 plan section 4.6: FC-2 owns the file; later tickets import it and define no equivalent; F-100). This ticket imports it and defines no posture helper of its own. If the file is absent at the train head the ticket stops (step 0).
3. **No second identity authority.** The tenant is the host's composition tenant (`tenant_id` of `wire_application_environment` / `build_harness_host_runtime`, F-04, F-12) or the verified principal's tenant. A client-supplied tenant never establishes identity.
4. **Normalize once.** The host harness passes the raw tenant to environment wiring (F-12) and a stripped-or-None value to the evaluation wiring (F-45). Environment wiring normalizes the tenant once (strip; blank is treated as absent) and every part below uses that value.
5. **Contract surfaces get no new `Any`, `object` or reflection dispatch** (OD-AF-11). The write set has no path under `intergrax/contracts/`; `intergrax/contracts/artifact_ref.py` is only read (F-31..F-33). New types are frozen dataclasses, `StrEnum`s or `Protocol`s with concrete field types.
6. **No secrets in code, logs, argv or test output.** Audit events carry a tenant id, a tool id, an outcome, a reason code and a SHA-256 digest of the logical key; never the key body, the object body or a credential.
7. **No Docker, vendor integration or production configuration change** (C10/C15 freeze). Backends in tests are in-process fakes, labelled as stubs in the PR body; tenant isolation inside a real sandbox or object store provider is PROD-Q.
8. **One change, one signature.** A changed function has one signature; callers inside the write set are updated in the same commit group. Callers outside the write set are a step-0 census item and a question (section 4), not a silent edit.
9. **Existing tests that pin the old behaviour** are named in section 4.2. This plan does not edit them; tests in the v18 A1 are edited in the part that changes the behaviour; any other test that turns red is a stop and an amendment request.
10. **Mechanical lint commit first (DR-TEN-05).** The pre-commit gate runs `ruff check` and `ruff format --check` on every staged file (`run_pre_commit_gate.py@T:36-53`), and several write-set files carry findings at the train head. The first commit of the ticket (group L0) is a behaviour-free lint commit of exactly the files that the ticket stages, made after PT-FC2, PT-FC3 and PT-WF have reformatted the shared hotspots, with the locked ruff version (0.15.20). Section 8 lists the findings of the dry run.
11. **Test markers (DR-TEN-06).** New test files carry the marker `unit` and not `gate` where they use threads or compose hosts (the full-run purity gate forbids `threading.Thread(` and `wire_application_environment(` in `gate`-marked unit tests); a red-first assertion on an absent API uses `inspect.signature`, `dataclasses.fields` or `model_fields`, never `getattr` (the harness no-getattr gate is already red at the train head and must not grow).

## 3. Parts

Each part is a separate commit group. The order is in section 8.

### Part A — per-tenant hosted sandbox sessions (EX-D4-04)

State at `H`. `wire_application_environment` opens the hosted sandbox session through `resolve_hosted_sandbox_session` with the literals `tenant_id="harness"` and `task_id="bootstrap"` whenever no session was injected and an integration profile exists (`environment_wiring.py@H:439-448`); the result is handed to the tool wiring (`:473`). The resolver already takes tenant and task (`sandbox_host_wiring.py@H:15`) and forwards them to `HostedSandboxSession.open` (`:30`), which stores them as plain attributes (`hosted_session.py@H:44`) and creates the backend session without them (`:67`, `:71`). The function receives the host tenant as an optional parameter (`environment_wiring.py@H:243`).

Design.
- `environment_wiring.py`: after the tenant is normalized (rule 4), the hosted session is opened with `tenant_id=<normalized host tenant>` and `task_id=<host label>`; the draft label is the manifest application id (the manifest is in scope; step S0-1 confirms the name). When the tenant is absent: if `wiring_fails_closed(env)` **and the integration profile resolves a `SANDBOX_HOST` backend (a prebuilt instance or a configured slug of that category)** the composition raises a typed `HostedSandboxTenantRequiredError(ValueError)` (new, in `hosted_session.py`); a strict host whose profile resolves no sandbox host composes as before (DR-TEN-03: five ci-smoke nodes of `test_diag_foundation_5_destructive_proof.py@T:143-156, :195-376` compose product hosts with a resolved integration profile, no tenant and no sandbox host and must stay green); otherwise the host keeps the two legacy labels, now as named module constants marked "lab only" in `hosted_session.py` (Q-TEN-12). An injected `sandbox_session` is used as before.
- `hosted_session.py`: `HostedSandboxSession.__init__` and `open` strip and refuse blank `tenant_id` and `task_id` (`ValueError`); the two lab constants and the error type live here. `execute` and the audit entries are unchanged (the sandbox audit model has no tenant field and is outside A1).
- `sandbox_host_wiring.py` is unchanged (its signature already carries the arguments).

Not done here: tenant-scoped sessions in the `SandboxHostBackend` contract and in provider backends (GAP-05); the isolation claim stays with the provider substrate (PROD-Q). The test proves that two hosts of two tenants open two backend sessions labelled with their own tenants; it does not prove isolation inside a backend.

### Part B — tenant in the tool context; tenant-bound keys; audit (EX-D8-03, EX-L-02)

State at `H`. `ToolWiringContext` is a mutable dataclass with the object storage and the shadow workspace and no tenant field (`wiring.py@H:77`, `:98`); environment wiring builds it from the integration profile without a tenant (`environment_wiring.py@H:347`). `storage.get`, `storage.put` and `storage.delete` pass the caller-chosen key verbatim to the provider (`storage/service.py@H:74`, `:76`, `:116`); the only refusal is the governed collaborative-work prefix (`:33`, `:60`). `workspace.export_artifact` and `workspace.import_artifact` write and read the caller-chosen storage key verbatim after the same prefix refusal (`workspace/service.py@H:129`, `:148`, `:168`). `storage.presigned_url` is already disabled and stays so (`storage/service.py@H:97`). No tool call emits an audit event.

Design.
- `wiring.py`: `ToolWiringContext` gains `tenant_id: str | None = None` and `storage_audit_sink: StorageAccessAuditSink | None = None`. New in the same module: `StorageAccessOutcome` (`StrEnum`: `ALLOWED`, `DENIED`), a frozen dataclass `StorageAccessAuditEvent` (`tenant_id: str`, `tool_id: str`, `outcome: StorageAccessOutcome`, `key_digest: str`, `reason_code: str`) and a `Protocol` `StorageAccessAuditSink` with `record(event) -> None`. Defined here (not in the service) to avoid an import cycle with the services.
- `environment_wiring.py`: after the tool context is built, `replace(wiring_context, tenant_id=<normalized tenant>, storage_audit_sink=<adapter>)`; the adapter over the host's existing event or audit port is chosen at step S0-3 (Q-TEN-05). The runtime event bus is created after the tool wiring at `H` (`environment_wiring.py@H:450` against `:502-513`), so the adapter is lazy or the construction order changes; neither is decided here.
- `storage/service.py`: one private `_scoped_key(ctx, key)` used by every raw tool: it requires a non-blank `ctx.tenant_id` (else `TenantScopeRequiredError(PermissionError)`, new), refuses a tenant id that is not one safe segment (empty, `.`, `..`, contains `/` or `\`), refuses a logical key that begins with the tenant namespace root (so a caller cannot address another tenant's namespace), and returns the physical key `tenants/<percent-encoded tenant>/<normalized logical key>`. The existing governed-prefix and `..` refusals run first on the logical key. Outputs return the logical key. Every call, allowed or refused, records one `StorageAccessAuditEvent`; a missing sink raises `RuntimeError("storage_audit_sink_not_configured")`, the pattern of `_require_storage`.
- `workspace/service.py`: `workspace_export_artifact` and `workspace_import_artifact` use the same scoped key and audit; the shadow-workspace file tools (`write_file`, `read_file`, `list_files`, `snapshot`, `delete_file`, `search`) address the host's own workspace and are not keyed (Q-TEN-06).

Edited tests (firm in v18 A1): `tests/unit/tools/providers/storage/test_storage_tools.py`, `tests/unit/tools/providers/test_t10_catalog_tools.py`, `tests/qualification/pdc_x/d8_repair/test_d8r_storage_tools_gate.py` and `tests/qualification/pdc_x/p8_repair/test_r3_governed_keys.py` build a `ToolWiringContext` for the raw tools; each gets a tenant and a recording audit sink in its fixture and asserts the logical key in the output and the physical key in the provider. Their existing refusals (governed prefix, `..`, presign) stay asserted.

Not done here: any change to the object storage contract or providers; migration of objects written before this change (they are unreachable by the tools, Q-TEN-04); the governed artifact port (Part C).

### Part C — artifact read bound to run (EX-D8-04)

State at `H`. `ArtifactAccessContext` carries the agent identity, task, run and attempt (`governed_access.py@H:107`); its tenant is the agent's tenant (`:118`). The read action compares the stored tenant and workspace with the context and verifies the digest (`:281`); the single gate compares the request tenant with the context tenant and then authorizes the capability (`:347`). The reference stores the producing run (`artifact_ref.py@H:127`, `:132`) and the producer agent id (`:97`), but nothing compares them with the reader. A system actor context exists for background work (`governed_access.py@H:121`) and the retention sweeper uses it (`retention.py@H:234`).

Design.
- `ArtifactAccessContext` gains a typed field `read_binding: ArtifactReadBinding` (`StrEnum`: `RUN_BOUND` default, `TENANT_SWEEP`); only `for_background_execution` sets `TENANT_SWEEP`. A frozen `ArtifactCrossRunGrant` (tuple of artifact ids plus the reader run id) is the only other way across runs; no caller creates one yet.
- `GovernedArtifactReadPort.read` and `issue_download_url`: when the binding is `RUN_BOUND`, the stored `lineage.run_id` must equal `context.run_id` or a grant must name the artifact; otherwise `ArtifactAccessDenied("run_mismatch", …)`, which the gate already turns into an audited deny outcome (`governed_access.py@H:382-384`). Legal hold and delete keep tenant and capability authorization only (the sweeper path is unchanged).
- Principal binding (the reader agent equals the producer agent unless granted) is written as a conditional node and a question (Q-TEN-07); the draft default binds the run only.

Not done here: callers that build contexts for runs other than the producing run (step S0-4 census; `agent_runtime_context.py@H:512` composes the port, F-35).

### Part D — tenant on the observation; strict host refuses (EX-D5-05)

State at `H`. The base `OnlineEvaluationObservation` has no tenant field (`online_evaluation_models.py@H:18`, `:27`); the tenant subclass in the registry module declares an optional tenant (`online_evaluation_registry.py@H:55`, an RW-SUPPLY path); `record_shadow_observation`, `NexusLoop` and the eval tool build the base model (`online_evaluation.py@H:115`, `nexus_loop.py@H:1012`, `tools/providers/eval/service.py@H:57`); a CTRL-X test pins the absence of `tenant_id` on the base model (`test_ctrl_x_r2_evaluation_plane.py@H:82`). `policy_wiring.py` builds the evaluation wiring without a tenant (`:154`) and the evaluation wiring binds the registry to a tenant only when one is given (`evaluation_wiring.py@H:64`). The host harness passes a stripped tenant or None (`harness_host_runtime.py@H:520`).

Design (draft default, Q-TEN-02). `OnlineEvaluationObservation` gains `tenant_id: str | None = None` with a validator that refuses a blank string, so every observation can carry the tenant and a blank one is rejected at construction; the strict-host refusal is a typed error at policy wiring when `wiring_fails_closed(env)` and the host has no tenant. The literal "required" field of the row would break the three caller files above, the registry subclass (a SUPPLY path) and persisted legacy records, and is not possible inside A1 (GAP-03, GAP-04). `policy_wiring.py` and `harness_host_runtime.py` are firm paths of A1; **the expected edit is none**: PT-WF Part JA already makes a STRICT or PRODUCT host with an absent, empty or blank tenant refuse to wire the shared registry (NT-WF-A01..A04; F-101, F-102) and PT-WF merges first. Step S0-5 re-reads both files at the train head and edits only what PT-WF and PT-FC1 left undone.

Edited test (firm in v18 A1, GAP-03): `test_r2_eval_05_tenant_verdict_not_tenant_aware_at_this_stage` in `tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_evaluation_plane.py` asserts that `tenant_id` is NOT a field of the base model (F-42); it is rewritten to assert the new contract (the field exists and a blank value is rejected) and its TRACKED-debt comment is reworded; its name is changed to match. The note at `ctrl_x/catalog.py:145` is edited only if its text pins the absence (conditional path). Adding the base field adds one `class.field.tenant` carrier, see Part I.

Not done here: the registry subclass and file stores (RW-SUPPLY path, conditional GAP-04); the three caller files (conditional GAP-04, needed only for Option 1); the persisted-record migration.

### Part E — install CAS (G-12 part of EX-L-09)

State at `H`. `install_agent` reads the install by id for an idempotent replay (`admin_service.py@H:599`, `:616`), resolves the active installation of the slot (`:618`) and computes the current slot revision token, but only to authorize the mutation (`:634`). It then admits, persists and promotes with no expected-active argument (`:690`). `promote_verified_to_active` takes `expected_active_installation_id` as optional (`installation_service.py@H:158`), and the store skips the compare when it is None (`in_memory_stores.py@H:225`), then demotes the prior active and promotes the candidate (`:241`). `InstallAgentRequest` has no expected-revision field (`admin_models.py@H:113`); `upgrade_agent` has one and refuses a mismatch (`:175`, `admin_service.py@H:809`).

Design (draft default, Q-TEN-03; inside A1 only).
- `install_agent`: when the slot is occupied by an installation other than the requested one, the call raises `InstallationSlotConflict("slot occupied: use upgrade_agent with the expected active installation")` before admission, persistence or promotion. The idempotent replay of the same installation is unchanged. A replacement is made only through `upgrade_agent`, which already carries the expected token.
- Concurrency: the occupancy check, the persistence and the promotion run inside one per-slot lock held by the admin service (key: environment id plus slot id); the second of two concurrent installs sees the occupant and is refused. The lock is in process only. A cross-process "expect absent" compare is not expressible with the store contract (None means unchecked, F-51) and needs the store files (GAP-02).
- SUPPLY's Parts A and B edit the same function and merge first; this part rebases onto them and adds no admission step.
- The 25 `install_agent(` test files of v18 A1 are conditional: each is edited only if the step-0 census (S0-7) shows that it installs a second, different installation into one occupied slot; the edit moves that step to `upgrade_agent` with the expected active installation or gives it a second slot.

Not done here: the expected-revision token on the request (GAP-01); store changes (GAP-02); the G-09 untenanted registry (PT-WF).

### Part F — MCP: composition fails closed, asserted identity refused (R-P8-01)

State at `H`. `resolve_mcp_run_agent_identity` binds the tenant and user from the verified principal; without one it falls to the assertion-only branch `(asserted_tenant_id or default_tenant_id).strip()` (`mcp_harness_principal.py@H:195`). The default mode `HARNESS_BOUND` refuses a call without a verified principal before that branch (`:134`; default at `mcp_identity_mode.py@H:28`); `LOCAL_DEV_EXPLICIT` is a code-level opt-in (`:25`) that admits the assertion for in-process calls and loopback clients (`:47`). Two existing tests pin both behaviours (`test_mcp_identity_mode_fail_closed.py@H:132`, `:141`, F-62, F-63). The composition callers (`mcp_application_server.py@H:111`, `mcp_nexus_server.py@H:212`) are outside A1.

Finding (reading, not a run). At the default mode a host without harness auth already refuses; the part of R-P8-01 that remains is a host composed with `LOCAL_DEV_EXPLICIT`, or a host composed without harness auth that is never told it must fail closed. The negative test of NT-ROW-074 ("must be shown red at HEAD") can therefore be red only for a new, explicit refusal path; a refusal that changes the existing `LOCAL_DEV_EXPLICIT` admission would invert the MRC-1 decision D-4 and its pinned tests (outside A1). Q-TEN-08 records this.

Design (draft default).
- `mcp_harness_principal.py`: new `assert_mcp_identity_composition(*, identity_mode, harness_auth, fails_closed)` raising a typed `McpIdentityCompositionError(ValueError)` when `fails_closed` and (the mode is `LOCAL_DEV_EXPLICIT` or no harness auth is composed); new keyword `fails_closed: bool = False` on `resolve_mcp_run_agent_identity` which, when true, refuses the assertion-only branch with `MCP_PRINCIPAL_REQUIRED`. With the default value nothing changes for any existing caller.
- The three shared call sites are firm in v18 A1 (GAP-06): `compose_application_mcp_server` (`mcp_application_server.py@T:66-111`) and `build_nexus_mcp_server` / the identity-resolving task function (`mcp_nexus_server.py@T:212-230`, `:256-306`) gain `fails_closed: bool = False`, call `assert_mcp_identity_composition` and pass the value to `resolve_mcp_run_agent_identity`; `mount_catalog_tools_on_mcp` (`mcp_catalog_tools.py@T:38-72`) gains the same keyword and calls the composition guard before it mounts the tools. The value comes from the caller.
- The callers that hold the profile are the application facades `applications/*/mcp/server.py` (16 call sites of the two composition functions under `applications/` at `T`, F-119). They are not in A1 (GAP-11), so with their current code `fails_closed` stays false and the guard is not reached by a shipped composition; the ticket records that R-P8-01 stays partly open until the facades pass `wiring_fails_closed(env)`.

Not done here: the application facades named by ADR-MCP-RC-01 (GAP-11); the B-1 check X-A8-022 and the independent audit (the register: closes only after both).

### Part G — tenant-scoped reindex hooks (F-D2-11, tenant part)

State at `H`. The hooks are one module-level list (`embedding_version_policy.py@H:24`); the request carries no tenant (`:28`); registration takes only the callback (`:41`); every hook runs for every request (`:163`). The ingest and freshness paths call the policy (`ingest_pipeline.py@H:270`, `source_freshness.py@H:251`; both firm in v18 A1, GAP-07).

Design (Q-TEN-09). `ReindexQueueRequest` gains `tenant_id: str | None = None` (last field, default keeps construction compatible). `register_reindex_queue_hook(callback, *, tenant_id: str | None = None)`: a hook registered with a tenant runs only for a request with the same tenant; a hook registered without one is a platform-wide hook and runs for every request (the legacy behaviour, kept so existing registrations stay valid). A request without a tenant runs only platform-wide hooks: a tenant-scoped hook never runs for it (fail closed). `evaluate_ingest_embedding_version` and the internal notifier gain an optional `tenant_id` keyword; the per-hook isolation of PDC-D2-R005 is unchanged. The two forwarders are firm in v18 A1 (GAP-07): `ingest_pipeline.py` passes the tenant it has already validated as a non-blank string (`ingest_pipeline.py@T:263-270`, F-118) to `evaluate_ingest_embedding_version`, and `assess_retrieval_freshness` (`source_freshness.py@T:211`) gains an optional `tenant_id` keyword that it forwards to the notifier. Callers of the freshness assessment outside A1 do not pass a tenant yet (the census S0-2 names them), so a scoped hook fires from the ingest path and not from freshness assessments until they do; a request without a tenant runs only platform-wide hooks (fail closed). The five reindex tests of v18 A1 (conditional) are expected unchanged because every new keyword has a default.

### Part H — STRICT forbids unscoped memory (EX-D2-06, class C)

State at `H`. `resolve_memory_platform_wiring` takes an optional tenant (`memory_wiring.py@H:466`) and passes it into the external-store materialization context (`:435`); `memory_provider_admission.py` gates the final user profile store (`validate_memory_platform_wiring_admission`, `:161`) and reads the persistence requirement from the memory profile flags (`:97`); environment wiring builds the session manager without a tenant when the host has none (`environment_wiring.py@H:391`) after asserting the strict memory bootstrap (`:386`). The register says the "strict-profile notion" did not exist; it exists once PT-FC2 has merged (F-100).

Design (Q-TEN-10). A new pure function in `memory_provider_admission.py`, `validate_memory_tenant_scope_admission(env, *, tenant_id)`, raises a typed `MemoryTenantScopeError(ValueError)` when `wiring_fails_closed(env)`, at least one of the three flags `enable_user_memory`, `enable_org_memory`, `enable_long_term_memory` of the memory profile is true (a closed list; `MemoryProfile` has no session-storage flag, only `session_storage_plugin_id`, `sub_profiles.py@T:174-191`; session storage is always composed, `environment_wiring.py@T:391-395`, and does not count; `enable_session_vector_index` and the plugin ids do not count either, Q-TEN-10) and the tenant is absent or blank (DR-TEN-03: the product presets leave the three flags false, so the five df5 hosts and every other product default host stay unrefused; only the lab `harness_memory_profile()` enables them, `presets.py@T:103-109`); `resolve_memory_platform_wiring` calls it first. A lab host is unchanged. PT-FC2 edits both files first and adds organization and session admission; this part adds one function and one call and rebases.

Class C: step S0-6 measures at the train head which capabilities are composed without a tenant. If none is, the row is recorded NOT_REPRODUCED with evidence and the work on it stops; as a DS-EXC item it then goes back to the Operator (RW-PLAN re-disposition rule, F-84).

### Part I — TENANT-X carrier re-pin (DR-TEN-02; request GAP-10)

State at `T`. The TENANT-X replay-head reconciliation pins the number of `class.field.tenant` carriers (831) and the SHA-256 of their ids, and classifies every carrier added after the final evidence with a role and a reason (`_replay_x_carrier_reconciliation.py@T:13-18, :135, :207-215`; the classification map is `REPLAY_P7_ADDED_CLASSIFICATION`; it was re-pinned once by T-IGX-041-I). Parts B, D and G add four such carriers: `ToolWiringContext.tenant_id`, `StorageAccessAuditEvent.tenant_id`, `OnlineEvaluationObservation.tenant_id` and `ReindexQueueRequest.tenant_id`. The dry run simulated the four fields in memory and found three new red nodes (835 discovered against the pin 831, a delta of 35 against 31, a digest mismatch): `test_tenant_x_p0_current_head.py::test_tx_p0_r3_head_q02_class_field_carrier_count_stable`, `test_tenant_x_replay_carrier_reconciliation.py::test_real_data_has_no_violation` and `::test_gate_helper_raises_on_a_wrong_digest`. They are in the regression row `tests/qualification/tenant_x`, in no known-red entry, and the pin file is in no ticket's scope.

Decision (draft): the pin file is edited, not designed around. The four fields are real tenant carriers; renaming them to avoid the discovery would hide them from the register that exists to count them, and is refused. The edit is one coordinated re-pin at the TENANT head: add one entry per new carrier to the added-classification map (role and reason, text without secrets), set the total to the number the discovery reports at that head, and set the new ids digest. The three red tests are not edited; they go green when the pin and the digest agree with the discovery. The final number is not fixed here (PT-WF and RW-SUPPLY may add carriers before this ticket; SUPPLY's NT-ROW-087 records the same count), so the implementer recomputes it at the head and states it in the PR body. A pinned classification register is Operator-visible: the packet carries the edit as its own commit group (Q-TEN-13).

## 4. A1 v18 reconciliation and remaining requests

Revision 1 listed nine gaps against A1 of RW-PLAN v13. The orchestrator ruling Q-TEN-01 (RW-PLAN v14, F-114) amended A1; RW-PLAN v18 has 19 / 32 / 42 and is the write set of this revision (section 5). Status of each gap:

| Gap | Paths | State in v18 A1 | Treatment in this plan |
|---|---|---|---|
| GAP-01 | `intergrax/agent_distribution/admin_models.py` | conditional (only if Q-TEN-03 is ruled option B, a token on the request) | not edited in the draft default |
| GAP-02 | `agent_distribution/stores.py`, `in_memory_stores.py`, `sqlite_stores.py` | conditional (a cross-process compare, if ruled into the ticket) | not edited in the draft default; limit recorded |
| GAP-03 | the CTRL-X test of the observation model | firm | Part D |
| GAP-04 | `online_evaluation.py`, `nexus_loop.py`, `tools/providers/eval/service.py`, `online_evaluation_registry.py` | conditional (only if Q-TEN-02 is ruled option 1, a required field) | not edited in the draft default |
| GAP-05 | `intergrax/integrations/contracts/sandbox_host.py` | conditional (only if the backend contract is ruled into the ticket); provider backends name no path | not edited in the draft default; limit recorded |
| GAP-06 | `mcp_application_server.py`, `mcp_nexus_server.py`, `mcp_catalog_tools.py` | firm | Part F |
| GAP-07 | `rag/ingest/ingest_pipeline.py`, `rag/governance/source_freshness.py` | firm | Part G |
| GAP-08 | the audit adapter home | no path (stop rule) | step S0-3; a need is a stop and an amendment |
| GAP-09 | existing tests | firm: four storage-tool tests; conditional: the CTRL-X catalog note, five reindex tests, two memory tests, 25 `install_agent(` test files | Parts B, D, E, G, H |

Requests that A1 v18 does not satisfy (from the dry run; none is in scope until the RW-PLAN is amended and the digest recomputed):

| Gap | Path | Why | Default |
|---|---|---|---|
| GAP-10 | `tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py` | the carrier pin and digest go red when the four fields land (DR-TEN-02; Part I) | request: firm path of this ticket, Operator-visible edit (Q-TEN-13) |
| GAP-11 | `applications/*/mcp/server.py` (the facades; 16 call sites of the two composition functions) | they hold the profile that would pass `fails_closed` (Part F) | not requested; R-P8-01 stays partly open |
| GAP-12 | `intergrax/platform/enterprise_audit/impact_map.py`, `.../data/changed_since_impact_map.v1.json` (PT-WD firm paths) | `hosted_session.py` and `nexus_loop.py` map to no check; the PDC P6 impact-coverage test is already red at the train head for seven other modules (DR-TEN-04) | not TENANT scope; Q-TEN-14 asks PT-WD to cover the two paths or to add a precedence edge |

The five ci-smoke df5 nodes (`test_diag_foundation_5_destructive_proof.py`, DR-TEN-03) are not a gap: with the closed conditions of Parts A and H they stay green, and they are regression and census items only. If Q-TEN-10 or Q-TEN-12 is ruled so that they turn red, the file becomes a conditional request.

### 4.2 Existing tests whose pins the change inverts (names from reading, not run)

- Firm edits in A1: the CTRL-X evaluation-plane test (Part D) and the four storage-tool tests (Part B): `test_storage_tools.py`, `test_t10_catalog_tools.py`, `test_d8r_storage_tools_gate.py`, `test_r3_governed_keys.py`.
- Conditional edits in A1: the 25 `install_agent(` test files (Part E, only those that install twice into one slot), the CTRL-X catalog note, the five reindex tests and the two memory tests (expected unchanged).
- Stay green by design, regression only: `test_mcp_identity_mode_fail_closed.py` and `test_mcp_trust_a_principal_binding.py` (the new keyword defaults to false, F-62, F-63); the five df5 nodes.

## 5. Write set

Branch rule: product commits go only to branch `arch-freeze/rw-train`. The set below equals `RW_PLAN.md` v18 Appendix A1 block `T-AF2-RW-TENANT` (74 rows, checked row by row against the A1 table by script; all existing paths exist at `H` and at `T`, the 8 new test files have an existing parent directory). Counts: **19 firm product paths**, **13 firm test paths** (8 new, 5 edited), **9 conditional product paths**, **33 conditional test paths**; firm total 32, conditional total 42. One path per line. GAP-10 (the carrier pin) is a request, not part of this set.

### Product paths (ACT-N11), firm

```
intergrax/runtime/sandbox/hosted_session.py
intergrax/applications/_shared/environment_wiring.py
intergrax/tools/providers/storage/service.py
intergrax/tools/providers/workspace/service.py
intergrax/tools/registry/wiring.py
intergrax/runtime/nexus/artifacts/governed_access.py
intergrax/runtime/architecture/online_evaluation_models.py
intergrax/applications/_shared/policy_wiring.py
intergrax/applications/_shared/harness_host_runtime.py
intergrax/agent_distribution/admin_service.py
intergrax/applications/_shared/mcp_harness_principal.py
intergrax/rag/governance/embedding_version_policy.py
intergrax/applications/_shared/memory_wiring.py
intergrax/applications/_shared/memory_provider_admission.py
intergrax/applications/_shared/mcp_application_server.py
intergrax/applications/_shared/mcp_nexus_server.py
intergrax/applications/_shared/mcp_catalog_tools.py
intergrax/rag/ingest/ingest_pipeline.py
intergrax/rag/governance/source_freshness.py
```

Part map: A `hosted_session.py`, `environment_wiring.py`; B `wiring.py`, `storage/service.py`, `workspace/service.py`, `environment_wiring.py`; C `governed_access.py`; D `online_evaluation_models.py`, `policy_wiring.py`, `harness_host_runtime.py`; E `admin_service.py`; F `mcp_harness_principal.py`, `mcp_application_server.py`, `mcp_nexus_server.py`, `mcp_catalog_tools.py`; G `embedding_version_policy.py`, `ingest_pipeline.py`, `source_freshness.py`; H `memory_wiring.py`, `memory_provider_admission.py`. `policy_wiring.py` and `harness_host_runtime.py` are firm in A1; the expected edit is none (Part D). A firm path with no needed edit is allowed and is recorded in the PR body.

### Product paths (ACT-N11), conditional

```
intergrax/agent_distribution/admin_models.py
intergrax/agent_distribution/stores.py
intergrax/agent_distribution/in_memory_stores.py
intergrax/agent_distribution/sqlite_stores.py
intergrax/runtime/architecture/online_evaluation.py
intergrax/runtime/nexus/nexus_loop.py
intergrax/tools/providers/eval/service.py
intergrax/runtime/architecture/online_evaluation_registry.py
intergrax/integrations/contracts/sandbox_host.py
```

Conditions: `admin_models.py` only if Q-TEN-03 is ruled option B; the three store files only if a cross-process compare is ruled into this ticket; `online_evaluation.py`, `nexus_loop.py`, `tools/providers/eval/service.py`, `online_evaluation_registry.py` only if Q-TEN-02 is ruled option 1 (required `tenant_id`); `sandbox_host.py` only if the backend contract is ruled into this ticket. `nexus_loop.py` is also a PT-FC1 and PT-WF path. In the draft default none of the nine is edited.

### Tests, new, firm

```
tests/unit/runtime/sandbox/test_rw_tenant_hosted_session_per_tenant.py
tests/unit/tools/providers/test_rw_tenant_storage_workspace_keys.py
tests/unit/runtime/nexus/test_rw_tenant_artifact_read_binding.py
tests/unit/applications/test_rw_tenant_online_evaluation_observation.py
tests/unit/agent_distribution/test_rw_tenant_install_cas.py
tests/unit/applications/test_rw_tenant_mcp_client_asserted_tenant.py
tests/unit/rag/governance/test_rw_tenant_reindex_hook_scope.py
tests/unit/applications/test_rw_tenant_strict_unscoped_memory.py
```

`test_rw_tenant_reindex_hook_scope.py` also falls under the `tests/unit/rag/*` glob of T-AF2-RW-TRUST; TRUST merges first.

### Tests, edited, firm

```
tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_evaluation_plane.py
tests/unit/tools/providers/storage/test_storage_tools.py
tests/unit/tools/providers/test_t10_catalog_tools.py
tests/qualification/pdc_x/d8_repair/test_d8r_storage_tools_gate.py
tests/qualification/pdc_x/p8_repair/test_r3_governed_keys.py
```

### Tests, edited, conditional (each only if the step-0 census shows the change turns it red)

The CTRL-X catalog note (only if its text pins the absence of `tenant_id`); the five reindex tests (expected unchanged, every new keyword has a default); the two memory tests (expected unchanged):

```
tests/qualification/control_planes/ctrl_x/catalog.py
tests/unit/rag/governance/test_embedding_version_policy.py
tests/unit/rag/ingest/test_ingest_embedding_version_policy.py
tests/qualification/pdc_x/d2/test_d2_rag_freshness.py
tests/qualification/pdc_x/d2/test_d2_rag_freshness_wiring.py
tests/qualification/pdc_x/d2_repair/test_d2r_freshness_hook_isolation.py
tests/unit/applications/test_memory_wiring.py
tests/unit/applications/test_mem_audit5a_production_provider_admission.py
```

and the 25 `install_agent(` test files (Part E, only those that install twice into one occupied slot):

```
tests/integration/agent_distribution/test_ac6_trust_lifecycle_e2e.py
tests/integration/agent_distribution/test_canonical_agent_lifecycle_e2e.py
tests/integration/agent_distribution/test_enterprise_projection_rehydration_e2e.py
tests/qualification/pdc_x/d1/_d1_harness.py
tests/qualification/pdc_x/d1/test_d1_adversarial_lifecycle.py
tests/qualification/pdc_x/d1/test_d1_upgrade_retire.py
tests/qualification/pdc_x/d1_repair/test_d1_repair_scope_and_evidence.py
tests/qualification/pdc_x/xd/test_xd_11_observability.py
tests/unit/agent_distribution/test_agent_contract_authority.py
tests/unit/agent_distribution/test_agent_distribution_build_remediation.py
tests/unit/agent_distribution/test_agent_distribution_control_plane_governance.py
tests/unit/agent_distribution/test_agent_distribution_desired_state_idempotent_scope.py
tests/unit/agent_distribution/test_agent_distribution_desired_state_remediation.py
tests/unit/agent_distribution/test_agent_distribution_drain_recovery_remediation.py
tests/unit/agent_distribution/test_agent_distribution_tenant_denial_remediation.py
tests/unit/agent_distribution/test_agent_manager.py
tests/unit/agent_distribution/test_agent_platform_admin_service.py
tests/unit/agent_distribution/test_agent_platform_composite_scope_coexistence.py
tests/unit/agent_distribution/test_agent_platform_cross_layer_acceptance.py
tests/unit/agent_distribution/test_delegated_subtasks.py
tests/unit/agent_distribution/test_task_scoped_agents.py
tests/unit/applications/test_ac3_phase5_production_lifecycle_e2e.py
tests/unit/applications/test_ac4_phase9_production_composition_e2e.py
tests/unit/applications/test_historical_projection_isolation_phase4e.py
tests/unit/architecture/test_ebh_2e_ar1_a_r1_r1_install_authority_closure.py
```

No scripts and no docs.

**Forbidden (a need for any of these stops the work):** the paths of section 4 marked "request" (GAP-10 to GAP-12) until an amendment adds them; `intergrax/contracts/**` (the contract-ratchet scope); `retention.py` (RW-SUPPLY); `intergrax/applications/_shared/production_lifecycle.py` and `evaluation_wiring.py` (PT-WF); `tests/unit/runtime/architecture/test_diag_foundation_5_destructive_proof.py` unless Q-TEN-10 or Q-TEN-12 turns it red; `.github/`, `pyproject.toml`, `uv.lock`; the roadmap, the FRZ checklist, IIAC and PDC-X record directories; Docker or vendor work; real credentials, real object stores or real sandbox providers.

## 6. Negative tests

Types: R = a configuration that must now be refused or isolated (red at the base for a behavioural reason); S = structural (red because the structure is absent); C = control (green before and after) or conditional on a step-0 measurement (class C of the RW-PLAN). Every R and S node must fail at the pre-change commit by an assertion, not at import or collection (RW-PLAN red-first rule, F-83): a test of a new API first asserts the old surface through `getattr` or `inspect.signature` and fails on that assertion. Node ids are `file::test`. RW-PLAN ids: NT-RW-TEN-01..09 for NT-ROW-065..074 (section d, F-80). 9 NT ids, 45 test nodes.

| ID | NT row | Configuration | Expected refusal / result | Node id | Type |
|---|---|---|---|---|---|
| NT-RW-TEN-01 | 067 (EX-D4-04) | host composed for tenant A with an injected fake sandbox backend | the opened session carries tenant A and the host label, not `harness`/`bootstrap` | `test_rw_tenant_hosted_session_per_tenant.py::test_ten_01a_host_opens_session_under_host_tenant` | R |
| NT-RW-TEN-01 | 067 | two hosts (A, B) sharing one backend instance | two backend sessions, labelled A and B | `...::test_ten_01b_two_tenants_two_backend_sessions` | R |
| NT-RW-TEN-01 | 067 | strict host (fails closed) with a sandbox host configured and no tenant | composition refused (`HostedSandboxTenantRequiredError`) | `...::test_ten_01c_strict_host_without_tenant_refused` | R |
| NT-RW-TEN-01 | 067 | `HostedSandboxSession.open` with a blank tenant or task | `ValueError` | `...::test_ten_01d_blank_labels_refused` | R |
| NT-RW-TEN-01 | 067 | an injected sandbox session | used as is; no hosted session opened | `...::test_ten_01e_injected_session_unchanged` | C |
| NT-RW-TEN-01 | 067 | strict host, no tenant, a profile that resolves no sandbox host (the df5 shape) | composes as before | `...::test_ten_01f_strict_host_without_sandbox_host_unchanged` | C |
| NT-RW-TEN-02 | 069 (EX-D8-03) | the dataclass fields of `ToolWiringContext` | `tenant_id` is a field | `test_rw_tenant_storage_workspace_keys.py::test_ten_02a_tool_context_carries_tenant` | S |
| NT-RW-TEN-02 | 069 | object written through tenant A's context, read through tenant B's, one shared in-memory storage | B gets `found=False`; A reads its own object | `...::test_ten_02b_object_written_for_a_not_readable_as_b` | R |
| NT-RW-TEN-02 | 069 | raw `storage.get/put/delete/exists` | each call (allowed or refused) emits one audit event with tenant, tool id, outcome, key digest | `...::test_ten_02c_each_raw_call_emits_audit_event` | R |
| NT-RW-TEN-02 | 069 | context without a tenant, or without an audit sink | refused (`TenantScopeRequiredError` / `storage_audit_sink_not_configured`) | `...::test_ten_02d_missing_tenant_or_sink_refused` | R |
| NT-RW-TEN-03 | 070 (EX-L-02) | a logical key that names another tenant's namespace root | refused; tenant B's object is not reached | `...::test_ten_03a_foreign_namespace_key_refused` | R |
| NT-RW-TEN-03 | 070 | `workspace.export_artifact` under A, `workspace.import_artifact` under B with the same key | B: `object_not_found` | `...::test_ten_03b_workspace_export_import_tenant_bound` | R |
| NT-RW-TEN-03 | 070 | tenant ids that are empty, `.`, `..` or contain a separator | refused | `...::test_ten_03c_unsafe_tenant_segment_refused` | R |
| NT-RW-TEN-03 | 070 | the audit event of a refused call | holds tenant, tool id, reason code, key digest; no key body, no object body | `...::test_ten_03d_audit_event_carries_no_content` | R |
| NT-RW-TEN-03 | 070 | a key under the governed collaborative-work prefix | still refused (existing behaviour) | `...::test_ten_03e_governed_prefix_refusal_unchanged` | C |
| NT-RW-TEN-06 | 071 (EX-D8-04) | a governed read by another run of the same tenant | `ArtifactAccessDenied("run_mismatch")` | `test_rw_tenant_artifact_read_binding.py::test_ten_06a_read_by_other_run_refused` | R |
| NT-RW-TEN-06 | 071 | a read by the producing run | allowed | `...::test_ten_06b_read_by_producing_run_allowed` | C |
| NT-RW-TEN-06 | 071 | `issue_download_url` by another run | refused | `...::test_ten_06c_download_by_other_run_refused` | R |
| NT-RW-TEN-06 | 071 | the refused read | one audited deny outcome with the reason code | `...::test_ten_06d_denied_read_is_audited` | R |
| NT-RW-TEN-06 | 071 | a background-execution (system actor) context | legal hold and delete paths unchanged | `...::test_ten_06e_system_actor_paths_unchanged` | C |
| NT-RW-TEN-06 | 071 | reader agent differs from the producer inside the producing run | refused unless granted; **only if Q-TEN-07 binds the principal** | `...::test_ten_06f_other_principal_same_run` | C |
| NT-RW-TEN-04 | 068 (EX-D5-05) | the fields of the base observation model | `tenant_id` is a field | `test_rw_tenant_online_evaluation_observation.py::test_ten_04a_base_model_has_tenant_field` | S |
| NT-RW-TEN-04 | 068 | an observation with a blank `tenant_id` | rejected at construction | `...::test_ten_04b_blank_tenant_rejected` | R |
| NT-RW-TEN-04 | 068 | a valid tenant | accepted; round-trips through `model_dump` | `...::test_ten_04c_valid_tenant_accepted` | C |
| NT-RW-TEN-04 | 068 | strict host without a tenant through policy wiring | refused; **shared with PT-WF NT-WF-A01..A04 (NT-TEN-01-2); green at the TENANT base if PT-WF has merged, then not a TENANT red-first row** | `...::test_ten_04d_strict_host_without_tenant_refused` | R |
| NT-RW-TEN-05 | 065, 066 (EX-L-09, G-12) | `install_agent` into a slot occupied by another installation | `InstallationSlotConflict`; the active installation is unchanged | `test_rw_tenant_install_cas.py::test_ten_05a_install_over_occupied_slot_refused` | R |
| NT-RW-TEN-05 | 065, 066 | the same install repeated | idempotent replay returns the existing installation | `...::test_ten_05b_idempotent_replay_unchanged` | C |
| NT-RW-TEN-05 | 065, 066 | two concurrent installs with different ids into one empty slot (threads, barrier) | exactly one wins; the other is refused; one active installation | `...::test_ten_05c_two_concurrent_installs_one_wins` | R |
| NT-RW-TEN-05 | 065, 066 | `upgrade_agent` with the expected active installation | succeeds (the replacement path) | `...::test_ten_05d_upgrade_with_expected_token_unchanged` | C |
| NT-RW-TEN-07 | 074 (R-P8-01) | default mode, no harness auth, asserted tenant `victim` | refused `MCP_PRINCIPAL_REQUIRED` (documents the MRC-1 state; green at the base) | `test_rw_tenant_mcp_client_asserted_tenant.py::test_ten_07a_default_mode_refuses_asserted_tenant` | C |
| NT-RW-TEN-07 | 074 | `assert_mcp_identity_composition` with `LOCAL_DEV_EXPLICIT` and `fails_closed=True` | refused (`McpIdentityCompositionError`) | `...::test_ten_07b_local_dev_mode_refused_when_fails_closed` | R |
| NT-RW-TEN-07 | 074 | the same with no harness auth composed | refused | `...::test_ten_07c_no_harness_auth_refused_when_fails_closed` | R |
| NT-RW-TEN-07 | 074 | `resolve_mcp_run_agent_identity(..., fails_closed=True)` with an asserted tenant, no principal | refused; with the default `fails_closed=False` the D-4 behaviour is unchanged | `...::test_ten_07d_assertion_branch_refused_when_fails_closed` | R |
| NT-RW-TEN-07 | 074 | a verified principal and a conflicting asserted tenant | refused (existing behaviour) | `...::test_ten_07e_conflicting_assertion_refused` | C |
| NT-RW-TEN-09 | 073 (F-D2-11) | hook registered for tenant A, request of tenant B | the hook does not run | `test_rw_tenant_reindex_hook_scope.py::test_ten_09a_hook_for_a_not_run_for_b` | R |
| NT-RW-TEN-09 | 073 | the same hook, request of tenant A | runs once | `...::test_ten_09b_hook_for_a_runs_for_a` | R |
| NT-RW-TEN-09 | 073 | request without a tenant | only platform-wide hooks run; the scoped hook does not | `...::test_ten_09c_tenantless_request_skips_scoped_hooks` | R |
| NT-RW-TEN-09 | 073 | platform-wide (unscoped) registration | runs for every request (legacy behaviour) | `...::test_ten_09d_platform_wide_hook_unchanged` | C |
| NT-RW-TEN-09 | 073 | a failing scoped hook | isolated; the other hooks still run (PDC-D2-R005) | `...::test_ten_09e_failing_hook_isolated` | C |
| NT-RW-TEN-08 | 072 (EX-D2-06) | strict host, memory capability enabled, no tenant | `MemoryTenantScopeError`; **red only if step S0-6 confirms an unscoped capability at the train head** | `test_rw_tenant_strict_unscoped_memory.py::test_ten_08a_strict_unscoped_memory_refused` | C |
| NT-RW-TEN-08 | 072 | the same with a tenant | composes | `...::test_ten_08b_strict_with_tenant_composes` | C |
| NT-RW-TEN-08 | 072 | a lab host without a tenant | composes unchanged | `...::test_ten_08c_lab_host_unchanged` | C |
| NT-RW-TEN-08 | 072 | memory capabilities disabled | no refusal | `...::test_ten_08d_memory_disabled_no_refusal` | C |
| NT-RW-TEN-08 | 072 | strict host, no tenant, the product defaults of the three flags (all false) | composes as before | `...::test_ten_08e_product_default_flags_not_refused` | C |
| NT-RW-TEN-08 | 072 | strict host, no tenant, only `enable_session_vector_index` or a session plugin id set | composes as before (not on the closed list; Q-TEN-10) | `...::test_ten_08f_session_flags_not_counted` | C |

Notes. By node: R 24, S 2, C 19 (45 in total); of the C nodes, `test_ten_06f` and `test_ten_08a` are conditional rows, and `test_ten_04d` is an R node that may already be green at the base (PT-WF). The three controls 01f, 08e and 08f (DR-TEN-03) pin the closed refusal conditions of Parts A and H. NT-RW-TEN-07 has one control (07a) because the default mode is already fail-closed at `H` (F-58, F-59, F-62); the red-first rows are the new composition path. The NT-ROW-074 text "must be shown red at HEAD" is met only for that new path (Q-TEN-08). A C node is never red-first evidence. Fakes for the sandbox backend, object storage, audit sink and artifact store are in-process doubles and are labelled as stubs in the PR body. Markers: every new file is marked `unit` and none `gate` except, optionally, the reindex file (no threads, no host composition); `test_rw_tenant_install_cas.py` (threads) and `test_rw_tenant_hosted_session_per_tenant.py` (host composition) must not carry `gate` (rule 11).

## 7. Regression scope and test commands

All commands run one process at a time under `D:/Projekty/_wt/_locks/pytest.lock` (`mkdir` must succeed, `rmdir` right after, also on failure; only the creator removes it), logs under `.tmp/session/arch-freeze/rw-tenant-impl/`, one `uv run pytest` per step with several paths, `-p no:cacheprovider`, no broad or diagnostic suite, no Docker, no vendor integration.

The RW-PLAN regression row (section e, F-85) for this ticket: `tests/qualification/tenant_x`; `tests/unit/runtime/sandbox`; `tests/unit/tools/providers`; `tests/unit/agent_distribution`; `tests/qualification/mcp_server_x`; `tests/unit/applications/test_mcp_identity_mode_fail_closed.py`; `tests/unit/rag/governance`. **Contract ratchet check (OD-AF-11, RWD7-02):** the write set has no path under `intergrax/contracts/`, so the wave-packet check (`.tmp/session/arch-freeze/closed-world/typ_measure.py` and `.tmp/session/arch-freeze/rw-plan/work/typ05/measure_typ05.py` over `intergrax/contracts/**`) must report nothing new: no metric above the HEAD baseline (Any 138, object 357, getattr_dynamic 1, hasattr 3, setattr 0, importlib 0, `__import__` 0) and no new string-keyed lookup table (still the 5 of TYP05_REDUCTION). If an amendment ever adds a `contracts/` path, any increase is removed or listed in `typ_ratchet_baseline.v1.json` with this ticket and the reason. It is run once at the ticket head.

Additional regression files named by this plan (the RW-PLAN row does not list them; INPUT records the difference): `tests/qualification/control_planes/ctrl_x`, `tests/qualification/pdc_x/d8_repair`, `pdc_x/p8_repair`, `pdc_x/d2`, `pdc_x/d2_repair`, `tests/unit/rag/ingest/test_ingest_embedding_version_policy.py`, `tests/unit/applications/test_memory_wiring.py`, `test_mem_audit5a_production_provider_admission.py`, `test_mcp_trust_a_principal_binding.py`, `tests/unit/runtime/nexus` (artifact port callers), and the five ci-smoke nodes `test_df5_case_a`, `test_df5_case_b`, `test_df5_case_c`, `test_df5_health_gate_fails_on_real_runtime_when_diagnostics_unavailable`, `test_df5_observability_gate_fails_when_central_diagnostics_not_attached` of `tests/unit/runtime/architecture/test_diag_foundation_5_destructive_proof.py` (DR-TEN-03). The three TENANT-X pin nodes of Part I are inside the regression row and must be green at the ticket head after the re-pin.

| Step | Command | Purpose |
|---|---|---|
| L0 lint | `uv run ruff check --fix` and `uv run ruff format` on exactly the files the ticket stages, then `ruff check` and `ruff format --check`, locked version 0.15.20 | one behaviour-free commit before any change (section 8) |
| S0 red-first | the eight new test files on a detached worktree at the pre-change commit (a worktree under `_wt/intergrax/rw-tenant-s0`, per CLAUDE.md) | every R and S node red for a behavioural reason; C nodes recorded; the log is kept for the auditor |
| S1 per part | the new test file of the part plus the regression directories of that part | after each commit group |
| S2 regression | the RW-PLAN row plus the additional files above, one command | no regression |
| S3 contract check | `typ_measure.py` and `measure_typ05.py` over `intergrax/contracts/**` | no new metric |
| S4 static | `uv run ruff check`, `uv run ruff format --check` and `uv run pyright` on the changed Python files | typing and style |

## 8. Order of work

0. Read-only census at the train head after every predecessor has merged, nothing edited: S0-1 the host label for the sandbox session and the lightest existing environment-wiring fixture; S0-2 the callers of the raw storage and workspace tools, the callers of the freshness assessment and the tests of section 4.2; S0-3 the existing audit or event port for the storage audit (candidates: the runtime event bus, the governance audit recorder); S0-4 who builds `ArtifactAccessContext` for runs other than the producing run; S0-5 what PT-WF, PT-FC1 and PT-WA left in `policy_wiring.py` and `harness_host_runtime.py` (the strict-host tenant guard); S0-6 which memory capabilities are composed without a tenant at the head (EX-D2-06, class C); S0-7 which of the 25 `install_agent` test files install twice into one slot; S0-8 that `wiring_posture.py` exists; S0-9 that the five df5 nodes and the other ci-smoke files are green with the conditions of Parts A and H (a dry computation, then one run in S2); S0-10 the carrier discovery count at the head (Part I). Stop and ask if the census exceeds the files of section 5 and the tests of section 4.2.
1. **L0 mechanical lint commit** (DR-TEN-05), the first commit of the ticket, only for files the ticket stages, with ruff 0.15.20. The findings at the dry-run head: `hosted_session.py` format; `environment_wiring.py` I001 and format; `tools/registry/wiring.py` four findings (I001 twice, UP035, UP045) and format; `online_evaluation_models.py` UP042 on `OnlineEvaluationMode(str, Enum)`, handled with `# noqa: UP042` and not by a `StrEnum` change (that changes `str()` and `format()` of a persisted model); `policy_wiring.py` two findings and format; `memory_wiring.py` two findings and format; `memory_provider_admission.py` format; tests `test_ctrl_x_r2_evaluation_plane.py` one finding, `test_storage_tools.py` four, `test_t10_catalog_tools.py` three. Clean at the dry-run head: `storage/service.py`, `workspace/service.py`, `governed_access.py`, `harness_host_runtime.py`, `admin_service.py`, `mcp_harness_principal.py`, `embedding_version_policy.py`, the three MCP call sites, `ingest_pipeline.py`, `source_freshness.py`. A conditional file that the census adds is linted in its own group (22 of the 42 conditional paths have findings, for example `sqlite_stores.py`). The commit message carries the `Ref:` trailer and no behaviour change; PT-FC2, PT-FC3 and PT-WF reformat the shared hotspots first, so the list is recomputed at the head.
2. Write the eight test files; run S0 red-first at the pre-change commit; keep the log.
3. Part G (reindex hooks and the two forwarders; smallest, independent), then Part F (MCP guard and the three call sites; independent).
4. Part C (artifact read binding).
5. Part A (sandbox session labels), then Part B (tool context, keys, audit, the four storage tests; shares `environment_wiring.py` with Part A).
6. Part D (observation model, CTRL-X test; policy wiring only if S0-5 shows a gap).
7. Part E (install CAS; rebase onto SUPPLY's hunks in `install_agent`; the census-named test files).
8. Part H (memory; rebase onto PT-FC2's hunks), if S0-6 confirms.
9. Part I (carrier re-pin) last, as its own commit group, after the discovery count is final.
10. S2, S3, S4.

## 9. Overlaps and merge order

Lane 7 (RW-PLAN section c, F-93): after PT-WA (merged) and PT-WB (merged); after PT-WF, PT-FC1, T-AF2-RW-SUPPLY, PT-FC3, PT-FC2 and T-AF2-RW-TRUST on the shared paths below; parallel only with T-AF2-RW-TYP; before PT-T01 (the NT-01..16 nodes that have a train fixer must be green when merged; NT-07 and NT-04b have no fixer and are held out of the merge set, RW-PLAN v18).

| Shared file | Other ticket | Rule |
|---|---|---|
| `intergrax/applications/_shared/harness_host_runtime.py` | PT-WA, PT-FC1 | they merge first; the expected edit here is none (Part D) |
| `intergrax/applications/_shared/policy_wiring.py` | PT-WF | PT-WF first; expected edit none |
| `intergrax/runtime/nexus/nexus_loop.py` (conditional here) | PT-WF, PT-FC1 | they merge first; edited only under Q-TEN-02 option 1 |
| `intergrax/applications/_shared/environment_wiring.py` | PT-FC3 | PT-FC3 first; Parts A and B are small hunks around `:347`, `:439-448`, `:473` |
| `intergrax/applications/_shared/memory_wiring.py`, `memory_provider_admission.py`, the two memory tests | PT-FC2 | PT-FC2 first; Part H adds one function and one call |
| `intergrax/agent_distribution/admin_service.py`, `online_evaluation_registry.py` (conditional here) | T-AF2-RW-SUPPLY | SUPPLY first; Part E rebases and adds no admission step; the registry is not edited by default |
| `tests/unit/rag/governance/test_rw_tenant_reindex_hook_scope.py`, `test_embedding_version_policy.py`, `tests/unit/rag/ingest/test_ingest_embedding_version_policy.py` | T-AF2-RW-TRUST | TRUST first |
| `tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py` (request GAP-10) | T-AF2-RW-SUPPLY, PT-WF (any carrier they add) | one re-pin at the TENANT head, after both |
| `intergrax/platform/enterprise_audit/impact_map.py` and its data file (GAP-12) | PT-WD (lane 3, no precedence edge to this ticket) | not edited here; Q-TEN-14 |

## 10. ACT-N11 / IIAC consequence

Every product path above under `intergrax/`, `agents/`, `applications/` is a product path by `PRODUCT_PATH_PREFIXES` (`scripts/governance/iiac_common.py`); the 19 firm product paths (and the conditional ones if they are edited) break the IIAC activation recorded for PDC-X P8 until the single RB-2 re-pin (DP-58, `PT-Q03`). This ticket does not perform the re-pin and must not merge outside the wave.

## 11. What closes each finding

Nothing is closed by this plan. A finding is closed only by its negative test green at the exact wave SHA, the regression scope green, the contract check clean and an independent audit in a separate session (node `PT-Q02`). EX-D4-04, EX-D5-05, EX-L-09 (G-12) and R-P8-01 stay partly open if the conditional paths GAP-02, GAP-04 and GAP-05 stay out of the ticket, and R-P8-01 stays partly open until the facades of GAP-11 pass `fails_closed`; the ticket then records which half remains. R-P8-01 closes only after the B-1 check X-A8-022 and an independent audit (F-96).

## 12. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R-1 | Keying raw storage objects by tenant makes objects written before the change unreachable by the tools | no migration in this ticket; Q-TEN-04; the PR body names the key contract change |
| R-2 | Tools without a tenant or sink refuse; tests and lab hosts that built a bare `ToolWiringContext` go red | the four storage-tool tests are firm in A1 (Part B); census S0-2; any other test that turns red is a stop and an amendment request |
| R-3 | The event bus is created after the tool wiring, so the audit adapter needs a reorder or a lazy binding | S0-3; Q-TEN-05; a stop if neither fits |
| R-4 | A run binding on governed reads breaks a legitimate cross-run read | S0-4 census; the typed grant is the only escape; the retention sweeper uses the system actor binding (F-34) |
| R-5 | The base observation field makes persisted or caller code diverge from the registry subclass (an optional-string override of the field) | step 0 reads the subclass; if pyright rejects the override the stop is a question (Q-TEN-02), not an edit of a SUPPLY path |
| R-6 | The per-slot lock is process-local; two processes can still race an empty slot | recorded limit; GAP-02; the NT test is in process only |
| R-7 | The MCP guard is not reached by a real composition until the callers are amended | recorded; R-P8-01 stays partly open (GAP-06) |
| R-8 | Scoped reindex hooks never fire from production ingest until the callers forward the tenant | recorded; fail closed by design (GAP-07) |
| R-9 | Merge conflicts on `environment_wiring.py` (600 lines), `admin_service.py` (2533 lines) and the two memory files | order in section 9; small hunks |
| R-10 | The strict memory refusal blocks a strict host that has no tenant by design | closed flag list (Part H); controls 08e, 08f; Q-TEN-10; class C; NOT_REPRODUCED goes back to the Operator |
| R-11 | The four new tenant carriers turn the TENANT-X pin red (DR-TEN-02) | Part I and GAP-10; one re-pin at the head; the packet shows it as its own group |
| R-12 | Lint findings block the first commit (DR-TEN-05) | commit group L0; ruff 0.15.20; `noqa` for UP042 |
| R-13 | A strict host composes with no tenant and no sandbox host (the df5 shape) and Part A refuses it | the closed condition of Part A; control 01f; the five df5 nodes in the regression list |
| R-14 | PDC P6 impact coverage maps `hosted_session.py` and `nexus_loop.py` to no check | not this ticket; GAP-12 and Q-TEN-14 |

## 13. Rollback

Revert the commit group of the part; the groups are independent except Part B depends on Part A for the shared `environment_wiring.py` hunk and the normalized tenant, and Part D's policy-wiring hunk (if any) depends on S0-5. Objects written under the tenant namespace by tests of this ticket stay in the fakes. The IIAC rebind must be redone after any revert. No setting or production configuration change is made.

## 14. Open questions with draft defaults

| ID | Who | Question | Draft default |
|---|---|---|---|
| Q-TEN-01 | orchestrator | Amend A1 with the paths of the first draft's gaps | **ruled in RW-PLAN v14** (A1 v18: 19 / 32 / 42); this revision adopts it |
| Q-TEN-02 | Operator / orchestrator | EX-D5-05: a required `tenant_id` on the base observation (Option 1, breaks three callers, the registry subclass and persisted records) or an optional, non-blank-validated field plus the strict-host refusal (Option 2) | Option 2; NT-ROW-068 "requires" is then met in its strict form; the CTRL-X pin is inverted in both options (GAP-03) |
| Q-TEN-03 | orchestrator | EX-L-09 G-12: refuse any install into an occupied slot (default, inside A1), or add an expected-revision token to the request (GAP-01), or a cross-process "expect absent" compare (GAP-02) | refuse any occupied-slot install; in-process lock; the rest recorded as limits |
| Q-TEN-04 | Operator | EX-D8-03 / EX-L-02: tenant on the wiring context with a key prefix (plan of record), or retire the raw `storage.*` tools; legacy keys unreachable | tenant prefix scheme; no migration |
| Q-TEN-05 | orchestrator | which existing audit or event port carries the storage audit event | decided by S0-3; a new event type is a gap (GAP-08) and a stop |
| Q-TEN-06 | orchestrator | do the shadow-workspace file tools need audit and keys | no; only the two tools that touch object storage |
| Q-TEN-07 | Operator | EX-D8-04: bind the principal as well as the run | run only; principal as a conditional node |
| Q-TEN-08 | orchestrator / Operator | R-P8-01: keep the D-4 decision (`LOCAL_DEV_EXPLICIT` admits an asserted identity for in-process and loopback callers) and add an opt-in `fails_closed` guard, or invert D-4 and its pinned tests | keep D-4; add the guard; amend the callers (GAP-06); the register row closes only after X-A8-022 and an audit |
| Q-TEN-09 | orchestrator | do platform-wide (unscoped) reindex hooks stay legal | yes (compatibility); a request without a tenant runs only those |
| Q-TEN-10 | Operator | EX-D2-06: STRICT forbids unscoped memory (DS-EXC recommended option); do session storage, `enable_session_vector_index` and the plugin ids count | refuse composition when the host fails closed, one of `enable_user_memory`, `enable_org_memory`, `enable_long_term_memory` is true and the tenant is blank; the session items do not count; class C |
| Q-TEN-11 | orchestrator | tools called with a context that has no tenant | refuse in every host (lab hosts give an explicit lab tenant) |
| Q-TEN-12 | orchestrator | the hosted sandbox session of a non-strict host without a tenant | keep the two legacy labels as named lab constants; a strict host refuses only when its profile resolves a sandbox host (DR-TEN-03) |
| Q-TEN-13 | Operator (packet) | GAP-10: edit the pinned TENANT-X carrier classification register and re-pin its total and digest once at the TENANT head | yes, as its own commit group of this ticket, as the dry run proposes; no design around the pin |
| Q-TEN-14 | orchestrator | GAP-12: PT-WD covers `intergrax/runtime/sandbox` and `nexus_loop.py` in the impact map, or a precedence edge PT-WD to this ticket; register the already-red P6 impact-coverage test | not TENANT scope; the ask goes to PT-WD and the register PR |

## 15. Authorization requests

| ID | Request | Needed for |
|---|---|---|
| TEN-A1 | the wave-packet authorization of OD-AF-22 for the product paths of this ticket | any product edit |
| TEN-A2 | permission to create one detached worktree under `D:/Projekty/_wt/intergrax/` for the red-first run (one per session, removed afterwards) | S0 |
| TEN-A3 | pytest runs under the shared lock, one process at a time, the files named in section 7 | S0-S2 |
| TEN-A4 | an amendment of RW-PLAN A1 for the paths and tests of section 4 that the orchestrator accepts | Q-TEN-01 |
| TEN-A6 | the Operator view of the carrier re-pin commit group (Q-TEN-13) | Part I |
| TEN-A5 | the Operator rulings Q-TEN-02, Q-TEN-04, Q-TEN-07, Q-TEN-08 (part), Q-TEN-10 | the semantic parts B, C, D, F and H |

## 16. Effort

| Part | Size | Note |
|---|---|---|
| A sandbox session labels | S | one call site, one validator |
| B tool context, keys, audit | L | audit port and construction order (S0-3) |
| C artifact read binding | M | enum, grant, two methods |
| D observation model | S | the field and a validator; callers are GAP-04 |
| E install CAS | M | refusal and per-slot lock; rebase on SUPPLY |
| F MCP guard | S | one function and one keyword |
| G reindex hooks | S | one module |
| H strict memory | S | depends on the S0-6 measurement |
| I carrier re-pin | S | one data file; the count is recomputed at the head |
| L0 lint | S | mechanical; recomputed at the head |
