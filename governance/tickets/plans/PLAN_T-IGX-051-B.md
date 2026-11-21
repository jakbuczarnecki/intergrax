# PLAN T-IGX-051-B — T-AF2-RW-SUPPLY: supply chain and security (attestation, revocation, trust hook order, digests, fail-open set, strict validator, secrets, evidence sink, retention)

Draft, 2026-11-17, **revision 2** (same day) after the step-0 dry run `dry-run/DRY_RUN_T-IGX-051-B.md` (label of another session: DRY_RUN_FINDINGS, 10 findings; every finding is answered in section 17). Nothing was run and nothing was changed outside `.tmp`. No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this plan; labels such as "audit", "DS-EXC", "PDC register" are quoted with their source. Evidence notation `path@H:line`; `H` = `05699b633cdb1318c215f6529e449420efc1382c` (`origin/arch-freeze/rw-train`, the merge of PT-WA, PR #642). The dry run re-checked the write set at the train head `881428fb231f34215a67364aa5788a4f2f986e83`: `git diff --name-only 05699b633..881428fb2` over the 22 product paths and `intergrax/core/plugins/` is empty, so every `@H` citation still holds; RW-PLAN is now v18 with the same scope for this ticket (20 firm product, 31 firm, 2 conditional). Every product path of the write set is byte-identical at `H` and at `origin/development` `2727a2c2364a97492202b931375772a400279a9b` (the only product delta between the two refs is the 45 PT-WA files, none of them a path cited here; checked with `git diff --name-only`). Defects are described by ID and title only. `FACT_CHECK.md` has the claim-by-claim table (rows F-01..F-88 are code facts whose line numbers a script resolved at `H`; F-89..F-127 are scope, register, dry-run and command facts; the items still UNVERIFIED or only partly answered are F-110..F-113).

Source of scope: `RW_PLAN.md` v13 (v18 is current, same scope for this ticket) Appendix A1 block `T-AF2-RW-SUPPLY (T-IGX-051-B)` (20 product firm, 31 firm in total, 2 conditional; section a row T14), its negative-test rows NT-ROW-075..096 and NT-ROW-159 (section d), its regression row (section e) and its lane and overlaps (sections b, c). Plan of record section E (DS-EXC dispositions) and section G (the RW train row) give the intent. The base write set (section 5) equals A1. Where the code shows that A1 is too narrow, section 4 lists the missing paths and section 5b lists the **proposed additions with class, reason and condition**; they are not in scope until the RW-PLAN is amended and the orchestrator and, where named, the Operator accept them. Revision 2 changed the design so that existing callers keep working (service-level default, section 2 rule 1), which cut the additions to 3 firm and 8 conditional paths and one proposed split ticket (section 5b, Q-SUP-11).

## 1. Purpose and findings

One product-path ticket of the RW train (lane 4, after PT-FC2 and PT-WB on the `_shared` wiring files; DP-11 single integrated wave). Binding intent (plan of record section E, quoted): attestation verified; revocation source and trust coordinator injected; trust hook before persist; digest check; fail-open set closed; strict validator refuses weak or disabled settings; secret-gate evasions closed; evidence-sink consumer; retention plus AUDIT_EVIDENCE producer.

| Finding | Source label | What this ticket does | What it cannot close alone |
|---|---|---|---|
| EX-D1-01 | DS-EXC, HIGH (F-D1-01) | Part A: when an attestation enforcement is composed (always, in the production factory), the install and upgrade path verifies the attestation behind a trust record; a caller-asserted record without signature material is refused when the policy requires signed packages. Direct constructions of the service keep today's behaviour (explicit 'not composed' state) | the production composition that supplies the verifier and the pinned publisher keys (Part B); the install-request shape if a new input is needed (A1 gap G-02) |
| EX-D1-02 | DS-EXC | Part B: the production admin service factory takes the trust coordinator, revocation-state source, policy and time sources as required inputs; the pipeline resolves its trust hook from the composed capability runtime (a composition without one keeps an explicit 'not composed' pass-through) | the GCA strict path may compose no admin service, so nothing there can carry a trust composition (S0-4, Q-SUP-02, expected-open with owner orchestrator) |
| EX-D1-04 | DS-EXC (Operator decision recorded in the plan of record: a revoked revision without a trusted target stops serving) | Part B: `BLOCKED_NO_SAFE_TARGET` becomes a serving stop, not a returned report | the serving read side if it lives outside A1 (step 0, G-04) |
| PDC-D1-R017 | PARTIAL row | Part B: a production consumer of `lifecycle_event_sink` writes lifecycle events to the evidence plane | tenant derivation for the events (step 0) |
| A-01 | audit finding | Part C: the trust hook runs before `persist_candidate_revision`; a refused revision is not persisted, validated, registered or prepared | none |
| EX-D6-03 | DS-EXC | Part D: an installed-file digest source and check; a mismatch is refused at admission | the packaging convention of the attested digest (Q-SUP-04); no host composes the resolver yet |
| FRZ-SEC-06 | FRZ C2 (basis C) | Part D: the security-defense loader refuses to run without provenance when told it is on a strict or product root; red only if step 0 confirms the path | the caller `environment_wiring.py` is a PT-FC3 file (G-05) |
| EX-D6-01, F-D6-02 | DS-EXC, audit | Part E: audit-trail entries are hash-chained and verified; a modified entry is detected | tail truncation without an external anchor (documented limit, Q-SUP-03); the `verify_replication` report of `multi_region_audit_trail.py` does not read stored entries back and is not in A1 (G-13) |
| F-D6-03 / PDC-D6-R007 (part) | debt, LOW | Part E: the listing never silently drops entries; secret-shaped values are redacted | none |
| F-D3-12 / PDC-D3-R018 | A-12 set, LOW | Part F: a malformed quota value denies the call | the carrier count of T-IGX-041-I is a record only |
| F-D6-09 | A-12 set, LOW | Part F: the poisoning default is most restrictive on both paths | the divergent branch is in `plan_context_invocation.py`, not in A1 (G-06); split-off proposal T-IGX-051-I (section 5b, Q-SUP-11) |
| F-D6-11 | A-12 set, LOW | Part F: a tenant-less resource is not open to an arbitrary scoped tenant; `RequestIdentity.tenant_id` is normalised | the strip of `RequestIdentity.tenant_id` stays here; the tenant-less-resource rule is an Operator decision (Q-SUP-05) with callers and a pinned test outside A1: split-off proposal T-IGX-051-I |
| F-D5-12 | A-12 set, LOW (class C) | Part F: the registry lock is released only by its owner; the id-existence oracle is closed if step 0 reproduces it | not reproduced across processes per the PDC register |
| F-D8-14 | A-12 set, LOW | Part F: a legal hold cannot be overwritten or lost | holder-scoped release needs the governed callers (G-08) |
| F-D6-10 / PDC-D6-R024 | debt, LOW | Part G: the strict-config validator refuses weak or disabled settings | no host calls it today (F-64); composition belongs to EX-D6-02 below |
| EX-D6-02, F-D6-04 | DS-EXC (REQUALIFICATION first, PT-Q12) | Part G: step 0 requalifies; the fix is a composition only if no control is composed at `H` | a host factory edit outside A1 (G-09) |
| F-D6-14 / PDC-D6-R007 | debt, INFO | Part H: a static secret-literal script gate that flags the known evasions | the evasions live in `raw_secret_gate.py`, not in A1 (G-10) |
| PDC-D6-R010 | PARTIAL row | Part H: an isolation-strength gate that refuses a present but weak configuration | the definition of "weak" (Q-SUP-07) |
| EX-D8-01 | DS-EXC, HIGH | Part I: an AUDIT_EVIDENCE producer and a retention composition gate (fail closed in production); or the explicit responsibility statement if the Operator chooses it | the host composition edit outside A1 (G-11); the statement needs docs, which A1 does not have |

The mapping of EX-D6-02 and F-D6-04 follows the plan of record: requalify first, fix only if confirmed. A negative test for them is class C.

Related items that the RW-PLAN row names but this ticket does not own: FRZ-PLG-05 (owner PT-FC1; its plugin half, an installed-file digest mismatch without re-hash, is NT-RW-SUP-06 = NT-PLG-05-1 here), FRZ-EXE-02 (owner PT-FC1; listed in RW-PLAN with RW-SUPPLY and PT-WA as dependencies; this plan edits nothing for it), A4-040 (owner PT-WB; the scope ruling is open, Q-RW-16 b; section 9 and Q-SUP-10).

## 2. Design rules applied to every part

1. **Fail closed at the composition, explicit 'not composed' below it.** The production composition (`build_production_agent_platform_admin_service`, the strict startup) takes every new trust, revocation, evidence or retention input as required and refuses a missing or lab-marked one at composition, never at first use (`intergrax-env-vendor-qualification`). A class that is constructed directly (the admin service, `PluginProvenanceAdmission`, the pipeline context) keeps its current behaviour when the new input is absent; that state is explicit, named in the class docstring and recorded in the evidence, and it is the lab/unit/reference construction, not a production state. This keeps the roughly 26 existing callers of `install_agent`/`upgrade_agent` and the launcher-driving tests working without edit (dry-run findings B-01, B-02, B-03). A lab value for a composition that must still be built (`lab_package_trust_composition()`) is a named function carrying `is_lab=True`, which the strict startup refuses.
2. **No second trust authority.** Signature verification reuses `AgentPackageAttestationVerifier` and `AgentPackageTrustCoordinator` (`intergrax/agent_distribution/package_attestation.py`, `package_trust.py`); `PluginProvenanceAdmission` already does the same for plugins (F-33). No new cryptography.
3. **Contract surfaces get no new `Any`, `object` or reflection dispatch** (OD-AF-11). The one contract edit, `intergrax/contracts/agent_run.py`, is typed with existing types only; the wave-packet contract check must report nothing new (section 7).
4. **No secrets in code, logs, argv or test output.** Tests use generated throwaway key material and fixture literals built from parts at run time; a finding or a log never echoes a candidate secret value (`raw_secret_gate` convention, `raw_secret_gate.py` module docstring).
5. **No Docker, vendor integration or production configuration change** (C10/C15 freeze, `intergrax-env-vendor-qualification`). Real publisher keys, a real revocation feed or an evidence backend are PROD-Q items; this ticket composes ports and proves them with in-process fixtures, labelled as such.
6. **Environment values enter through environment variables and typed configuration** only; no literal key, endpoint or path in product code.
7. **One change, one signature.** A changed function has one signature; callers inside the write set are updated in the same commit group. Callers outside the write set are a step-0 census item and a question (section 4), not a silent edit.
8. **Existing tests that pin the old behaviour** are named in section 4.2 and, for the ones that must change, in section 5b. A test outside the write set is never edited by the implementer without the amendment; an expected-red is recorded with an owner (section 17).
9. **Pre-existing pins are kept.** The textual order `_authorize_admission(` before `persist_candidate_revision(` and `_authorize_activation(` before `.commit_activation(` in the deploy pipeline (`build_remediation.py`, `activation_remediation.py`), a single `uvicorn.run` site in the serving command, no `cast(` or `type: ignore` in `intergrax/tools/providers/rag/service.py`, no class named `*Registry` or `*Scheduler` in `retention.py`, `FileOnlineEvaluationRegistry` kept, and (if `discovery.py` is edited) no class ending in `Plugin` that exposes `execute`.

## 3. Parts

Each part is a separate commit group. The order is in section 8.

### Part A — attestation verified on install and upgrade (EX-D1-01)

State at `H`. Both `install_agent` and `upgrade_agent` admit a package with `assert_install_admission` (`admin_service.py@H:654`, `:834`). That method validates the caller-supplied `AgentInstallationTrustRecord`, then revocation and freshness; it has no signature step (`package_trust.py@H:544`). The coordinator's attestation verifier is optional and the admin service builds a coordinator without one (`admin_service.py@H:327`, `package_trust.py@H:86`). The only place signature evidence is re-validated is the `evaluate` helper (`package_trust.py@H:611`). `AgentInstallationTrustRecord` carries evidence references and a status but no signature material (`trust.py@H:131`); the attestation evidence ref carries id, key, algorithm and digest, not the signature (`package_attestation.py@H:257`).

Design.
- `trust.py`: `AgentInstallationTrustRecord` gains one field, a tuple of a new frozen model `AgentTrustAttestationMaterial` (`attestation_id`, `key_id`, `algorithm`, `publisher_id`, `signature_b64`; all `str`/enum, `extra="forbid"`, stripped, non-empty). Existing records stay constructible with an empty tuple. A new frozen dataclass `PackageAttestationEnforcement` (verifier) lives in the same file. No `Any`.
- `admin_service.py`: a new keyword `package_attestation_enforcement: PackageAttestationEnforcement | None = None`. `None` is the explicit 'not composed' state: no new step, today's behaviour, so the direct constructions of the service in existing tests are unchanged. When it is composed, a private step `_assert_trust_record_attested(trust_record, identity, policy)` runs immediately after `assert_install_admission` in `install_agent` and `upgrade_agent` (nowhere earlier than the authorization boundary): for each material item it calls `verifier.verify(AgentPackageAttestationVerificationRequest(...))` for the exact digest-pinned identity; the result must be `verified`, the publisher must equal `trust_record.publisher_identity_ref`, and the item's reference must appear in `trust_evidence_refs`. When `policy.forbid_unsigned_or_unqualified` is true (the policy default, `trust.py@H:231`) and no verified signature exists, the call raises `AgentPackageTrustError` with the existing reason codes (`UNSIGNED_FORBIDDEN`, `MALFORMED_EVIDENCE`). A policy with `forbid_unsigned_or_unqualified=False` is the explicit lab policy, as in `PluginProvenanceAdmission` (`admission.py@H:228` class docstring).
- The production factory (Part B) always composes the enforcement, so the **composed install/upgrade path** of EX-D1-01 is closed; the test of NT-RW-SUP-01 drives that path through the factory.
- `emergency_revocation_response.py` and `package_trust.py` stay untouched (G-02 records why).

Not done here: choosing the production publisher key source or the revocation feed (Part B composes the ports; keys and feeds are PROD-Q).

### Part B — trust composition, revocation source, evidence sink, both launchers, no-target stop (EX-D1-02, EX-D1-04, PDC-D1-R017)

State at `H`. `build_production_agent_platform_admin_service` passes no coordinator, revocation, policy, time source or event sink (`production_agent_capability_runtime.py@H:181`, `:204`); `ProductionAgentPlatformAdminConfig` has none of these fields (`:104`). The revocation source is an optional keyword with an empty fallback (`admin_service.py@H:285`, `:329`). The pipeline hook is optional and the comment says the launchers do not pass it (`deploy_activate_pipeline.py@H:93`). Both lifecycle launchers build the pipeline context without it (`production_lifecycle.py@H:215`, `reference_production_lifecycle.py@H:196`, both outside A1; the first is an A1 path of PT-WF). No code under `intergrax`, `applications` or `agents` constructs `ProductionAgentPlatformAdminConfig` (git grep); `testing_support/canonical_agent_lifecycle_composition.py` and `tests/unit/applications/test_ac4_phase9_production_composition_e2e.py` do, so a required field breaks both (section 4.2). The GCA strict startup takes `application_composition` as an optional `None` (`single_node_profile_v1_strict_startup.py@H:44`), and the strict serving command passes none (`package_b_strict_production_serving_command.py@H:129`). On a revoked serving revision with no trusted target the emergency response returns `BLOCKED_NO_SAFE_TARGET` and nothing stops serving (`emergency_revocation_response.py@H:300`).

Design.
- `production_agent_capability_runtime.py`: a frozen dataclass `ProductionPackageTrustComposition` (verifier, revocation-state source, policy source, time source, lifecycle event sink, `is_lab: bool = False`) and a required field `package_trust` of `ProductionAgentPlatformAdminConfig` with no default. `build_production_agent_platform_admin_service` builds the coordinator with the verifier, composes the attestation enforcement of Part A and passes all inputs into `AgentPlatformAdminService`. A missing `package_trust` is a `TypeError` at construction; a `None` member raises a `ValueError` naming the missing input. `lab_package_trust_composition()` returns a value with `is_lab=True` (lab policy, empty revocation state, system clock, a no-op sink) for the three callers that build the config in tests and support code (section 5b, items P1-P3). A small private adapter in this file turns `AgentDistributionEvent` tuples into security audit entries through `EvidencePlaneSecurityAuditTrail.append` (Part E file); the tenant comes from the environment tenant resolver already in `admin_config`. The consumer is the answer to PDC-D1-R017; it decides and stores nothing of its own.
- `deploy_activate_pipeline.py`: `run_deploy_and_activate` takes the trust admission from `ctx.revision_trust_admission` if set, otherwise from `ctx.composition.agent_capability_runtime.admin_service.assert_revision_trust_admissible` when a capability runtime is composed. When neither exists (a reference composition without a capability runtime, `production_process_composition.py@881428fb2:45`, optional) the step is an explicit pass-through recorded as `trust_admission_not_composed`, as today. Both launchers use this function, so neither needs an edit (`production_lifecycle.py` is a PT-WF path), and the roughly 23 launcher-driving tests keep working (dry-run finding B-02). The strict production composition cannot reach the pass-through, because the strict startup refuses a lab or missing `package_trust` on a given composition (next bullet).
- `admin_service.py`: `respond_to_emergency_revocation` turns `BLOCKED_NO_SAFE_TARGET` for a revoked serving revision into a serving stop through the existing activation service (the mechanism, for example the post-cutover failure path `handle_post_cutover_failure`, `admin_service.py@H:1653`, is a step-0 read; if no stop primitive exists in the files of A1 the part stops and asks, G-04). The response report is still returned.
- `single_node_profile_v1_strict_startup.py`: the strict startup validates a given `application_composition`: its `admin_config.package_trust` must be present and `is_lab` false, else `ValueError` before the session opens. It does **not** refuse `application_composition=None` and does not require new operator environment variables: the shipped entrypoints, the docker-compose local qualification and the RUNBOOK would need a configuration change, which is production configuration and PT-WB Part G territory (freeze, rule 5). Whether the GCA strict path should compose an admin service at all is S0-4 and Q-SUP-02.
- `package_b_strict_production_serving_command.py`: **no edit** in the draft default (revision 1 planned one call; the dry run found that the PROD-Q serving-wiring gate patches the command's preamble, finding B-06, and the file is the shared file with PT-WB Part G). The firm A1 path stays listed because a later amendment may need it; unedited firm paths are allowed. The bind-host default and the `uvicorn.run` block (`:153-162`, F-29) are not touched in any case; this ticket takes no position on A4-040.

Open point that decides the size of this part: whether the GCA strict path composes an admin service at all today (step 0, S0-4). The strict startup takes `application_composition` as an optional `None` (`single_node_profile_v1_strict_startup.py@H:44`) and the command passes none (F-28), so by code reading the shipped GCA strict path has no capability runtime and no admin service. If that is confirmed, EX-D1-02 is closed for every composition that exists and stays **expected-open for the GCA strict path**; closing it needs a standalone revision-trust admission, a new product module that A1 does not contain (G-03, Q-SUP-02, owner: orchestrator).

### Part C — trust hook before persist (A-01)

State at `H`. `persist_candidate_revision` runs first (`deploy_activate_pipeline.py@H:174`), then `mark_validated`, the projection input registration and `prepare_candidate` (`:178-183`), and only then the trust hook (`:204`).

Design. In `run_deploy_and_activate` the trust admission (resolved as in Part B, including the explicit pass-through) moves to immediately after the control-plane admission authorization (`_authorize_admission`) and before `persist_candidate_revision`; the activation-time authorization stays where it is. The textual order pinned by `build_remediation.py` and `activation_remediation.py` (rule 9) is kept. The revision passed to the hook is the candidate value that the function builds before persisting. A refusal raises before any store write: nothing is persisted, validated, registered or prepared. A second check after `prepare_candidate` is not kept (one check, one place; no dual path).

### Part D — installed-file digest and security-defense provenance (EX-D6-03, FRZ-SEC-06)

State at `H`. `PluginProvenanceAdmission` verifies the signed attestation and the installed distribution version, and states that it does not re-hash the installed files (`admission.py@H:243`, `:314`). The attestation carries distribution, version, package digest, publisher, attestation id, key id and signature (`admission.py@H:190`).

Design (EX-D6-03).
- `admission.py` (a `core/plugins/*` path): `PluginProvenanceAdmission` gains `installed_digest_resolver: Callable[[str], str | None] | None = None`; `None` is the explicit 'not composed' state, today's behaviour, so the existing positive admission tests (`pdc_x/d6`, `pdc_x/d6_repair`, which use a stub version resolver) stay green (dry-run finding B-03). A public function `installed_tree_digest(distribution)` in `admission.py` is the resolver a composition passes: a deterministic digest from `importlib.metadata` (sorted relative path and per-file sha256 of every file of the distribution except the metadata files that change at install time and compiled caches; every file is re-hashed from disk and compared with its RECORD hash, then the tree digest is compared with the attested `package_digest`). When a resolver is set, a mismatch, a missing file or an unreadable file is `PLUGIN_PROVENANCE_INVALID`, and the check is mandatory for a strict policy (`forbid_unsigned_or_unqualified` true). No product code constructs `PluginProvenanceAdmission` today (git grep over `intergrax`, `applications`, `agents`, `testing_support`, `platform_proofs`; only the two d6 tests do), so the composition that passes `installed_tree_digest` does not exist yet; the plan states that in the PR body and EX-D6-03 stays partly open until a host composes it (owner: orchestrator, with the FRZ-SEC-06 caller G-05).
- This fixes the attested `package_digest` of a plugin distribution as the installed-tree digest. That is a packaging convention for publishers, so it is question Q-SUP-04 with the draft default above.
- `discovery.py` is touched only if step 0 shows the resolver cannot be reached through `PluginProvenanceAdmission`.

Design (FRZ-SEC-06, basis C; red only if step 0 confirms it).
- `defense_plugin_loader.py`: `SecurityDefenseAdmissionPolicy` gains `require_provenance: bool = False`. When discovery is on, `require_provenance` is true and `provenance` is `None`, the loader raises `SecurityConfigurationError` with `PLUGIN_TRUST_POLICY_MISSING`, before any entry point is loaded; `register_plugins_with_report` already has the same rule for ordinary plugin groups (`discovery.py@H:342`, F-39), so this aligns the two loaders.
- `core/security_bootstrap.py` (conditional): `bootstrap_security_providers` passes the policy through; it is needed only if the flag cannot ride on `SecurityDefenseAdmissionPolicy` alone. The only product caller passes just the discovery flag (`environment_wiring.py@H:198`, a PT-FC3 file): a STRICT root must pass `require_provenance=True` there (G-05). Without that edit the check exists but no root asks for it, and the test of NT-SEC-06-1 stays class C.

### Part E — audit-trail tamper evidence, listing, redaction (EX-D6-01, F-D6-02, F-D6-03)

State at `H`. `SecurityAuditEntry` has no digest, chain or signature (`security_audit_trail.py@H:40`). The in-memory trail stores metadata without any check (`:74`). The evidence-plane trail validates metadata keys against a secret-safe policy with no value patterns (`:89`, `:183`) and lists at most 1000 entries (`:232`). The listing returns the oldest 1000 only (PDC register F-D6-03). The multi-region wrapper's `verify_replication` compares in-process id lists and, by its own docstring, never reads the stored entries back (`multi_region_audit_trail.py@05699b633:7`, outside A1).

Design.
- Chain. Each entry carries `prev_digest` and `entry_digest` (sha256 over the canonical JSON of the entry fields and `prev_digest`); a tenant's stream starts from a fixed genesis value. Both trails expose `verify_chain(tenant_id)` that returns a typed report (verified count, first broken index and reason) and raises nothing for a clean chain. A per-trail lock serializes the read-last-then-append step. `SecurityAuditPayloadV1` gains the two fields with empty-string defaults so older stored events still parse (a version bump of the payload schema is the alternative; step 0 reads the registry rules, risk R-5).
- Honest limit. A hash chain detects a modified or a removed interior entry; it does not detect truncation of the tail without an external anchor (a signed or separately stored head). The plan builds the chain and the verification, exposes the head digest, and records the limit. The DS-EXC alternative (`NOT_CLAIMED` with reviewed SEC/AUD wording, plan of record section E) is the Operator's choice (Q-SUP-03, draft default: build).
- Listing. The port cannot page: `list_for_task` has only `limit: int = 1000` and no cursor or offset (`persistence_port.py@881428fb2:51-57`, dry-run finding B-05). `list_entries` therefore raises a typed `AuditTrailListingTruncated` **only when the number of returned events equals the limit** (the list may be truncated), and returns normally below it; `check_immutable_security_audit_trail.py` (`ENT-A-SEC-01`, fast-mandatory in `ent-audit-fast`, 2 entries) stays green. A cursor in `intergrax/contracts/execution_evidence/persistence_port.py` is a contract-surface edit and a new path; it is not proposed (it would also fall under the OD-AF-11 ratchet) and is listed in section 5b only as an option of the orchestrator.
- Single append site. The chain keeps exactly one `self._evidence.append` call in the evidence-plane trail (`security_audit_trail.py@881428fb2:222`), because COMPAT-X pins that site (dry-run finding B-07); the digest is computed before it, in the same function.
- Redaction. `secret_safety.py` gains a pure function that replaces secret-shaped string values (credential formats and high-entropy secret-named values) with a fixed marker, and the audit metadata policy uses it before validation and storage. In-memory and evidence-plane trails behave the same. Keys that the policy already rejects are still rejected.
- `verify_replication` and the multi-region wrapper are not edited (G-13); the chain verification is a method of the two trails, so the "immutable" flag of that report stays an id-list statement and its docstring stays accurate.
- `scripts/maintenance/check_immutable_security_audit_trail.py` (existing, regression scope) must keep passing; it is not edited here.

### Part F — fail-open set (F-D3-12, F-D6-09, F-D6-11, F-D5-12, F-D8-14)

F-D3-12 (`quota.py`). A value that is not a non-negative integer maps to 0, which the module treats as disabled (`quota.py@H:23-28`, `:42`). Design: unset or the literal `0` keeps the documented "disabled" meaning; any other unparsable or negative value raises a new `LLMQuotaConfigurationError` that subclasses `LLMQuotaExceeded`, so every existing handler maps it to a denial. The adapter call sites (`base_llm_adapter.py@H:292`) belong to RW-RUNTIME and are not touched.

F-D6-09 (`service.py`). The RAG tool path already resolves the toggle through `retrieval_poisoning_defense_enabled_or_default` and is ON when the profile is absent (`service.py@H:212`). The plan-context path keeps the filter OFF when the profile is absent and `production_mode` is false (`plan_context_invocation.py@H:151`, outside A1, G-06), and an existing test pins that (`test_d6_amendment1_gap_closures.py@H:346`, G-12). The only edit that closes the finding is in `plan_context_invocation.py`. In `service.py` the plan adds nothing beyond a docstring that names the single resolver (the docstring must contain neither `cast(` nor `type: ignore`, rule 9). **NT-RW-SUP-11 cannot go green inside the A1 write set** (dry-run finding B-04): it needs `plan_context_invocation.py` and the edit of `test_d6_amendment1_gap_closures.py`. Both are in the split-off proposal T-IGX-051-I (section 5b); if the split is not accepted they are cond additions of this ticket (C1, C2). Until one of them is decided **NT-RW-SUP-11 is held out of the merge set** (section 6, 'Held-out tests'); F-D6-09 stays open with owner Operator (the divergence stays, ADR text is the alternative the audit named).

F-D6-11 (`tenant_scope.py`, `agent_run.py`). A scoped request against a tenant-less resource is valid even with `allow_unscoped=False` (`tenant_scope.py@H:31`); the existing matrix pins it (`test_tenant_scope.py@H:36`). `RequestIdentity.tenant_id` is validated non-empty but not stripped (`agent_run.py@H:34-38`), while the scope code strips. Design: `RequestIdentity` strips the tenant on validation (NT-RW-SUP-12b; stays in this ticket; `agent_run.py` is the contract edit). The semantic part, `tenant_scope_is_valid` returning `allow_unscoped` for a scoped request and a tenant-less resource (NT-RW-SUP-12a), **inverts the matrix row of `test_tenant_scope.py`**, which is in the RW-PLAN regression row `tests/unit/runtime/security` and not in A1 (finding B-04); it is an Operator decision (Q-SUP-05) and belongs to the split-off proposal T-IGX-051-I (`tenant_scope.py` would move with it; `tenant_scope.py` is also a named EBH-7 F10 implementation path, so the change adds to KR-09, the PT-Q11 re-pin). The two callers pass explicit values today: the tenant middleware passes `False` (`application_security_wiring.py@H:191`), the defense plugin passes `True` (`defense_plugin.py@H:101`). With the changed rule the middleware would block task intake that has no resource tenant, which is a behaviour decision (Q-SUP-05, G-07). The contract edit adds no type; the ratchet check applies.

F-D5-12 (`online_evaluation_registry.py`, class C). The file lock is released by an unconditional unlink (`:227`) and a stale lock is removed by age only (`:249`). Design: the lock file carries a per-acquisition owner token (pid plus random value); release removes the file only if the token still matches; a writer that finds a foreign token on release does nothing and reports a typed warning. The id-existence oracle half is checked at step 0 and fixed only if reproduced.

F-D8-14 (hold stores). SQLite holds are `INSERT OR REPLACE` on a (tenant, artifact) key (`sqlite_artifact_store.py@H:348`, schema `:134`); in-memory holds are a bare key set with no holder or reason (`in_memory_artifact_store.py@H:36`). Design: a hold is identified by (tenant, artifact, holder); placing again by the same holder keeps the first record (original reason and time), placing by another holder adds a record; `is_under_legal_hold` is true while any record exists; the hold API keeps its signatures (`store_base.py@H:129-135`), so the callers in `sources.py` and `governed_access.py` stay valid. A schema migration copies the old table into the new one inside one transaction. Release by holder needs the governed layer and is Q-SUP-06 (G-08); release without a holder keeps today's meaning.

### Part G — strict-config validator (F-D6-10; EX-D6-02, F-D6-04 requalified)

State at `H`. `validate_strict_security_configuration` returns nothing for a non-PRODUCT profile (`security_assembly_resolver.py@H:203`), tests only that the signing secret is not blank (`:221`), does not check that the defenses are on, and takes `plugin_entry_points_enabled` as a caller flag defaulting to False (`:194`). `assert_strict_security_configuration` has no product caller (F-64).

Design (F-D6-10).
- A signing secret shorter than a stated minimum is refused (reason code from `security_configuration.py`, no value in the message).
- A product profile with signing, tool-injection defense, poisoning defense or the audit trail disabled is refused unless the profile carries an explicit, named lab-style override that the validator rejects for product anyway; draft default: refused without exception.
- A lab profile in `ExecutionMode.STRICT` is validated with the same rules.
- Plugin entry-point discovery is read from the environment profile or the plugin flag helper, not from a caller-supplied default of False.

Requalification (EX-D6-02, F-D6-04; plan of record: requalify first). Step S0-6 reads whether any shipped host factory composes the D6 controls (`assert_strict_security_configuration`, the audit trail, the defense bootstrap). At `H` the assertion function has no product caller. If that is confirmed the finding is real, the red-first test NT-RW-SUP-08 is red, and the fix is a call in the host factory that builds the environment wiring. That factory is `environment_wiring.py` (PT-FC3) or an `_shared` composition file; neither is in A1 (G-09). `security_runtime_bridge.py` is the conditional path: it is edited only if step 0 shows the bridge is the right composition point for a D6 control (condition recorded in the ticket).

### Part H — static secret gate and sandbox-strength gate (F-D6-14 / PDC-D6-R007, PDC-D6-R010)

State at `H`. A static scan exists as a module (`raw_secret_gate.py@H:146`, `:230`) and runs only as a pytest test (PDC audit F-D6-14); no script or CI step runs it. The audit lists the escapes: string concatenation, f-strings, bytes, a split token prefix and `or "literal"` fallbacks. `check_sandbox_policy_wiring.py` checks that a sandbox is wired, not how strong it is (`:23`); `SandboxProfile` has two fields, `root` and `enable_exec_tool` (`sandbox_profile.py@H:18`).

Design.
- New `scripts/maintenance/check_static_secret_literals.py`: scans `intergrax/`, `agents/`, `applications/` with the existing `scan_tree_for_raw_secrets` and a second, script-owned AST pass for the five escape shapes (constant folding of concatenation and f-strings with constant parts, bytes literals, adjacent-literal and join forms of a credential prefix, `or "literal"` and `if not x: x = "literal"` fallbacks bound to a secret-named target). Findings carry path, line, rule and the identifier, never the value. Exit 1 on any finding. The module `raw_secret_gate.py` is not edited (A1 equality); moving the new rules into it is G-10 and Q-SUP-08. `secret_safety.py` (A1) holds only the redaction helper of Part E.
- The gate must report at least one real finding over the product tree at the pre-change commit, or the NT row is re-dispositioned (RW-PLAN row note). The dry run answered S0-7: the existing scanner reports 0 findings at the head, and a rough AST probe of the five shapes finds 3 candidate sites in 3 product files outside A1 (`intergrax/agent_distribution/control_plane_governance.py`, `intergrax/runtime/diagnostics/in_memory_problem_persistence.py`, `intergrax/runtime/diagnostics/persistence_conformance.py`; values not printed). Step 0 triages them: a false positive gets one entry in a **script-owned allowlist** (path, identifier, reason; a constant in the script, so it needs no new tracked file and no product edit); a true positive is a defect recorded as an **expected-red with owner Operator** (a product edit outside A1) and the script's tree run stays non-zero for that site until fixed. The seeded tests (NT-RW-SUP-16a, 16b) do not depend on the tree result (dry-run finding B-08).
- New `scripts/maintenance/check_sandbox_isolation_strength.py`: composes a product profile with the exec tool enabled and a weak sandbox configuration and expects refusal; composes a strong one and expects acceptance. "Weak" is defined in Q-SUP-07; draft default: the strength attributes that exist on the sandbox host policy at `H`, read at step 0, with a table in the script (UNVERIFIED which attributes exist; `SandboxProfile` alone has none).

### Part I — retention composition and AUDIT_EVIDENCE producer (EX-D8-01)

State at `H`. `AUDIT_EVIDENCE` is a retention class that never expires (`retention.py@H:63`) and is protected from deletion (`store_base.py@H:65`); no product code assigns it to an artifact (F-78). The sweeper and its registration exist (`retention.py@H:105`, `:253`) and the worker composes them only when an artifact store is passed (`worker_bootstrap.py@H:128`, optional `None` at `:77`; `agent_runtime_context.py@H:482`).

Design (compose, plan of record default).
- `retention.py`: a producer function that builds an `Artifact` with `retention_class=AUDIT_EVIDENCE` for a stated evidence kind and writes it through `ArtifactStore.put`; and a composition gate `require_retention_composition(production_mode, artifact_store, audit, ...)` that raises when production is requested and the store, the audit recorder or the sweep registration is absent. Both are pure functions with in-process fixtures as tests.
- Wiring the gate into the worker root (`worker_bootstrap.py`, `agent_runtime_context.py`) is outside A1 (G-11). Without that wiring the gate exists but no shipped root calls it, so NT-RW-SUP-15 is only partly green (the producer half); the plan says so in the ticket.
- The alternative named in the plan of record (an explicit responsibility statement with owner, scope and a fail-closed gate) is a documentation artefact and A1 has no docs; it is Q-SUP-09.

## 4. A1 gaps

The code reading found that A1 does not contain every path needed to close the findings it lists. This plan keeps its base write set equal to A1 (section 5) and records the rest here for the orchestrator; section 5b turns the accepted-looking ones into proposed additions with a class. Nothing below is claimed as in scope.

### 4.1 Product paths needed but not in A1

| Gap | Path | Why | Owner today | Draft default |
|---|---|---|---|---|
| G-01 | `intergrax/applications/_shared/production_lifecycle.py`, `reference_production_lifecycle.py` | EX-D1-02 names both launchers; they build the pipeline context (F-22, F-23) | the first is an A1 path of PT-WF | Not needed under the Part B design (the pipeline resolves the hook from the composition); add only if the Operator wants explicit per-launcher wiring |
| G-02 | `intergrax/agent_distribution/package_trust.py` | `assert_install_admission` has no signature step (F-06) | no train ticket | Part A adds the step in `admin_service.py` instead; add the file only if the coordinator should own it |
| G-03 | a new product module for a standalone revision-trust admission (no admin service on the GCA strict path) and for operator-input loaders | A1 has no new product module | none | loaders live in `production_agent_capability_runtime.py`; the standalone admission needs a ruling if step S0-4 shows no admin service on the GCA path |
| G-04 | the serving read side (`serving_execution_admission.py` or the activation service) for EX-D1-04 | a serving stop may need a primitive outside A1 | none | step-0 read; stop if none exists in A1 files |
| G-05 | `intergrax/applications/_shared/environment_wiring.py` | the only caller of `bootstrap_security_providers` (F-37) | PT-FC3 | one keyword at the call, after PT-FC3 merges |
| G-06 | `intergrax/runtime/nexus/tools/plan_context_invocation.py` | the divergent poisoning default (F-57) | none | add as firm; otherwise F-D6-09 is re-dispositioned |
| G-07 | `intergrax/applications/_shared/application_security_wiring.py` | the middleware caller of `tenant_scope_is_valid` (F-54) | none | follows the answer to Q-SUP-05 |
| G-08 | `intergrax/runtime/nexus/artifacts/sources.py`, `governed_access.py` | holder-scoped release (F-71) | none | not needed for the draft default |
| G-09 | host factory or `environment_wiring.py` call of `assert_strict_security_configuration` | no caller exists (F-64) | PT-FC3 for `environment_wiring.py` | decided by requalification (S0-6) |
| G-10 | `intergrax/core/security/raw_secret_gate.py` | the evasions are in the scanner module (F-46) | none | the script owns the second pass; moving the rules is Q-SUP-08 |
| G-11 | `intergrax/runtime/task/worker_bootstrap.py`, `intergrax/runtime/execution/agent_runtime_context.py` | a shipped root must call the retention gate (F-75, F-76) | none | add as firm if the Operator chooses "compose" |
| G-13 | `intergrax/runtime/security/multi_region_audit_trail.py` | `verify_replication` would need to call the chain verification to report tamper evidence | none | not edited; the claim stays bounded by its docstring |

### 4.2 Existing tests and support modules touched by the design (census of the dry run at `881428fb2`)

The revision-2 design keeps the direct constructions working, so most of the dry run's census needs no edit. Disposition per group:

| Group | Files (dry-run count) | Disposition |
|---|---|---|
| callers of `install_agent`/`upgrade_agent` without a lab policy | 30 files (13 in `tests/unit/agent_distribution`, 3 `testing_support`, 3 `tests/integration/agent_distribution`, 3+2 `tests/qualification/pdc_x/d1`/`d1_repair`, 3 `tests/unit/applications`, 1 `tests/unit/architecture`, 1 `platform_proofs`) | no edit: the service-level default is 'not composed' (rule 1); the new strict behaviour is reached only through the production factory |
| launcher and `run_deploy_and_activate` drivers | up to 23 files | no edit: the pipeline pass-through when no capability runtime is composed |
| constructors of `ProductionAgentPlatformAdminConfig` | `testing_support/canonical_agent_lifecycle_composition.py:471`, `platform_proofs/scenarios/ai_incident_investigation/integration/production_validation.py:335`, `tests/unit/applications/test_ac4_phase9_production_composition_e2e.py:360` | **edit** (section 5b, P1-P3): pass `lab_package_trust_composition()`; the plan's first grep missed the `platform_proofs` one |
| plugin admission positive tests | `tests/qualification/pdc_x/d6/test_d6_plugin_provenance.py`, `d6_repair/test_d6_f05_plugin_provenance_binding.py` | no edit: `installed_digest_resolver` defaults to `None` |
| `tests/unit/runtime/security/test_tenant_scope.py` (matrix row `("A", None, False, True)`) | 1 file | inverted by NT-RW-SUP-12a: split-off T-IGX-051-I or cond C3 |
| `tests/qualification/pdc_x/d6/test_d6_amendment1_gap_closures.py` (non-strict plan-context pin) | 1 file | inverted by NT-RW-SUP-11: split-off T-IGX-051-I or cond C2 |
| trail and schema pins: `tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py`, `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json`, `tests/qualification/ebh_5/_sqlite_touchpoint_registry.py`, `tests/qualification/state_x/_state_x_explicit_mechanism_classifications.py` | 4 files | keep one append site and `_SCHEMA_VERSION` where possible (Parts E, F); if a pin still moves, cond additions C4-C7 |
| validator tests (`pdc_x/d6/test_d6_security_config_fail_closed.py`, `test_d6_demo_seed_gate.py`) and hold tests (`pdc_x/d8/*`, `d8_repair/*`) | read at step 0 | no edit expected: the validator rules apply to weak settings only; the hold API keeps its signatures |
| `tests/qualification/pdc_x/d1_repair/test_d1_repair_pipeline_hook.py` | 1 file | no edit: the context still accepts the hook keyword |
| PROD-Q serving-wiring gate `tests/qualification/prod_q/test_prod_q_r1_r7_q_package_b_strict_serving_wiring_gate.py` | 1 file | no edit: the serving command is not edited (Part B); the PROD-Q node needs its own authorization and is not run here |

## 5. Write set

Branch rule: product commits go only to branch `arch-freeze/rw-train`. The set below equals `RW_PLAN.md` Appendix A1 block `T-AF2-RW-SUPPLY` (the 33 rows, checked row by row against `rw_plan.v1.json`; all 31 firm and 2 conditional paths exist at `H` or are new with an existing parent directory, checked with `git ls-tree`). Counts: **20 firm product paths** (19 files and one glob), **11 firm test and script paths** (9 new test files, 2 new scripts), **2 conditional product paths**; firm total 31, conditional 2. One path per line.

### Product paths (ACT-N11), firm

```
intergrax/agent_distribution/trust.py
intergrax/agent_distribution/admin_service.py
intergrax/applications/_shared/deploy_activate_pipeline.py
intergrax/applications/_shared/production_agent_capability_runtime.py
applications/governed_contractor_application/host/package_b_strict_production_serving_command.py
applications/governed_contractor_application/host/single_node_profile_v1_strict_startup.py
intergrax/core/plugins/*
intergrax/runtime/security/security_audit_trail.py
intergrax/runtime/security/tenant_scope.py
intergrax/runtime/security/defense_plugin_loader.py
intergrax/contracts/agent_run.py
intergrax/core/security/secret_safety.py
intergrax/llm_adapters/governance/quota.py
intergrax/applications/_shared/security_assembly_resolver.py
intergrax/tools/providers/rag/service.py
intergrax/runtime/architecture/online_evaluation_registry.py
intergrax/runtime/nexus/artifacts/retention.py
intergrax/runtime/nexus/artifacts/store_base.py
intergrax/runtime/nexus/artifacts/sqlite_artifact_store.py
intergrax/runtime/nexus/artifacts/in_memory_artifact_store.py
```

`intergrax/core/plugins/*` is a glob that matches 9 tracked files at `H`; the expected edit is `admission.py`, and `discovery.py` only if Part D step 0 requires it. The other seven stay untouched (the ruff findings in `__init__.py`, `manifest_io.py`, `package_contract.py` and the format hunks in `platform_qualification.py` are not touched: pre-commit ruff checks staged files only). `service.py` receives no behavioural edit in the draft default (Part F, F-D6-09). `package_b_strict_production_serving_command.py` is firm in A1 and **unedited** in the draft default (Part B).

### Product paths (ACT-N11), conditional

```
intergrax/core/security_bootstrap.py
intergrax/applications/_shared/security_runtime_bridge.py
```

Conditions. `core/security_bootstrap.py`: only if step S0-5 shows that the strict or product root reaches `load_security_defense_plugin_report` without provenance and the `require_provenance` flag cannot ride on `SecurityDefenseAdmissionPolicy` alone (FRZ-SEC-06, basis C). `security_runtime_bridge.py`: only if the requalification S0-6 shows the bridge is the composition point of a D6 control that no host composes (EX-D6-02, F-D6-04).

### Tests and scripts (not ACT-N11 paths), new, firm

```
scripts/maintenance/check_static_secret_literals.py
scripts/maintenance/check_sandbox_isolation_strength.py
tests/unit/runtime/security/test_rw_supply_defense_plugin_provenance_required.py
tests/unit/agent_distribution/test_rw_supply_attestation_and_revocation.py
tests/unit/applications/test_rw_supply_trust_hook_before_persist.py
tests/unit/core/plugins/test_rw_supply_installed_file_digest.py
tests/unit/runtime/security/test_rw_supply_audit_trail_tamper_evidence.py
tests/unit/runtime/security/test_rw_supply_fail_open_set.py
tests/unit/applications/test_rw_supply_strict_validator_refuses_weak.py
tests/unit/scripts/test_rw_supply_static_secret_gate.py
tests/unit/runtime/nexus/test_rw_supply_retention_and_legal_hold.py
```

The two scripts are under `scripts/`, which is not an ACT-N11 product prefix (`PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")`, `scripts/governance/iiac_common.py@2727a2c23:22`); commits that touch `scripts/` still carry the `Ref:` trailer (AGENTS.md).

### Edited tests

None are in A1 for this ticket. Section 5b proposes the ones the design needs.

### Docs

None. The EX-D8-01 responsibility statement alternative is documentation and is not in this ticket (Q-SUP-09).

### 5b. Proposed additions to A1 (not in scope until the RW-PLAN is amended)

Each path has a reason and a class. `firm` = needed by the design as drafted; `cond` = needed only under the stated condition. Counts if all are accepted: **+3 firm, +8 conditional**; the base counts above (31 firm, 2 conditional) do not change.

| ID | Path | Class | Reason / condition |
|---|---|---|---|
| P1 | `testing_support/canonical_agent_lifecycle_composition.py` | firm | builds `ProductionAgentPlatformAdminConfig` (`:471`); passes `lab_package_trust_composition()` once `package_trust` is required (B-02) |
| P2 | `platform_proofs/scenarios/ai_incident_investigation/integration/production_validation.py` | firm | the same (`:335`); missed by the revision-1 grep |
| P3 | `tests/unit/applications/test_ac4_phase9_production_composition_e2e.py` | firm | the same (`:360`) |
| C1 | `intergrax/runtime/nexus/tools/plan_context_invocation.py` | cond | NT-RW-SUP-11 (F-D6-09) cannot go green without it; condition: Q-SUP-01 accepts G-06 and the split of F-D6-09 is not made (otherwise it moves to T-IGX-051-I) |
| C2 | `tests/qualification/pdc_x/d6/test_d6_amendment1_gap_closures.py` | cond | pins the non-strict plan-context behaviour (`:346`); same condition as C1 |
| C3 | `tests/unit/runtime/security/test_tenant_scope.py` | cond | the matrix row inverted by NT-RW-SUP-12a; condition: Q-SUP-05 answered 'close' and the split is not made |
| C3b | `intergrax/applications/_shared/application_security_wiring.py` | cond | the middleware caller (`:191`) if Q-SUP-05 changes intake semantics; same condition as C3 |
| C4-C7 | `tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py`, `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json`, `tests/qualification/ebh_5/_sqlite_touchpoint_registry.py`, `tests/qualification/state_x/_state_x_explicit_mechanism_classifications.py` | cond | pins of the trail append site and the SQLite schema constant; condition: step 0 or S1 shows a pin moves despite the design (B-07) |

Not proposed, listed for the record: `intergrax/contracts/execution_evidence/persistence_port.py` (a paging cursor; contract surface, ratchet, a new dependency of every port implementation); the three product files of the secret-gate candidates (B-08; a true positive is an open item with owner Operator, section 17; no test asserts exit 0 over the tree); G-01..G-11 and G-13 of section 4.1 other than C1 and C3b.

**Proposed split (Q-SUP-11): T-IGX-051-I, T-AF2-RW-SUPPLY-TENANTSCOPE.** F-D6-09 and the semantic half of F-D6-11 (NT-RW-SUP-11, NT-RW-SUP-12a) need an Operator decision (Q-SUP-05), invert two pinned tests, touch a named EBH-7 F10 path and two hotspot callers. Paths of the split ticket: `intergrax/runtime/security/tenant_scope.py` (moves out of this ticket's A1 block), `intergrax/runtime/nexus/tools/plan_context_invocation.py`, `intergrax/applications/_shared/application_security_wiring.py` (cond), `tests/unit/runtime/security/test_tenant_scope.py`, `tests/qualification/pdc_x/d6/test_d6_amendment1_gap_closures.py`, and its own new test file for NT-RW-SUP-11 and 12a. It merges after this ticket (no shared file: this ticket keeps `agent_run.py` and `service.py`). RW-SUPPLY then has 19 firm product paths (20 minus `tenant_scope.py`), 30 firm in total and 2 conditional, and 18 NT ids in 48 test nodes (NT-11 and the 12a node move). If the split is not accepted, C1-C3b are cond additions of this ticket.

One further, separate item for the orchestrator: the caller migration of revision 1 (about 50 files) is **not** needed under the revision-2 design, so no migration ticket is proposed.

**Forbidden (a need for any of these stops the work):** the paths of section 4.1 and 5b until an amendment adds them; `intergrax/applications/_shared/production_lifecycle.py` (PT-WF); `applications/governed_contractor_application/host/` files other than the two listed; `package_b_strict_production_serving_command.py` (any edit; PT-WB Part G and A4-040 territory, F-29); the compose file, the RUNBOOK and the env example of PT-WB Part G; `.github/`, `pyproject.toml`, `uv.lock`; the roadmap, the FRZ checklist, IIAC and PDC-X record directories; gate scripts other than the two new ones; the F10 inputs; Docker or vendor work; real keys, real revocation feeds or live backends.

## 6. Negative tests

Types: R = a configuration that must now be refused or detected (red at `H`); S = structural (red at `H` because the structure is absent); C = control (green before and after) or requalification (red only if step 0 confirms); G = gate (red at `H`: a probe at the pre-change commit runs the same assertion against the existing module and fails). Every R, S and G test must fail at the pre-change commit for a behavioural reason, not for a missing import alone; for tests of a new function the red-first run asserts the old surface (for example, the old `place_legal_hold` result) so the failure is an assertion. Node ids are `file::test`. RW-PLAN ids: NT-ROW-075..096 and 159 (section d). 19 NT ids and 50 test nodes are **written**; the **merge set** is 18 ids and 48 nodes (NT-RW-SUP-11 and the 12a node are held out, see below; with the split of Q-SUP-11 they leave this ticket for good).

**Held-out tests.** **No expected-red test merges inside this ticket** (orchestrator ruling on the RW-PLAN v19 audit): NT-RW-SUP-11 and NT-RW-SUP-12a are written and kept in the worktree or the draft folder but are **held out of the merge set** until Q-SUP-11 or the split ticket T-IGX-051-I gives them a fixer (the same rule PT-T01 applies to its NT-07 and NT-04b); the same rule applies to NT-RW-SUP-08 and NT-SEC-06-1b if step 0 shows them red and their fixer paths (G-09, G-05) are not added. They are listed as open items with their owner and their blocking effect: F-D6-09 and the semantic half of F-D6-11 are not closed by this ticket, and the PDC rows that depend on them (PDC-D6-R014 note, the A-12 set) stay open; owner Operator for Q-SUP-05, orchestrator for Q-SUP-01 and Q-SUP-11. Each merged test of this ticket is green at the ticket head; a test whose fixer is outside the write set does not merge.

| ID | NT row | Configuration | Expected refusal / result | Node id | Type |
|---|---|---|---|---|---|
| NT-RW-SUP-01 | 075 (EX-D1-01) | install through the production admin service factory with a trust record that has no signature material, policy default | `AgentPackageTrustError` (unsigned forbidden) | `test_rw_supply_attestation_and_revocation.py::test_sup_01a_unsigned_install_refused` | R |
| NT-RW-SUP-01 | 075 | upgrade with material whose signature does not match the digest | refused; nothing persisted | `...::test_sup_01b_tampered_upgrade_refused` | R |
| NT-RW-SUP-01 | 075 | caller-asserted record naming a signature evidence ref without material | refused (forged ref) | `...::test_sup_01c_forged_evidence_ref_refused` | R |
| NT-RW-SUP-01 | 075 | valid signature with a generated throwaway key | admitted | `...::test_sup_01d_valid_attestation_admitted` | C |
| NT-RW-SUP-02 | 076 (EX-D1-02) | production admin service factory without `package_trust` | `TypeError`; a `None` member is a `ValueError`; a lab composition is built only through `lab_package_trust_composition()` | `...::test_sup_02a_factory_requires_trust_composition` | S |
| NT-RW-SUP-02 | 076 | revoked digest through `ProductionLifecycleLauncher.deploy_and_activate` | refused before any store write | `...::test_sup_02b_revoked_digest_refused_production_launcher` | R |
| NT-RW-SUP-02 | 076 | the same through `ReferenceProductionLifecycleLauncher` | refused | `...::test_sup_02c_revoked_digest_refused_reference_launcher` | R |
| NT-RW-SUP-02 | 076 | GCA strict startup given an `application_composition` whose admin config carries a lab or missing `package_trust` | `ValueError` before the session opens | `...::test_sup_02d_strict_startup_refuses_lab_trust_composition` | R |
| NT-RW-SUP-03 | 077 (EX-D1-04) | revoked serving revision, no trusted rollback target | serving stops (typed stop), report still returned | `...::test_sup_03_revoked_without_target_stops_serving` | R |
| NT-RW-SUP-04 | 078 (D1-R017) | install through the production composition with an evidence-plane port | the install events appear as audit entries of the tenant | `...::test_sup_04_install_event_reaches_evidence_plane` | R |
| NT-RW-SUP-04 | 078 | composition with no event sink | `ValueError` at composition | `...::test_sup_04b_missing_sink_refused_at_composition` | S |
| NT-RW-SUP-05 | 079 (A-01) | the trust hook raises for the candidate | no revision persisted | `test_rw_supply_trust_hook_before_persist.py::test_sup_05a_refused_revision_not_persisted` | R |
| NT-RW-SUP-05 | 079 | the same | no validation, projection registration or `prepare_candidate` call | `...::test_sup_05b_nothing_validated_registered_prepared` | R |
| NT-RW-SUP-05 | 079 | a composed capability runtime and no hook on the context (reference launcher) | the admin service's admission is called; a refused revision is not persisted | `...::test_sup_05c_hook_resolved_from_composed_runtime` | R |
| NT-RW-SUP-05 | 079 | a composition without a capability runtime and no hook | explicit pass-through recorded as `trust_admission_not_composed`; behaviour as today | `...::test_sup_05d_no_runtime_pass_through_is_explicit` | C |
| NT-RW-SUP-06 (= NT-PLG-05-1) | 080 (EX-D6-03) | a fixture distribution with one installed file altered after the digest was recorded, admission built with `installed_digest_resolver=installed_tree_digest` | `PLUGIN_PROVENANCE_INVALID` | `test_rw_supply_installed_file_digest.py::test_sup_06a_altered_installed_file_refused` | R |
| NT-RW-SUP-06 | 080 | a RECORD entry removed | refused | `...::test_sup_06b_missing_file_refused` | R |
| NT-RW-SUP-06 | 080 | matching digest | admitted | `...::test_sup_06c_matching_digest_admitted` | C |
| NT-SEC-06-1 | 159 (FRZ-SEC-06) | `require_provenance=True`, discovery on, `provenance=None` | `SecurityConfigurationError`; the entry point is never loaded | `test_rw_supply_defense_plugin_provenance_required.py::test_sec_06_1_strict_root_refuses_unattested_defense_plugins` | C |
| NT-SEC-06-1 | 159 | a strict root built through `environment_wiring.py` | refuses (only after G-05) | `...::test_sec_06_1b_strict_environment_wiring_requires_provenance` | C |
| NT-RW-SUP-07 | 081, 082 (EX-D6-01, F-D6-02) | one stored entry modified | `verify_chain` reports the entry (both trails) | `test_rw_supply_audit_trail_tamper_evidence.py::test_sup_07a_modified_entry_detected` | R |
| NT-RW-SUP-07 | 081 | one interior entry removed | detected | `...::test_sup_07b_removed_interior_entry_detected` | R |
| NT-RW-SUP-07 | 081 | untouched trail | verifies clean; head digest stable | `...::test_sup_07c_clean_chain_verifies` | C |
| NT-RW-SUP-07 | 081 | (only if Q-SUP-03 picks `NOT_CLAIMED`) claim-text check over the SEC/AUD wording | fails on any tamper-evidence claim | `...::test_sup_07d_not_claimed_wording_check` | C |
| NT-RW-SUP-08 | 083, 084 (EX-D6-02, F-D6-04) | a production host factory | composes `assert_strict_security_configuration` | `test_rw_supply_strict_validator_refuses_weak.py::test_sup_08_production_host_composes_d6_controls` | C |
| NT-RW-SUP-09 | 085 (F-D6-10) | product profile, one-character signing secret | issue returned | `...::test_sup_09a_short_signing_secret_refused` | R |
| NT-RW-SUP-09 | 085 | product profile with signing, defenses and audit trail disabled | issues returned | `...::test_sup_09b_disabled_defenses_refused` | R |
| NT-RW-SUP-09 | 085 | lab profile in STRICT mode | validated like product | `...::test_sup_09c_lab_profile_in_strict_mode_validated` | R |
| NT-RW-SUP-09 | 085 | plugin discovery on, no trust policy, flag not supplied by the caller | issue returned | `...::test_sup_09d_plugin_discovery_without_policy_refused` | R |
| NT-RW-SUP-10 | 086, 087 (F-D3-12, D3-R018) | `INTERGRAX_LLM_TENANT_MAX_TOKENS` set to a non-numeric value | `LLMQuotaConfigurationError` (denial) | `test_rw_supply_fail_open_set.py::test_sup_10a_malformed_quota_denies` | R |
| NT-RW-SUP-10 | 086 | negative value | denial | `...::test_sup_10b_negative_quota_denies` | R |
| NT-RW-SUP-10 | 086 | unset and `0` | disabled, as documented | `...::test_sup_10c_unset_and_zero_disable` | C |
| NT-RW-SUP-10 | 087 | the TENANT-X carrier count of T-IGX-041-I | recorded in the PR body (a record, not a test node) | step S0-8 | — |
| NT-RW-SUP-11 | 088 (F-D6-09; **held out of the merge set** until C1/C2 or the split ticket, B-04) | profile absent, `production_mode` false, plan-context path | defense ON, same as the RAG path | `...::test_sup_11_poisoning_default_same_on_both_paths` | R |
| NT-RW-SUP-12 | 089 (F-D6-11; **12a held out of the merge set** until C3/C3b or the split ticket, B-04) | scoped request, tenant-less resource, `allow_unscoped=False` | invalid | `...::test_sup_12a_tenantless_resource_not_open` | R |
| NT-RW-SUP-12 | 089 | `RequestIdentity(tenant_id="  t  ")` | stored as `t` | `...::test_sup_12b_request_identity_tenant_normalised` | R |
| NT-RW-SUP-13 | 090 (F-D5-12) | a writer whose lock was taken over releases | the new owner's lock stays | `...::test_sup_13a_foreign_release_keeps_lock` | C |
| NT-RW-SUP-13 | 090 | lookup of an id of another tenant vs an absent id | the same response | `...::test_sup_13b_existence_oracle_closed` | C |
| NT-RW-SUP-14 | 091 (F-D8-14) | two holders place a hold on one artifact (SQLite) | both records kept; first reason intact | `test_rw_supply_retention_and_legal_hold.py::test_sup_14a_second_hold_does_not_overwrite_sqlite` | R |
| NT-RW-SUP-14 | 091 | the same (in memory) | holder and reason retained | `...::test_sup_14b_second_hold_does_not_overwrite_in_memory` | R |
| NT-RW-SUP-14 | 091 | the old table is migrated | existing holds survive | `...::test_sup_14c_old_table_migrated_without_loss` | C |
| NT-RW-SUP-15 | 092 (EX-D8-01) | the AUDIT_EVIDENCE producer | the stored artifact has class `AUDIT_EVIDENCE` and is not deletable | `...::test_sup_15a_audit_evidence_producer` | R |
| NT-RW-SUP-15 | 092 | production composition without store, audit recorder or sweep registration | `ValueError` at composition | `...::test_sup_15b_production_requires_retention_composition` | R |
| NT-RW-SUP-16 | 093, 094 (F-D6-14, D6-R007) | seeded sources for the five escape shapes | each flagged by the script | `test_rw_supply_static_secret_gate.py::test_sup_16a_seeded_evasions_flagged` | G |
| NT-RW-SUP-16 | 093 | a secret-shaped literal in a temporary product-like tree | exit 1, no value in the output | `...::test_sup_16b_secret_literal_fails_gate_without_echo` | G |
| NT-RW-SUP-16 | 093 | the real product tree | the finding count is recorded; zero findings means re-disposition | `...::test_sup_16c_product_tree_scan_recorded` | G |
| NT-RW-SUP-17 | 095 (F-D6-03) | a trail with exactly the port limit of events | `AuditTrailListingTruncated` (never a silent 1000); below the limit the list is returned | `test_rw_supply_audit_trail_tamper_evidence.py::test_sup_17a_listing_reports_possible_truncation` | R |
| NT-RW-SUP-17 | 095 | a credential-shaped value under a benign key (in-memory trail) | stored redacted | `...::test_sup_17b_in_memory_value_redacted` | R |
| NT-RW-SUP-17 | 095 | the same (evidence-plane trail) | stored redacted | `...::test_sup_17c_evidence_plane_value_redacted` | R |
| NT-RW-SUP-18 | 096 (D6-R010) | exec tool enabled and a sandbox configuration that is present but weak | refused by the gate | `test_rw_supply_static_secret_gate.py::test_sup_18a_weak_sandbox_refused` | G |
| NT-RW-SUP-18 | 096 | a strong configuration | accepted | `...::test_sup_18b_strong_sandbox_accepted` | C |

Notes. The C-type tests are controls or conditional requalification rows and are not red-first evidence; NT-RW-SUP-08, 13 and NT-SEC-06-1 are red-first only if step 0 shows the condition at `H`, otherwise the row is re-dispositioned (RW-PLAN rule); a conditional row that is red with its fixer outside the write set is held out like NT-RW-SUP-11. The audit-trail file carries two NT ids (07, 17) and the fail-open file four (10 to 13), as in A1. NT-RW-SUP-10 also records the TENANT-X carrier count in the PR body; that record is not a test node.

## 7. Regression scope and test commands

All commands run one process at a time under `D:/Projekty/_wt/_locks/pytest.lock` (`mkdir` must succeed, `rmdir` right after, also on failure; only the creator removes it), logs under `.tmp/session/arch-freeze/rw-supply-impl/`, one `uv run pytest` per step with several paths, `-p no:cacheprovider`, no broad or diagnostic suite, no Docker, no vendor integration.

The RW-PLAN regression row (section e) for this ticket: `tests/unit/agent_distribution`; `tests/unit/core/plugins`; `tests/unit/runtime/security`; `tests/qualification/pdc_x/xd`; `scripts/maintenance/check_immutable_security_audit_trail.py`; `scripts/maintenance/check_sandbox_policy_wiring.py`; the contract typing ratchet check (RWD7-02). **Contract edit check:** this ticket edits `intergrax/contracts/agent_run.py`; at the ticket head the wave-packet check must run `.tmp/session/arch-freeze/closed-world/typ_measure.py` and `.tmp/session/arch-freeze/rw-plan/work/typ05/measure_typ05.py` over `intergrax/contracts/**` and report no metric above the HEAD baseline (Any 138, object 357, reflection counts as in `RATCHET_BASELINE`) and no new string-keyed lookup table (still the 5 of TYP05_REDUCTION). Any increase is removed or listed with this ticket and the reason (OD-AF-11); `typ_ratchet_baseline.v1.json` is not a tracked file at the train head (`git ls-files` 0 hits), so the ratchet exists only as those two `.tmp` scripts and the PR body records their output.

Additional regression files named by this plan (the RW-PLAN row does not list them; INPUT records the difference): `tests/qualification/pdc_x/d1_repair/test_d1_repair_pipeline_hook.py`, `tests/qualification/pdc_x/d1`, `tests/qualification/pdc_x/d6`, `tests/qualification/pdc_x/d8`, `tests/qualification/pdc_x/d8_repair`, `tests/unit/llm_adapters/test_llm_quota.py`, `tests/unit/applications/test_ac3_phase5_production_lifecycle_e2e.py`, the text-pin files of rule 9 (`test_agent_distribution_activation_remediation.py`, `test_d8_retention.py`, `test_wb_serving_bind_defaults.py`, `test_ebh_2g_r1_retrieval_contract_typing_gate.py`).

**Expected-red and registered classes at the train head (dry run, not caused by this ticket):** `governance-fast` KR-08 (15 ACT-N11 nodes, red on every PR into the train); EBH-7 F10 `tenant_scope.py` is a named path (KR-09, PT-Q11 re-pin). `tests/unit/runtime/security/test_tenant_scope.py` and the pdc_x d6 amendment test stay green and unchanged in the merge set, because NT-RW-SUP-11 and 12a are held out.

**Planned expected-red of existing tests (owner RW-TENANT, registered by the orchestrator):** RW-SUPPLY may add tenant carriers (the tenant on lifecycle-evidence audit entries in Part B, the `RequestIdentity` tenant strip in Part F, the audit-trail entries of Part E). If it does, the pin nodes of `tests/qualification/tenant_x` that count carriers turn red; that is a planned, registered expected-red of existing nodes (not a new test of this ticket), the TENANT-X carrier count of T-IGX-041-I is recorded in the PR body (NT-RW-SUP-10, S0-8), and RW-TENANT fixes the pins. Merge-time CI checks the ticket must pass (the dry run's list): `ci-smoke`, `governance-policy-pin`, `ticket-governance`, `ent-audit-fast` (`ENT-A-SEC-01`), `ent-audit-pr-gate`, `evol-x-b4-preflight`, `conformance`, `rag-gate` (`check_rag_maturity_labels.py`), pre-commit ruff.

| Step | Command | Purpose |
|---|---|---|
| S0 red-first | the nine new test files and the two script checks on a detached worktree at the pre-change commit (a worktree under `_wt/intergrax/rw-supply-s0`, per CLAUDE.md) | every R, S and G test red for a behavioural reason; C tests recorded; the log is kept for the auditor |
| S1 per part | the new test file of the part plus the regression directories of that part | after each commit group |
| S2 regression | the RW-PLAN row plus the additional files above, one command | no regression |
| S3 gates | `uv run python scripts/maintenance/check_immutable_security_audit_trail.py`; `... check_sandbox_policy_wiring.py`; `check_rag_maturity_labels.py`; the two new scripts | exit 0 (the secret gate: exit 0 over the tree only after the findings are triaged, step S0-7 and Part H) |
| S4 contract check | `typ_measure.py` and `measure_typ05.py` over `intergrax/contracts/**` | no new metric |
| S5 static | `uv run ruff check`, `uv run ruff format --check` and `uv run pyright` on the changed Python files | typing and style; the dry run found 8 autofixable check findings and 8 files that would reformat at the head, of which 6 are touched files (`production_agent_capability_runtime.py`, `single_node_profile_v1_strict_startup.py`, `quota.py`, `trust.py`, `agent_run.py`, `secret_safety.py`; the check findings are I001 and UP045), so each touched file gets a mechanical lint-and-format commit **before** its first behavioural hunk (B-09) |

## 8. Order of work

0. Read-only census, nothing edited: S0-1 which shipped hosts build `AgentPlatformAdminService` and which call `ProductionLifecycleLauncher` or `ReferenceProductionLifecycleLauncher`; S0-2 the attributes of the install events (tenant, environment id); S0-3 `EvidencePersistencePort.list_for_task` paging; S0-4 whether the GCA strict path composes an admin service; S0-5 whether a strict or product root reaches `load_security_defense_plugin_report` without provenance (FRZ-SEC-06); S0-6 requalification of EX-D6-02 and F-D6-04; S0-7 one run of the existing scan over the product tree for the F-D6-14 finding count; S0-8 the TENANT-X carrier count of T-IGX-041-I (record only); S0-9 the callers and pinned tests of every changed signature (section 4.2; the dry run's census is the starting point); S0-10 triage of the three secret-gate candidate sites (Part H).
0b. A mechanical lint-and-format commit per touched file that fails `ruff check` or `ruff format --check` at the head (the six files named in S5); no behavioural change in these commits.
1. Write the nine test files and the two scripts' tests; run S0 red-first at the pre-change commit; keep the log.
2. Part F (quota, tenant scope, identity, lock owner, holds): small, independent files; contract check after `agent_run.py`.
3. Part E (audit trail chain, listing, redaction).
4. Part D (installed digest, defense provenance).
5. Part A (attestation on install and upgrade), then Part B (composition, sink, no-target stop), then Part C (hook before persist); the strict serving command is not edited (Part B).
6. Part G (strict validator) and Part H (the two scripts).
7. Part I (producer and retention gate).
8. S2, S3, S4, S5.

## 9. Overlaps and merge order

Lane 4 (RW-PLAN section c): after PT-WA (merged) and PT-WB; after PT-FC2 (plan Z: serialized on `applications/_shared` wiring); parallel with PT-FC3, T-AF2-RW-GATES, PT-WE; before T-AF2-RW-TENANT and T-AF2-RW-TYP.

| Shared file | Other ticket | Rule |
|---|---|---|
| `applications/governed_contractor_application/host/package_b_strict_production_serving_command.py` | PT-WB Part G (`cond`, A4-040 ruling open, Q-RW-16 b) | the file is listed firm in A1 and is **not edited** by this plan (the dry run: PT-WB Part G has not landed, the PROD-Q serving-wiring gate patches this file's preamble). **No A4-040 content** (bind-host default, `uvicorn.run`, compose, RUNBOOK, env example). If the ruling lands in RW-SUPPLY instead, it needs its own amendment and a new plan digest |
| `intergrax/agent_distribution/admin_service.py` | T-AF2-RW-TENANT (install CAS) | this ticket first; TENANT rebases onto Parts A and B |
| `intergrax/runtime/architecture/online_evaluation_registry.py` | T-AF2-RW-TENANT (cond, GAP-04, RW-PLAN v18) | not named in revision 1; this ticket first (lock-owner token), RW-TENANT rebases; the ticket edits only the lock release and stale-removal functions |
| `intergrax/contracts/agent_run.py` | T-AF2-RW-TYP (v13 text; in v18 not a scope overlap) | RW-TYP merges after this ticket if it still lists the contract surface |
| `intergrax/applications/_shared/production_lifecycle.py` | PT-WF (A1 path) | not edited here (Part B design) |
| `environment_wiring.py` (G-05), `plan_context_invocation.py` (G-06) | PT-FC3, none | only after an amendment, in the owner's order |

## 10. ACT-N11 / IIAC consequence

Every product path above under `intergrax/`, `agents/`, `applications/` is a product path by `PRODUCT_PATH_PREFIXES` (`scripts/governance/iiac_common.py`); 20 firm and 2 conditional paths break the IIAC activation recorded for PDC-X P8 until the single RB-2 re-pin (DP-58, `PT-Q03`). This ticket does not perform the re-pin and must not merge outside the wave.

## 11. What closes each finding

Nothing is closed by this plan. A finding is closed only by its negative test green at the exact wave SHA, the regression scope green, the contract check clean and an independent audit in a separate session (node `PT-Q02`). EX-D1-01, EX-D1-02, EX-D6-02 and EX-D8-01 stay partly open if the A1 gaps G-01 to G-11 are refused; the ticket then records which half remains.

## 12. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R-1 | The production factory always composes the attestation enforcement; the three config constructors (P1-P3) must pass a lab composition | the direct service constructions are unchanged (rule 1); P1-P3 listed in section 5b |
| R-2 | A composition without a capability runtime has no trust admission (explicit pass-through), and the GCA strict path may be such a composition | S0-4; Q-SUP-02; recorded as expected-open with owner orchestrator, never reported as closed |
| R-3 | A hold-table migration loses holds | one transaction, copy then swap, NT-RW-SUP-14c |
| R-4 | Quota malformed value denies every LLM call of a host that has a typo in the variable | intended fail-closed; message names the variable, never a value |
| R-5 | A payload schema change for the chain breaks the stored audit events | optional fields with defaults; step 0 reads the payload registry rules; a version bump is the fallback |
| R-6 | The installed-tree digest convention does not match how publishers sign | Q-SUP-04 before Part D |
| R-7 | Tenant-scope semantics change blocks intake that has no resource tenant | Q-SUP-05, G-07; the middleware is not changed until decided |
| R-8 | The static secret gate reports no real finding, or many false positives | S0-7 run; RW-PLAN re-disposition rule; findings never echo values |
| R-9 | Merge conflict on `admin_service.py` (2533 lines) with RW-TENANT | order in section 9; Parts A and B land in small hunks |
| R-10 | Real keys or a revocation feed are needed for a production proof | out of scope (PROD-Q); fixtures only, labelled as stubs in the PR body |
| R-11 | The 3 secret-gate candidate sites are true positives in product files outside A1 | script-owned allowlist for false positives; true positives are an open item with owner Operator (section 17) |
| R-12 | A pin on the trail append site or the SQLite schema constant moves | one append site, `_SCHEMA_VERSION` untouched where possible; cond C4-C7 otherwise |
| R-13 | Lint-and-format commits conflict with PT-WB or RW-TENANT on shared files | only touched files get a lint commit; `package_b_...command.py` and the seven untouched `core/plugins` files get none |

## 13. Rollback

Revert the commit group of the part; the groups are independent except B depends on A and E (the sink writes audit entries), C depends on B (the hook is resolved from the composition). The hold-table migration is forward-only; its rollback is restoring the copy kept by the migration. The IIAC rebind must be redone after any revert. No setting or production configuration change is made.

## 14. Open questions with draft defaults

| ID | Who | Question | Draft default |
|---|---|---|---|
| Q-SUP-01 | orchestrator | Amend A1 with the paths of section 4.1 and the pinned tests of section 4.2, or narrow the findings | amend G-06 (plan-context default), G-10 only if the script cannot own the rules, G-11 if "compose" is chosen; the rest decided by step 0 |
| Q-SUP-02 | orchestrator | If the GCA strict path composes no admin service, accept a standalone revision-trust admission (a new product module) | decide after S0-4; until then EX-D1-02 is expected-open for the GCA strict path |
| Q-SUP-03 | Operator | EX-D6-01: build tamper evidence, or `NOT_CLAIMED` with reviewed wording (plan of record section E) | build the hash chain; record the tail-truncation limit |
| Q-SUP-04 | orchestrator | EX-D6-03: the attested `package_digest` of a plugin distribution is the installed-tree digest | yes, with the algorithm of Part D |
| Q-SUP-05 | Operator | F-D6-11: is a tenant-less resource open to a scoped tenant at task intake by intent | no; explicit opt-in only; the middleware follows G-07 |
| Q-SUP-06 | orchestrator | holder-scoped release of legal holds (needs the governed callers) | not in this ticket |
| Q-SUP-07 | orchestrator | what "weak" means for the sandbox gate (PDC-D6-R010) | the strength attributes of the sandbox host policy found at S0 |
| Q-SUP-08 | orchestrator | evasion rules in `raw_secret_gate.py` or in the new script | in the script (A1 equality) |
| Q-SUP-09 | Operator | EX-D8-01: compose retention and the producer, or publish the responsibility statement | compose (plan of record default); the statement needs a docs ticket |
| Q-SUP-10 | orchestrator | A4-040 stays in PT-WB Part G; RW-SUPPLY takes no part in it | yes |
| Q-SUP-11 | orchestrator, Operator for Q-SUP-05 | split F-D6-09 and F-D6-11a (NT-RW-SUP-11, 12a) into T-IGX-051-I (section 5b) | split; if refused, cond additions C1-C3b and the Operator ruling of Q-SUP-05 |

## 15. Authorization requests

| ID | Request | Needed for |
|---|---|---|
| SUP-A1 | the wave-packet authorization of OD-AF-22 for the product paths of this ticket | any product edit |
| SUP-A2 | permission to create one detached worktree under `D:/Projekty/_wt/intergrax/` for the red-first run (one per session, removed afterwards) | S0 |
| SUP-A3 | pytest runs under the shared lock, one process at a time, the files named in section 7 | S0-S2 |
| SUP-A4 | a read-only run of the existing `scan_tree_for_raw_secrets` over the product tree (step S0-7) | Part H |
| SUP-A5 | an amendment of RW-PLAN A1 for the additions of section 5b that the orchestrator accepts (3 firm, 8 conditional) and, if accepted, the new ticket T-IGX-051-I | Q-SUP-01, Q-SUP-11 |

## 16. Effort

| Part | Size | Note |
|---|---|---|
| A attestation | M | trust record field, one admin step |
| B composition, launchers, sink, no-target stop | L | depends on S0-4 and G-03 |
| C hook before persist | S | one pipeline function |
| D installed digest, defense provenance | M | digest algorithm and convention |
| E audit trail chain, listing, redaction | M | two trails, schema |
| F fail-open set | M | five small fixes, one migration |
| G strict validator | S | rules plus requalification |
| H two gates | M | AST second pass |
| I retention producer and gate | S | wiring depends on G-11 |
| **Whole ticket** | **L** | 20 firm product paths, nine commit groups |

## 17. Dry-run dispositions (revision 2)

Source: `dry-run/DRY_RUN_T-IGX-051-B.md` (another session's record, DRY_RUN_FINDINGS, 10 findings; quoted as a label, not as a verdict of this plan). Each HIGH and MEDIUM finding is a plan change or an open item with an owner; no expected-red test merges inside this ticket.

| Finding | Severity | Disposition |
|---|---|---|
| B-01 strict-by-default attestation reds about 26 files | HIGH | plan change: service-level default is 'not composed'; enforcement composed only by the production factory (Part A, rule 1); no migration of the 26 files |
| B-02 fail-closed hook and required `package_trust` reach up to 23 launcher drivers and one `platform_proofs` module | HIGH | plan change: pipeline pass-through when no capability runtime (Part B); the three config constructors become firm additions P1-P3 |
| B-03 mandatory installed-digest check reds the d6 positive tests | HIGH | plan change: `installed_digest_resolver` defaults to `None` (Part D); EX-D6-03 stays partly open until a host composes it (owner: orchestrator) |
| B-04 NT-RW-SUP-11 and 12a cannot go green inside the write set | HIGH | proposed split T-IGX-051-I or cond C1-C3b; until decided both tests are **held out of the merge set** (orchestrator ruling, RW-PLAN v19 audit; no expected-red test merges inside this ticket); open items, owner Operator (Q-SUP-05) and orchestrator (Q-SUP-11) |
| B-05 the evidence port cannot page; `ENT-A-SEC-01` calls `list_entries` | MEDIUM | plan change: raise only when the limit is reached (Part E); the paging cursor is not proposed |
| B-06 PROD-Q wiring gate patches the serving-command preamble | MEDIUM | plan change: the command is not edited; the PROD-Q node is not run here (needs its own authorization) |
| B-07 COMPAT-X / STATE-X / EBH-5 pins on the append site and schema constant | MEDIUM | design keeps one append site and `_SCHEMA_VERSION` where possible; otherwise cond C4-C7 |
| B-08 secret gate: 0 findings at the head, 3 candidate sites outside A1 | MEDIUM | plan change: script-owned allowlist for false positives; true positives are an **open item, owner Operator**, no test asserts exit 0 over the tree (Part H, R-11) |
| B-09 8 ruff findings in 7 files, 8 files would reformat | LOW | lint-and-format commit per touched file before its first behavioural hunk (S5, step 0b) |
| B-10 undeclared overlap on `online_evaluation_registry.py`; v13 text; untracked ratchet baseline | LOW | section 9 row added; v18 noted; ratchet text corrected (section 7) |

Open (owner named): EX-D1-02 for the GCA strict path (orchestrator, S0-4, Q-SUP-02); EX-D6-03 composition (orchestrator); NT-RW-SUP-11 and 12a, held out of the merge set (Operator for Q-SUP-05, orchestrator for Q-SUP-11; blocking effect: F-D6-09 and the semantic half of F-D6-11 stay open); secret-gate true positives (Operator). Planned expected-red of existing `tests/qualification/tenant_x` pin nodes if carriers are added: owner RW-TENANT, registered by the orchestrator (section 7).
