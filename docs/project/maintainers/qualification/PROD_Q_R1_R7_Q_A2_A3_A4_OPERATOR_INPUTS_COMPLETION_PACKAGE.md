# PROD-Q-R1-R7-Q — A2/A3/A4 operator inputs completion package

| Field | Value |
|-------|-------|
| **Disposition** | **READY FOR AUDIT — OP02/OP03 EXTERNAL DEPENDENCIES DECISION PROPOSAL** (decision preparation only; **not** provisioning · **not** CLOSED) |
| **Task** | `PROD-Q-R1-R7-Q-OP02-OP03-EXTERNAL-DEPENDENCIES-DECISION` (2026-10-22) · prior: `PROD-Q-R1-R7-Q-A2-A4-APPROVED-STATIC-CONFIGURATION` · `PROD-Q-R1-R7-Q-OP01-TENANT-WIDE-AUTHORITY-RATIFICATION` |
| **Operator authorization** | **`PROD-Q-R1-R7-Q-A2-A4-APPROVED-STATIC-CONFIGURATION`** (2026-10-22, unchanged) — **D1** identifiers, **D2** container root + named volume, **D3** gitignored `.env` injection + strict prod/context_only/unauth=false; **did not** authorize effective ALLOW at that time. **`PROD-Q-R1-R7-Q-OP01-TENANT-WIDE-AUTHORITY-RATIFICATION`** (2026-10-22) — tenant `qual-intergrax-gca`; SERVICE principal `gca-qual-strict-service`; governance bundle `qual.intergrax.gca.control_plane`; **only** `agent_distribution.admit_runtime_revision` and `agent_distribution.activate_runtime_revision`; **tenant-wide** resource scope within qual tenant (`match_action` only — **no** `resource_id` rule matching); **A5 NOT AUTHORIZED**; **no** strict serving, physical mutation, or production traffic |
| **Branch** | `development` |
| **START_HEAD** | `893619d27eafedb6a1486f00034e68a04371126b` (OP-02/OP-03 external-dependencies decision @ same HEAD) |
| **Investigation baseline** | §14 closed-world record @ `10ea294cf7b70363d1e2ea256dd8a5aae98e8861` (not re-scanned) |
| **Parent @ START_HEAD** | (see `git log -1 --format=%P` @ ratification commit) |
| **Production code delta** | **0** (documentation only) |
| **Canonical charter** | [`PROD_Q_R1_R7_Q_QUALIFICATION_ENVIRONMENT_CHARTER.md`](PROD_Q_R1_R7_Q_QUALIFICATION_ENVIRONMENT_CHARTER.md) §C–§G, §O.1–O.9 |
| **Related evidence** | [`PROD_Q_R1_R7_Q_C_PACKAGE_B_OPERATOR_PRODUCTION_PROFILES_QUALIFICATION_EVIDENCE.md`](PROD_Q_R1_R7_Q_C_PACKAGE_B_OPERATOR_PRODUCTION_PROFILES_QUALIFICATION_EVIDENCE.md) · [`PROD_Q_R1_R7_Q_QUALIFICATION_INVENTORY_AND_GATE_PLAN.md`](PROD_Q_R1_R7_Q_QUALIFICATION_INVENTORY_AND_GATE_PLAN.md) · [`PROD_Q_R1_R7_Q_SECTION_C_REMAINING_GATES_RECONCILIATION.md`](PROD_Q_R1_R7_Q_SECTION_C_REMAINING_GATES_RECONCILIATION.md) |

**Per-OP preparation status (not verification):**

| OP | Status |
|----|--------|
| **OP-01** | **TENANT-WIDE ALLOW RATIFIED (operator 2026-10-22) / CONFIGURATION-ONLY VERIFIED @ HEAD / LIVE QUAL COMPOSE REFERENCE REMAINS CONTAINED / HOST PUBLICATION & STRICT-SERVE WIRING PENDING** |
| **OP-02** | **`BLOCKED — APPROVED SOURCE BUILD REQUIRED`** (§14.1) — **operator-ratifiable delivery model §15.2** |
| **OP-03** | **Provider: `BLOCKED — APPROVED PROVIDER REQUIRED`** · **Credentials: issuance blocked** (§14.2–14.3) — **§15.3–15.4** |

**Explicitly not claimed:** global **OP-01 QUALIFIED**, full `OP-01/02/03 CONFIGURATION VERIFIED`, DIM6/DIM17 **QUALIFIED**, R7-Q **CLOSED**, PROD-Q **PASS**, global **FRZ-*** PASS delta **0**, **A5** authorized.

---

## 1. Root cause and scope closure

| Before | After (this package) |
|--------|----------------------|
| Repeated negative gates proving the same missing OP-01…03 | Contract-aligned **shapes**, safe **proposals**, **parameter matrix**, and **post-acceptance validation commands** in one SSOT |
| “Undelivered” without actionable operator checklist | **OPERATOR APPROVED — IDENTIFIER ONLY** (D1/D2/D3 static config) vs **PROVISIONING BLOCKED** security/authority items in §7 |
| Ambiguous bundle / env formats | Exact `ImmutableRuntimePolicyBundle` JSON schema, env prefixes, and loader entrypoints cited from HEAD |

**Closed-world inventory (unchanged @ HEAD):**

| # | Surface | Canonical owner / loader |
|---|---------|--------------------------|
| 1 | Governance policy | `ControlPlaneMutationAuthorizationBoundary` ← `load_package_b_operator_production_control_plane_governance_from_env` · `intergrax/applications/_shared/package_b_operator_production_governance_configuration.py` |
| 2 | Trusted artifact roots | `VenvBundleColocatedRuntimeDeploymentAdapter` + `resolve_production_artifact_root` · `INTERGRAX_SINGLE_NODE_PRODUCTION_V1_TRUSTED_ARTIFACT_PARENT_ROOTS` |
| 3 | GCA typed settings | `GovernedContractorBackendSettings.from_env` · `applications/governed_contractor_application/host/settings.py` |
| 4 | Docker qualification overlay | `applications/governed_contractor_application/deploy/docker-compose-local-qualification/` |
| 5 | This completion package | `docs/project/maintainers/qualification/` (this file) |

---

## 2. Lifecycle separation

| Phase | Who | What happens | Authorized in A2–A4? |
|-------|-----|--------------|----------------------|
| **Preparation** | Engineering + operator review | Shapes, proposals, matrix, runbook bindings | **Yes** (this task) |
| **Provisioning** | Operator | Write bundle file, materialize bundles under roots, inject secrets | **No** — requires **A2/A3/A4** acceptance of concrete values |
| **Activation** | Operator | Governance ALLOW + `deploy_and_activate` / strict serve | **No** — **A5** / charter **G.1b** |
| **Qualification** | Operator + auditor | B6/B17 replay @ `EXECUTION_SHA`, **OP-04** capture | **No** — separate authorization |

---

## 3. OP-01 — Governance policy (mutation bundle)

### 3.1 Contract summary

| Item | Value |
|------|-------|
| Semantic owner | `ControlPlaneMutationAuthorizationBoundary` |
| Loader | `load_package_b_operator_production_control_plane_governance_from_env` |
| Env prefix | `INTERGRAX_OPERATOR_PRODUCTION_` |
| Required env vars | `MUTATION_POLICY_BUNDLE_ID`, `MUTATION_POLICY_BUNDLE_PATH`, `GOVERNANCE_TENANT_ID`; optional `GOVERNANCE_SERVICE_ID` (default `gca-strict-production`) |
| Artifact schema | `runtime_policy_bundle.v1` · `ImmutableRuntimePolicyBundle` (`intergrax/contracts/runtime_policy_bundle.py`) |
| Rule matching | `match_action` == `ControlPlaneMutationRequest.mutation_type` (see `control_plane_mutation_bundle_match_input`); **no rule → DENY** |
| Forbidden bundle IDs (in-repo lab) | `reference_production.control_plane`, `harness.control_plane` |
| Fail-closed | Missing path/ID, parse error, ID mismatch, forbidden bundle_id → `PackageBOperatorProductionGovernanceConfigurationError` |

### 3.2 Qualification-scope mutation types (candidate allow list)

Reference production lab bundle admits only lifecycle mutations (not task-control harness mutations). For qualification strict activation, the **minimum** mutation types aligned with Package B lab positives are:

| `match_action` (constant) | String value | Operator must approve for qual ALLOW? |
|---------------------------|--------------|--------------------------------------|
| `MUTATION_TYPE_ADMIT_RUNTIME_REVISION` | `agent_distribution.admit_runtime_revision` | **OPERATOR DECISION REQUIRED** |
| `MUTATION_TYPE_ACTIVATE_RUNTIME_REVISION` | `agent_distribution.activate_runtime_revision` | **OPERATOR DECISION REQUIRED** |

Any additional mutation type (task cancel/resume/autonomy) is **out of scope** unless explicitly added to the approved bundle — default posture: **do not add** (implicit deny).

**Resource IDs / scopes:** Bundle rules do not filter `resource_id` today; isolation relies on **tenant** (`GOVERNANCE_TENANT_ID`), **principal** (`GOVERNANCE_SERVICE_ID` → service principal), and **finite ALLOW rules**. Operator must confirm qualification tenant has **no** production principals and **no** customer linkage.

### 3.3 Non-operative profile template (JSON shape only)

Store the **published** bundle only **outside Git**. Below is a **shape template** — `bundle_id` is operator-approved for static configuration; replace every remaining `OPERATOR DECISION REQUIRED` before publication; **do not** deploy until ALLOW rules and provenance are approved; **do not** deploy this file verbatim without publication workflow.

```json
{
  "schema_version": "runtime_policy_bundle.v1",
  "bundle_id": "qual.intergrax.gca.control_plane",
  "version": "OPERATOR DECISION REQUIRED — e.g. 2026-10-22.1",
  "issued_at": "OPERATOR DECISION REQUIRED — ISO-8601 UTC",
  "canonical_digest": "",
  "rules": [
    {
      "rule_id": "OPERATOR DECISION REQUIRED — e.g. qual.gca.admit_runtime_revision",
      "description": "Qualification-only runtime revision admission (scoped tenant)",
      "match_action": "agent_distribution.admit_runtime_revision",
      "effect": "allow"
    },
    {
      "rule_id": "OPERATOR DECISION REQUIRED — e.g. qual.gca.activate_runtime_revision",
      "description": "Qualification-only runtime revision activation (scoped tenant)",
      "match_action": "agent_distribution.activate_runtime_revision",
      "effect": "allow"
    }
  ]
}
```

After operator edits: compute digest with `ImmutableRuntimePolicyBundle.model_validate_json(...).with_canonical_digest()` and record **SHA256(file bytes)** in provenance (charter §C).

**Operator-approved qualification identifiers (static configuration only — not effective ALLOW):**

| Parameter | Approved value | Status |
|-----------|----------------|--------|
| `bundle_id` | `qual.intergrax.gca.control_plane` | **OPERATOR APPROVED — IDENTIFIER ONLY** |
| `GOVERNANCE_TENANT_ID` | `qual-intergrax-gca` (disjoint from prod/LAB) | **OPERATOR APPROVED — IDENTIFIER ONLY** |
| `GOVERNANCE_SERVICE_ID` | `gca-qual-strict-service` | **OPERATOR APPROVED — IDENTIFIER ONLY** |
| `INTERGRAX_STRICT_PRODUCTION_SERVING_APPLICATION_ENVIRONMENT_ID` | `qual-intergrax-gca-env-01` (align with tenant) | **OPERATOR APPROVED — IDENTIFIER ONLY** |
| Bundle filesystem path | e.g. `/etc/intergrax/qual/op-01/mutation-policy.json` (Linux qual host) or operator-mounted secret volume in Docker | **OPERATOR DECISION REQUIRED** (path not in static-config approval) |
| Policy `version`, `issued_at`, rule IDs, ALLOW rules | `2026-10-22.1` · `2026-10-22T11:20:00Z` · `qual.gca.admit_runtime_revision` · `qual.gca.activate_runtime_revision` | **OPERATOR RATIFIED — TENANT-WIDE ALLOW** (see §3.4; artifact **outside Git**) |
| Live Docker qual `.env` OP-01 injection | Commented / disabled post-containment | **CONTAINED** — configuration-only validation uses session path only; **not** re-enabled on active runtime reference |

**Provenance record (out-of-repo):** approver identity, UTC timestamp, qualification scope statement, file SHA256, `bundle_id`, `canonical_digest`, approved mutation list.

### 3.4 OP-01 tenant-wide authority ratification (2026-10-22)

**Authorization event (new — not backdated):** `PROD-Q-R1-R7-Q-OP01-TENANT-WIDE-AUTHORITY-RATIFICATION` records operator approval of **effective** tenant-wide mutation authority for the dedicated qualification tenant only. This is **distinct from** bundle file creation (`issued_at` above), **distinct from** containment (disabled compose reference), and **distinct from** any future host publication / strict-serve materialization.

| Item | Value |
|------|-------|
| Tenant | `qual-intergrax-gca` |
| SERVICE principal (`GOVERNANCE_SERVICE_ID`) | `gca-qual-strict-service` |
| `bundle_id` | `qual.intergrax.gca.control_plane` |
| `version` | `2026-10-22.1` |
| Permitted `match_action` values (finite — **no others**) | `agent_distribution.admit_runtime_revision`, `agent_distribution.activate_runtime_revision` |
| Resource scope | **Tenant-wide** within qual tenant: rules match **`match_action` only**; **no** `resource_id` filtering in `runtime_policy_bundle.v1` (accepted for this isolated qual tenant) |
| Cross-resource within tenant | Same-tenant requests with different `application_id:application_environment_id` **ALLOW** at policy layer — **intentional approved authority**, not a defect |
| **A5** | **NOT AUTHORIZED** |
| Session evidence path (host-local, not Git) | `.tmp/session/PROD-Q-R1-R7-Q-A2-A4/op-01/mutation-policy.json` |
| `file_sha256` (bytes) | `7092bc14fc576cbf19d0a047ef114f342c45247d34c102c2a3ce1a652b5864bb` |
| `canonical_digest` | `sha256:ccdda034edd301ae10b26b35d04312c2cf25778686d3c2d0536714680c661e10` |
| Prior provenance reconciled | `op-01/provenance.json` **matches** recomputed digests @ ratification |
| Containment analysis | `.tmp/session/PROD-Q-R1-R7-Q-A2-A4/OP01-CONTAINMENT-AND-AUTHORITY-ANALYSIS.md` |

**Provenance timeline (do not conflate):**

1. **Artifact creation** — bundle JSON `issued_at` `2026-10-22T11:20:00Z` (engineering/session; **not** prior effective operator ALLOW).
2. **Static configuration** — `PROD-Q-R1-R7-Q-A2-A4-APPROVED-STATIC-CONFIGURATION` (2026-10-22): identifiers only.
3. **Containment** — qual compose `.env` OP-01 / governance / strict-env references **disabled**; no workload mutation path via sanctioned strict serve while unset.
4. **Ratification (this event)** — operator tenant-wide ALLOW grant for two mutation types only.
5. **Configuration-only validation** — loader + synthetic `authorize_scoped_control_plane_mutation` @ HEAD; log `.tmp/session/PROD-Q-R1-R7-Q-A2-A4/op-01-ratification-validation.log` (**exit 0**). **No** physical mutation, admission, activation, or strict serving.

**Configuration-only validation matrix (@ HEAD, sanctioned loader):**

| Case | Expected | Observed |
|------|----------|----------|
| Approved admit + qual tenant/principal | ALLOW | ALLOW (`qual.gca.admit_runtime_revision`) |
| Approved activate + qual tenant/principal | ALLOW | ALLOW (`qual.gca.activate_runtime_revision`) |
| Same tenant, different app/env resource | ALLOW (tenant-wide grant) | ALLOW |
| Unapproved mutation type | DENY | DENY (`no_matching_rule`) |
| Wrong tenant principal | DENY | DENY (`tenant_authority_mismatch`) |
| Missing `INTERGRAX_OPERATOR_PRODUCTION_*` policy env | fail-closed | `PackageBOperatorProductionGovernanceConfigurationError` |

**Isolation:** Qualification overlay remains disjoint from production; Docker Desktop **not** required for this evidence (offline loader path). Re-enabling OP-01 in live qual compose **without** OP-02/03/A5 remains **out of scope** for this ratification step.

---

## 4. OP-02 — Trusted artifact roots and integrity

### 4.1 Contract summary

| Item | Value |
|------|-------|
| Env var | `INTERGRAX_SINGLE_NODE_PRODUCTION_V1_TRUSTED_ARTIFACT_PARENT_ROOTS` |
| Format | Absolute paths separated by `os.pathsep` (`:` on Linux, `;` on Windows) |
| Default if unset | `{DATA_DIR}/trusted_artifacts` |
| Production adapter | `allow_qualification_locators=false` (mandatory for qual — `package_b_strict_production_serving_command.py`) |
| Locators | Production path: `file://` only via `resolve_venv_bundle_artifact_directory`; `reference://`, `test://` (when flag false), `artifact://` → error |
| Digest | `verify_venv_bundle_artifact_bytes` + `directory_content_digest` · lock via `verify_venv_bundle_lock_matches_revision` |

### 4.2 Proposed logical layout (one qualification tenant)

**Do not** pre-create empty “trusted” trees in Git or in this task. Operator materializes after **A3** attestation.

**Docker overlay (in-container paths — matches Compose @ HEAD):**

| Item | Approved value | Static-config status |
|------|----------------|----------------------|
| Container trusted-artifact parent root | `/var/lib/intergrax/production-v1/data/trusted_artifacts` | **OPERATOR APPROVED — IDENTIFIER ONLY** |
| Docker named volume | `intergrax_gca_strict_local_qualification_data` → mount at `/var/lib/intergrax/production-v1/data` | **OPERATOR APPROVED — IDENTIFIER ONLY** |
| Filesystem provisioning under root | — | **PROVISIONING BLOCKED** (not covered by static approval) |
| Artifact materialization / digests | — | **PROVISIONING BLOCKED** |
| Integrity attestation / trust admission | — | **PROVISIONING BLOCKED** |

```text
/var/lib/intergrax/production-v1/data/          ← INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR (named volume)
  trusted_artifacts/                            ← INTERGRAX_SINGLE_NODE_PRODUCTION_V1_TRUSTED_ARTIFACT_PARENT_ROOTS
    _operator_manifest/                         ← attestation + expected digests (operator-written)
    bundles/
      <runtime_revision_id>/                    ← immutable venv bundle directory (file:// locator target)
```

**Linux VM profile v1 (host path proposal — OPERATOR DECISION REQUIRED):**

```text
/var/lib/intergrax/qual/<tenant-id>/data/trusted_artifacts/
```

**ENV example (Docker qual — inline in Compose @ HEAD; attestation still pending):**

```text
INTERGRAX_SINGLE_NODE_PRODUCTION_V1_TRUSTED_ARTIFACT_PARENT_ROOTS=/var/lib/intergrax/production-v1/data/trusted_artifacts
```

`allow_qualification_locators=false` remains mandatory (code); **not** relaxed by static-configuration approval.

### 4.3 Expected-artifacts manifest (operator-owned, out-of-repo)

| Field | Required | Notes |
|-------|----------|-------|
| `tenant_id` | Yes | Must match OP-01 / OP-03 |
| `trusted_artifact_parent_roots` | Yes | Copy of approved absolute list |
| `bundles[]` | Yes | One row per immutable bundle |
| `bundles[].revision_id` | Yes | Agent distribution revision identity |
| `bundles[].artifact_locator` | Yes | `file:///…` under approved root |
| `bundles[].directory_content_digest` | Yes | From `directory_content_digest(bundle_root)` |
| `bundles[].source_build_ref` | Yes | CI/build id or operator build record |
| `bundles[].operator_attestation` | Yes | Sign-off id + UTC time — **PENDING until A3** |

### 4.4 Canonical verification procedure (post-provisioning)

1. Confirm `allow_qualification_locators=false` on adapter ctor path (strict serve command).
2. For each manifest row: `resolve_venv_bundle_artifact_directory(locator, trusted_parent_roots=roots, allow_qualification_locators=False)`.
3. `verify_venv_bundle_artifact_bytes(root, expected_digest)`.
4. `verify_venv_bundle_lock_matches_revision(revision, runtime_lock)` per materialization record.
5. Negative: locator outside roots or `reference://` must fail before `adapter.prepare` (B17-04 lineage).

---

## 5. OP-03 — GCA credentials and injection

### 5.1 Contract summary (`GovernedContractorBackendSettings.from_env`)

| Requirement | Env / behavior |
|-------------|----------------|
| Production API mode | `GOVERNED_CONTRACTOR_BACKEND_ENV=prod` |
| Identity | `GOVERNED_CONTRACTOR_IDENTITY_SOURCE=context_only` (required in prod) |
| Unauthenticated bypass | `GOVERNED_CONTRACTOR_BACKEND_ALLOW_UNAUTHENTICATED` absent or `false` (**qualification forbids true**) |
| API keys | `GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON` **or** bootstrap triplet — **qualification must use JSON only** (bootstrap = LAB dev) |
| Mutual exclusion | Bootstrap key **xor** JSON map — both set → `ValueError` |
| JSON shape | Object: key → `{ "tenant_id": str, "user_id"?: str, "scopes"?: string[] }` |
| Strict serve extras | `GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_ID` + `GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_DIGEST` required by `package_b_strict_production_serving_command` |

### 5.2 Docker injection path (@ HEAD)

| Mechanism | Detail | Static-config status |
|-----------|--------|----------------------|
| Compose `env_file` | `intergrax-gca-docker-local-qualification.env` (optional, **gitignored**) | **OPERATOR APPROVED — IDENTIFIER ONLY** |
| Example template | `intergrax-gca-docker-local-qualification.env.example` | Repo-prepared shape |
| Gitignore | `.gitignore` → `applications/.../intergrax-gca-docker-local-qualification.env` | Confirmed gitignored |
| Inline Compose env | Profile v1 paths + `GOVERNED_CONTRACTOR_BACKEND_ENV=prod` | **OPERATOR APPROVED — IDENTIFIER ONLY** |
| `GOVERNED_CONTRACTOR_IDENTITY_SOURCE` | `context_only` (via operator `.env` at provision time) | **OPERATOR APPROVED — IDENTIFIER ONLY** |
| `GOVERNED_CONTRACTOR_BACKEND_ALLOW_UNAUTHENTICATED` | absent or `false` | **OPERATOR APPROVED — IDENTIFIER ONLY** |

**Separation (D3):** approved = configuration **mechanism** and strict **setting names/values** above. **Not** approved: credential issuer, generation, secure storage, injection of effective secrets, validation at runtime, revocation procedures execution.

**Non-secret mapping document (operator host — do not create in Git):**

```text
# intergrax-gca-docker-local-qualification.env (never commit)

INTERGRAX_OPERATOR_PRODUCTION_MUTATION_POLICY_BUNDLE_ID=qual.intergrax.gca.control_plane
INTERGRAX_OPERATOR_PRODUCTION_MUTATION_POLICY_BUNDLE_PATH=<container path to mounted OP-01 file — PROVISIONING BLOCKED>
INTERGRAX_OPERATOR_PRODUCTION_GOVERNANCE_TENANT_ID=qual-intergrax-gca
INTERGRAX_OPERATOR_PRODUCTION_GOVERNANCE_SERVICE_ID=gca-qual-strict-service

INTERGRAX_STRICT_PRODUCTION_SERVING_APPLICATION_ENVIRONMENT_ID=qual-intergrax-gca-env-01

GOVERNED_CONTRACTOR_BACKEND_ENV=prod
GOVERNED_CONTRACTOR_IDENTITY_SOURCE=context_only
# GOVERNED_CONTRACTOR_BACKEND_ALLOW_UNAUTHENTICATED unset or false

GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON=<PROVISIONING BLOCKED — issuance not approved>

GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_ID=<OPERATOR DECISION REQUIRED>
GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_DIGEST=<OPERATOR DECISION REQUIRED>
```

Mount OP-01 JSON via additional operator volume bind — **not** in repo Compose @ HEAD; operator adds bind mount at provision time only.

### 5.3 Secret handling, rotation, revocation

| Topic | Requirement |
|-------|-------------|
| Storage | OS-restricted file (mode `600`) or operator vault → env at container start |
| Minimum privilege | Process user `intergrax-gca` (profile v1) read-only on secret file |
| Rotation | Issue new key in JSON map, redeploy env, revoke old key in vault; **STOP** qual run on suspected leak |
| Revocation | Remove key from JSON + container recreate; teardown qual instance → destroy volume if policy requires |
| Logging | Never log `GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON` or key values |
| Bootstrap | **Forbidden** on qualification positive path |

---

## 6. Unified parameter matrix

| Parameter | Static config (D1–D3) | Operator decision (provisioning) | Validation before provision |
|-----------|----------------------|----------------------------------|----------------------------|
| `INTERGRAX_OPERATOR_PRODUCTION_MUTATION_POLICY_BUNDLE_ID` | **APPROVED** `qual.intergrax.gca.control_plane` | Path + published JSON | ID == JSON `bundle_id`; not forbidden lab ID |
| `INTERGRAX_OPERATOR_PRODUCTION_MUTATION_POLICY_BUNDLE_PATH` | — | **REQUIRED** | File exists; `model_validate_json` + `with_canonical_digest` |
| `INTERGRAX_OPERATOR_PRODUCTION_GOVERNANCE_TENANT_ID` | **APPROVED** `qual-intergrax-gca` | Inject via `.env` | Matches API key `tenant_id` + strict env id policy |
| `INTERGRAX_OPERATOR_PRODUCTION_GOVERNANCE_SERVICE_ID` | **APPROVED** `gca-qual-strict-service` | Inject via `.env` | Service principal only; no prod principal |
| `INTERGRAX_STRICT_PRODUCTION_SERVING_APPLICATION_ENVIRONMENT_ID` | **APPROVED** `qual-intergrax-gca-env-01` | Inject via `.env` | Non-empty; charter correlation |
| OP-01 rule: `admit_runtime_revision` ALLOW | — | **REQUIRED** | B6-01 operator path |
| OP-01 rule: `activate_runtime_revision` ALLOW | — | **REQUIRED** | B6-01 operator path |
| OP-01 `version`, `issued_at`, rule IDs, signed provenance | — | **REQUIRED** | External approval record; no invented issuance |
| OP-01 provenance SHA256 | — | **REQUIRED** | External approval record |
| `INTERGRAX_SINGLE_NODE_PRODUCTION_V1_TRUSTED_ARTIFACT_PARENT_ROOTS` | **APPROVED** `/var/lib/intergrax/production-v1/data/trusted_artifacts` | Host mapping + materialization | Ephemeral/tmp rejected by profile loader |
| Docker volume `intergrax_gca_strict_local_qualification_data` | **APPROVED** (Compose topology) | Volume lifecycle on host | Mount visible in container |
| Immutable venv bundle(s) | — | **REQUIRED** | Digest + lock verification |
| `allow_qualification_locators` | **false** (code; fixed) | N/A | Static in strict serve command |
| `GOVERNED_CONTRACTOR_BACKEND_ENV` | **APPROVED** `prod` (Compose + `.env`) | N/A | Settings load |
| `GOVERNED_CONTRACTOR_IDENTITY_SOURCE` | **APPROVED** `context_only` | Inject via `.env` | Prod guard in settings |
| `GOVERNED_CONTRACTOR_BACKEND_ALLOW_UNAUTHENTICATED` | **APPROVED** forbid `true` | Inject via `.env` | Prod fail-closed |
| Gitignored `intergrax-gca-docker-local-qualification.env` | **APPROVED** mechanism | Populate on operator host only | Never commit |
| `GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON` | — | **REQUIRED** (issuance) | Non-empty map; no bootstrap |
| `GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_*` | — | **REQUIRED** | Strict serve command gate |
| OP-01 file mount (Docker) | — | **REQUIRED** | Path visible in container |
| Bootstrap credentials absent on qual path | — | **REQUIRED** (verification) | Negative check before serve |

### 6.1 Approval / decision ledger (static configuration @ 2026-10-22)

| Decision ID | Resolved (identifier / topology / mechanism only) | Still blocked |
|-------------|---------------------------------------------------|---------------|
| **D1** | `bundle_id`, `tenant_id`, `service_id`, `application_environment_id` as in §3.3 | Effective mutation ALLOW policy; principal scope; bundle publication; `version` / `issued_at` / rule IDs |
| **D2** | Container root `/var/lib/intergrax/production-v1/data/trusted_artifacts`; volume `intergrax_gca_strict_local_qualification_data` | Host-side ownership mapping; filesystem provisioning; bundles; attestation; trust admission |
| **D3** | Gitignored `.env` injection; `prod` + `context_only` + unauth forbidden | Credential issuer; generation; storage; effective secret injection; validation; revocation execution |
| **A5** | — | Activation window (out of scope for this task) |

---

## 7. Provisioning gate — resolved vs outstanding

**Resolved by static-configuration approval (not provisioning):**

1. **D1** — `bundle_id` `qual.intergrax.gca.control_plane`; `GOVERNANCE_TENANT_ID` `qual-intergrax-gca`; `GOVERNANCE_SERVICE_ID` `gca-qual-strict-service`; `INTERGRAX_STRICT_PRODUCTION_SERVING_APPLICATION_ENVIRONMENT_ID` `qual-intergrax-gca-env-01`.
2. **D2** — container trusted-artifact parent root `/var/lib/intergrax/production-v1/data/trusted_artifacts`; Docker named volume `intergrax_gca_strict_local_qualification_data` (Compose @ HEAD).
3. **D3** — qualification isolation via existing gitignored `intergrax-gca-docker-local-qualification.env`; strict settings `GOVERNED_CONTRACTOR_BACKEND_ENV=prod`, `GOVERNED_CONTRACTOR_IDENTITY_SOURCE=context_only`, `GOVERNED_CONTRACTOR_BACKEND_ALLOW_UNAUTHENTICATED` absent or `false`.

**Still blocking full Package B operator-profile qualification (security / materialization / runtime):**

1. ~~Approve **OP-01** effective bundle~~ — **RATIFIED** (§3.4); **remaining:** operator-secured host path + Docker bind mount for **published** JSON on qual host (session copy is evidence only).
2. **OP-01** live compose injection — **intentionally disabled** until OP-02/03/A6 path is safe; ratification does **not** authorize reconnect to active strict serve.
3. **OP-02** — attest host/container root list beyond approved topology; materialize immutable venv bundle(s) + manifest digests.
4. **OP-03** — issue API key(s); select external-work provider ID and descriptor digest; populate secrets in operator `.env`.
5. Verify **no** `GOVERNED_CONTRACTOR_BACKEND_BOOTSTRAP_*` on qual host.
6. Authorize **A5** activation window separately (not part of static-configuration approval).

Until outstanding items 2–6 are satisfied, full operator-profile provisioning remains **BLOCKED** (OP-01 ratification alone does **not** authorize execution or **A5**).

---

## 8. Post-acceptance validation commands (operator / auditor)

Run @ pinned `EXECUTION_SHA` after secrets and bundles exist **only on qual host**. Do not commit outputs containing secrets.

### 8.1 OP-01 bundle file (offline)

```bash
uv run python -c "
from pathlib import Path
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle
p = Path('OPERATOR_PATH_TO_BUNDLE.json')
b = ImmutableRuntimePolicyBundle.model_validate_json(p.read_text(encoding='utf-8')).with_canonical_digest()
print('bundle_id', b.bundle_id)
print('canonical_digest', b.canonical_digest)
"
```

### 8.2 OP-01 governance loader

```bash
# Env vars set in shell from operator store — do not log values
uv run python -c "
from intergrax.applications._shared.package_b_operator_production_governance_configuration import (
    load_package_b_operator_production_control_plane_governance_from_env,
)
g = load_package_b_operator_production_control_plane_governance_from_env()
print('tenant', g.principal.tenant_id)
"
```

### 8.3 OP-03 settings (fail-closed negative — no real key in repo)

```bash
uv run python -c "
import os
os.environ['GOVERNED_CONTRACTOR_BACKEND_ENV'] = 'prod'
os.environ.pop('GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON', None)
os.environ.pop('GOVERNED_CONTRACTOR_BACKEND_BOOTSTRAP_API_KEY', None)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
try:
    GovernedContractorBackendSettings.from_env()
except ValueError as e:
    print('expected_fail_closed', type(e).__name__)
"
```

### 8.4 OP-02 / profile layout (no strict serve — charter A6 / RUNBOOK)

```bash
docker compose -f applications/governed_contractor_application/deploy/docker-compose-local-qualification/docker-compose.yml \
  -p intergrax-gca-docker-local-qualification run --rm \
  gca-strict-local-qualification --configured-verify
```

Requires operator authorization **A6** and materialized image; succeeds for layout only — **not** OP-01/03 verification.

### 8.5 Package B static gate (no 83-test replay)

```bash
uv run pytest -p no:xdist tests/qualification/prod_q/test_prod_q_r1_r7_q_package_b_strict_serving_wiring_gate.py::test_pb_s05_missing_operator_governance_configuration_fails_closed -q
```

### 8.6 Preparation doc hygiene

```bash
git diff --check
```

---

## 9. Enterprise audit matrix (@ HEAD, preparation task)

| Invariant | Result | Notes |
|-----------|--------|-------|
| Hard tier boundaries | **PASS** | Doc-only; no imports changed |
| Contracts over implementations | **PASS** | Shapes cite loaders/contracts |
| Exactly-one semantic owner | **PASS** | No new owners |
| Exactly-one composition owner | **PASS** | Strict serve / launcher unchanged |
| Strong typing | **PASS** | No new `Any` / reflection |
| Replaceability | **PASS** | Env + out-of-repo bundles |
| Tenant / secret isolation | **PASS** (design) | Proposals disjoint from prod |
| No alternate paths | **PASS** | No harness/reference for Package B |
| No shadow authorities | **PASS** | Lab bundle IDs forbidden in loader |
| Fail-closed | **PASS** | Documented loader behavior |
| Governance ≠ execution | **PASS** | Preparation ≠ ALLOW |
| Proposal ≠ permission ≠ execution | **PASS** | Static IDs ≠ ALLOW ≠ provisioned |
| No authority expansion | **PASS** | No ALLOW / credentials / bundle publication in doc delta |
| Evidence provenance | **PARTIAL** | OP-04 still post-activation |
| Regression protection | **N/A** | No code delta |

**Architecture STOP:** none identified.

---

## 10. FRZ / ECAP evidence (preparation only)

| Criterion | Preparation contribution | Qualification status |
|-----------|-------------------------|----------------------|
| **FRZ-PRD-01** | Static qual identifiers + OP-01 tenant-wide ALLOW ratified; **no** qualified strict execution | **OPEN** |
| **FRZ-PRD-03** | OP-01 configuration-only fail-closed/ALLOW matrix evidenced; unsupported live config still blocked | **OPEN** |
| **FRZ-PRD-08** | Approved gitignored `.env` mechanism; OP-01 compose reference contained; credentials not issued | **OPEN** |
| **FRZ-SEC-04** | Tenant/principal-bound policy evaluation evidenced; **A5** / live bypass paths not authorized | **OPEN** |
| **FRZ-GOV / FRZ-TEN / FRZ-SEC / FRZ-BND / FRZ-OWN / FRZ-CTR / FRZ-TYP** | Aligned with Package B evidence doc | **unchanged** |
| **ECAP-P1 / P2 / P4** | Contract traceability in §3–5; static-config evidence ref `PROD-Q-R1-R7-Q-A2-A4-APPROVED-STATIC-CONFIGURATION` | Configuration authorization only — **not** ECAP closure |

**Global FRZ PASS delta:** **0**. **DIM6** / **DIM17** remain **NOT QUALIFIED**.

---

## 11. Forbidden in A2–A4 (restatement)

- **A5** governance ALLOW and bounded activation  
- Strict serving with live traffic  
- Physical mutation / production data  
- Cloud provisioning  
- New semantic or composition owners  
- LAB policy posing as operator production policy  
- Secrets in Git or logs  
- Global FRZ PASS or DIM6/DIM17/R7-Q/PROD-Q closure  
- 83-test replay or broad diagnostic pytest  

---

## 12. Exit criteria for real provisioning (after operator acceptance)

1. Operator-approved OP-01 JSON on secured path; loader succeeds (§8.2).  
2. OP-02 manifest attested; bundles verify under approved roots (§4.4).  
3. OP-03 JSON keys issued; settings load with prod guards; strict provider pins set.  
4. `INTERGRAX_STRICT_PRODUCTION_SERVING_APPLICATION_ENVIRONMENT_ID` aligned.  
5. Docker env file populated on operator host only; compose project isolated (`-p intergrax-gca-docker-local-qualification`).  
6. **A5** + **A6** authorized; `--configured-verify` then bounded activation per charter **G.1b**.  
7. Capture **OP-04** audit chain; targeted B6/B17 replay @ `EXECUTION_SHA` (not in A2–A4).  
8. Independent auditor review @ GitHub commit — not Cursor report alone.

---

## 13. Next recommended action

Per [`PROD_Q_R1_R7_Q_QUALIFICATION_ENVIRONMENT_CHARTER.md`](PROD_Q_R1_R7_Q_QUALIFICATION_ENVIRONMENT_CHARTER.md) **§O.6–O.7**: after independent audit of OP-01 ratification commit, operator publishes bundle to **secured host path**, completes **OP-02** materialization/attestation and **OP-03** credential/provider issuance, then authorizes **A6** runtime verification and **A5** activation window; only then re-enable qual compose OP-01 injection, re-run Package B operator-profile gates, and capture **OP-04**. OP-01 tenant-wide ratification (§3.4) does **not** close §7 or PROD-Q.

---

## 14. OP-02 / OP-03 qualification materialization (@ `10ea294cf7b70363d1e2ea256dd8a5aae98e8861`)

Session evidence (redacted, not Git): `.tmp/session/PROD-Q-R1-R7-Q-A2-A4/op02-op03/` (`op02-artifact-materialization-evidence.md`, `op03-discovery-scan.json`).

### 14.1 OP-02 — artifact candidate

| Item | Finding |
|------|---------|
| Disposition | **`BLOCKED — APPROVED SOURCE BUILD REQUIRED`** (not `CANDIDATE MATERIALIZED`) |
| Sanctioned production VENV_BUNDLE materializer @ HEAD | `UnsupportedVenvBundleMaterializationAdapter` in `intergrax/agent_distribution/materialization_adapters.py` — coordinator path **does not** build immutable bundles |
| External authority (unchanged) | Operator/CI immutable `file://` directory per `PROD_Q_R1_R7_ADR_01` DEC-06; digest via `directory_content_digest` / `verify_venv_bundle_artifact_bytes` |
| OCI-only in-repo build | `scripts/build/build_application_image.py` → runtime graph context under `applications/governed_contractor_application/docker/runtime-context/` — **OCI staging**, not deployable VENV_BUNDLE `file://` artifact |
| Strict bind expectations | `VenvBundleColocatedRuntimeDeploymentAdapter(..., allow_qualification_locators=False)` in `package_b_strict_production_serving_command.py`; `resolve_production_artifact_root` rejects `reference://` |
| LAB excluded | `write_minimal_venv_bundle` (`testing_support/…`) — not used |
| Approved container root / volume | `/var/lib/intergrax/production-v1/data/trusted_artifacts` · `intergrax_gca_strict_local_qualification_data` (Compose @ HEAD) — **not populated** on audit host (Docker engine unavailable) |
| Trust / attestation | **Not designated** — no operator manifest row |

**Operator missing:** approved source build ref + SHA, runtime revision ID, materialized bundle path, directory digest, lock integrity record, attestation sign-off.

### 14.2 OP-03 — provider discovery

| Item | Finding |
|------|---------|
| Disposition | **`BLOCKED — APPROVED PROVIDER REQUIRED`** (not `REAL PROVIDER CANDIDATE DISCOVERED`) |
| Mechanism | `resolve_strict_external_work_provider_capabilities(integration, settings=…)` — `integration.discover()` then pin match on `GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_*` |
| Qualification environment | No bound non-LAB `ExternalWorkIntegration` on host; in-tree `discover()` besides tests: `DeterministicExternalWorkFake` (`gec3_deterministic_fake`) — **excluded** |
| Discovery command | **Not run** (no approved provider instance) |

**Operator missing:** real provider implementation reference, operator-approved `provider_id` + `descriptor_digest` from genuine side-effect-free `discover()`.

### 14.3 OP-03 — credentials readiness

| Item | Finding |
|------|---------|
| Disposition | **Issuance blocked** — no approved issuer/process |
| Gitignored `intergrax-gca-docker-local-qualification.env` | Present on host: `GOVERNED_CONTRACTOR_BACKEND_ENV=prod`, `GOVERNED_CONTRACTOR_IDENTITY_SOURCE=context_only` only — **no** API keys JSON, **no** OP-01 paths, **no** provider pins |
| Bootstrap on process | **Absent** (`GOVERNED_CONTRACTOR_BACKEND_BOOTSTRAP_*` = 0) |
| Schema | `GovernedContractorBackendSettings.from_env` — JSON object map; prod without keys → **ValueError** (verified) |
| Tenant binding target | `qual-intergrax-gca` (keys must carry matching `tenant_id` when issued) |

### 14.4 Consolidated disposition table

| Component | Required evidence | Disposition |
|-----------|-------------------|-------------|
| OP-01 | Ratified authority | Existing scoped evidence (§3.4); compose OP-01 injection **not** re-enabled |
| OP-02 | Real candidate + digest + source | **BLOCKED** — approved source build required |
| OP-03 provider | Real descriptor/discovery | **BLOCKED** — approved provider required |
| OP-03 credentials | Issuer + secure issuance | **BLOCKED** until operator approves issuer |
| A5 | Activation authorization | **NOT AUTHORIZED** |

---

## 15. OP-02/OP-03 — EXTERNAL DEPENDENCY DELIVERY DECISION PROPOSAL

**Task:** `PROD-Q-R1-R7-Q-OP02-OP03-EXTERNAL-DEPENDENCIES-DECISION` · **Facts SHA:** `893619d27eafedb6a1486f00034e68a04371126b` · **Investigation record:** §14 @ `10ea294cf7b70363d1e2ea256dd8a5aae98e8861` (closed world; no repeat scans).

**Disposition:** **READY FOR AUDIT — OP02/OP03 EXTERNAL DEPENDENCIES DECISION PROPOSAL** — operator ratification of **models** below; **not** authorization to build, issue secrets, run strict serve, or execute A5/A6.

### 15.1 Established facts (reuse §14; no new discovery)

| Area | Canonical contract / owner | Established blocker |
|------|---------------------------|---------------------|
| OP-02 materialization | `UnsupportedVenvBundleMaterializationAdapter` · Agent Distribution (`intergrax/agent_distribution/materialization_adapters.py`) | Coordinator does **not** produce immutable `VENV_BUNDLE`; external producer required per [`PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md`](PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md) **DEC-06** |
| OP-02 consumption | `VenvBundleColocatedRuntimeDeploymentAdapter` · `resolve_production_artifact_root` / `resolve_venv_bundle_artifact_directory` · Apps strict serve (`package_b_strict_production_serving_command.py`, `allow_qualification_locators=false`) | No approved `file://` bundle on qual host; OCI staging ≠ `VENV_BUNDLE`; LAB `write_minimal_venv_bundle` excluded |
| OP-03 provider | `ExternalWorkIntegration` · `resolve_strict_external_work_provider_capabilities` · `GovernedContractorBackendSettings.strict_external_work_provider_*` | No non-LAB provider on host; `gec3_deterministic_fake` forbidden for qual evidence |
| OP-03 credentials | `GovernedContractorBackendSettings.from_env` · gitignored `intergrax-gca-docker-local-qualification.env` | Prod profile without keys fails closed; **no** approved issuer/process |

**Stage:** `PROD-Q-R1-R7-Q` **NOT QUALIFIED** · **DIM6** / **DIM17** **NOT QUALIFIED** · global **FRZ-*** **OPEN**.

### 15.2 Recommended OP-02 solution shape (single model)

**Pattern:** **External operator/CI build → immutable `VENV_BUNDLE` directory → provenance/attestation record → transfer into approved trusted parent root(s) → existing production resolver path** (`file://` locator → digest verification → registry projection → `VenvBundleColocatedRuntimeDeploymentAdapter`).

| Constraint | Requirement |
|------------|-------------|
| Producer locus | **Outside** Agent Distribution materialization coordinator unless ADR explicitly assigns build ownership (current HEAD: **external** per DEC-06) |
| Artifact contract | Immutable on-disk `VENV_BUNDLE` per `production_registry_projection_input_bundle.py` / ADR DEC-01 + DEC-06 — **not** OCI-only substitute |
| Locator | Canonical `file://` under operator-approved `trusted_artifact_parent_roots` |
| Provenance fields (manifest) | **Operator-supplied values only** at execution time: source Git SHA, `runtime_revision_id`, dependency lock identity, `directory_content_digest` (or equivalent verified digest), materialization topology `VENV_BUNDLE` |
| Trust | Existing registry + `resolve_*` ownership — **no** second resolver or shadow trust authority |
| Qualification strictness | `allow_qualification_locators=false` **unchanged** |
| Admission | Materialization into trusted roots **only after** operator attestation sign-off (charter §D / §O.4) |

**External build — inputs (operator/CI, pre-build):**

| Input | Owner |
|-------|--------|
| Approved source repository + **exact Git SHA** (build ref) | Operator / release authority |
| Target `runtime_revision_id` + roster lock authority already admitted in qual tenant | Operator (coordination with OP-01 scope) |
| Reproducible dependency lock file(s) and build recipe (pinned tool versions) | Operator / build-system owner |
| Build environment classification (qual vs prod artifact lane) | Operator |

**External build — outputs (handoff to platform path, post-build):**

| Output | Consumer on host |
|--------|------------------|
| Immutable bundle directory tree on disk | `resolve_venv_bundle_artifact_directory(..., allow_qualification_locators=False)` |
| `file://` locator relative to approved parent root | Registry projection / revision binding |
| Content digest + lock integrity record | Attestation manifest row; `verify_venv_bundle_artifact_bytes` / projection evidence |
| Signed or operator-approved attestation record (no secrets) | Qual evidence ledger; charter §O.4 OP-02 |

**Responsibility split:**

| Role | Responsibilities |
|------|------------------|
| **Platform (unchanged)** | Resolver contracts, digest verification, strict adapter, registry projection semantics, fail-closed negatives |
| **Operator** | Source revision selection; trusted root list; attestation; secure transfer into qual volume/path; retention/teardown |
| **External CI/build owner** | Reproducible build execution; manifest/checksum generation; artifact packaging — **CI vendor not chosen in this proposal** |

**Architecture decision points:** None identified for this shape within ADR DEC-06 / existing contracts. Any proposal to make Agent Distribution the **bundle builder**, relax `allow_qualification_locators`, or substitute OCI for `VENV_BUNDLE` authority → **`STOP — ARCHITECTURE DECISION REQUIRED`**.

### 15.3 Recommended OP-03 provider solution shape (single model)

**Pattern:** **Existing typed `ExternalWorkIntegration` ← genuine provider adapter injected through sanctioned host composition** (no new registry; no discovery-time execution for qual evidence capture beyond side-effect-free `discover()`).

| Requirement | Canonical hook @ HEAD |
|-------------|----------------------|
| Contract | `intergrax/integrations/contracts/external_work.py` — `ExternalWorkIntegration` |
| Strict pin | `resolve_strict_external_work_provider_capabilities(integration, settings=…)` → match `GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_ID` + `GOVERNED_CONTRACTOR_STRICT_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_DIGEST` from **real** `discover()` descriptor |
| Composition injection | `GovernedContractorHostRuntimeComposition.external_work_integration` via `compose_governed_contractor_host_runtime(..., integration=…)` · runtime assembly `build_governed_external_work_production_runtime(integration, …)` · strict serve factory path (`create_governed_contractor_backend_app` / `factory.py`) |
| Governance ≠ execution | Policy bundle (OP-01) separate from provider effect path — unchanged |

**Provider sourcing decision (operator must ratify one branch):**

| Branch | When | Owner | Evidence |
|--------|------|-------|----------|
| **A — Operator-supplied adapter** | Vendor or internal service already exposes conforming integration | Operator integration owner | Descriptor digest from live `discover()` on qual host; contract tests against `ExternalWorkIntegration` |
| **B — New platform adapter** | No acceptable operator-owned implementation | **Platform / agents** (Tier-2) — **separate authorized child task** | New adapter module + composition wiring only through existing injection points; **not** in this task |

**Do not** select a provider by interface similarity alone; pins must match **returned** descriptor identity.

**Required contract tests (post-adapter, pre-strict-serve):** existing modules `tests/qualification/prod_q/test_prod_q_r1_strict_external_work_provider_qualification_config.py`, `tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_r3_ac3_provenance_and_capabilities.py`, Package B strict wiring gate — replay @ provisioning SHA after injection.

**External dependencies (operator disclosure):** vendor API endpoint(s), egress allowlist (charter §C), spend/credentials for vendor — **declared in child provisioning task**, not here.

**Architecture decision points:** Using `discover()` output as CONFIGURED adoption identity without contract change remains **out of scope** (TRACE-X C1). Shadow provider registry or runtime reflection-based probing → **`STOP — ARCHITECTURE DECISION REQUIRED`**.

### 15.4 Recommended OP-03 credentials issuance model

**Target profile (qualification, unchanged):**

| Setting | Value |
|---------|--------|
| Tenant | `qual-intergrax-gca` |
| `GOVERNED_CONTRACTOR_BACKEND_ENV` | `prod` |
| `GOVERNED_CONTRACTOR_IDENTITY_SOURCE` | `context_only` |
| `GOVERNED_CONTRACTOR_BACKEND_ALLOW_UNAUTHENTICATED` | **false** / unset |
| Injection | Existing gitignored `intergrax-gca-docker-local-qualification.env` (compose env_file) — **no secrets in Git** |

| Topic | Proposed model |
|-------|----------------|
| **Issuer / owner** | Operator security / qualification authority (named role in ratification) — **not** platform auto-issuance |
| **Principal binding** | API key map entries with `tenant_id` = `qual-intergrax-gca`; align SERVICE principal `gca-qual-strict-service` (OP-01) in governance context |
| **Issuance & transfer** | Out-of-band secret store → one-time secure transfer to operator workstation → populate `GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON` in gitignored env only |
| **Storage** | Host-local gitignored env + qual Docker secret injection path; access restricted to qual operators |
| **Rotation** | Operator-triggered re-issue; update env hash manifest in evidence |
| **Revocation** | Remove key from issuer + delete env entry; restart strict session if exposed |
| **Teardown** | Revoke keys; `docker compose … down -v` on isolated qual project per RUNBOOK; wipe attested artifact copies per charter §O.4 |
| **Evidence** | Env variable **name** list + cryptographic **hash** of values (or issuer attestation id) — **never** secret values in docs/logs |

**No issuance in this task.**

### 15.5 Canonical owners and interfaces (unchanged @ HEAD)

| # | Surface | Owner / entry |
|---|---------|---------------|
| 1 | Governance | `load_package_b_operator_production_control_plane_governance_from_env` |
| 2 | Trusted roots + bundle resolve | `VenvBundleColocatedRuntimeDeploymentAdapter` + `resolve_production_artifact_root` |
| 3 | GCA settings | `GovernedContractorBackendSettings.from_env` |
| 4 | External work runtime | `build_governed_external_work_production_runtime` |
| 5 | Strict provider pin | `resolve_strict_external_work_provider_capabilities` |
| 6 | Qual overlay | `docker-compose-local-qualification/` |

### 15.6 Inputs required before provisioning (gate)

| ID | Input | Ratified in |
|----|-------|-------------|
| I1 | OP-01 bundle + tenant-wide authority (existing §3.4) | Done (config); host publication pending |
| I2 | OP-02: build SHA, `runtime_revision_id`, `file://` path, digest, lock record, attestation | Operator sign-off on §15.2 |
| I3 | OP-03 provider: implementation ref + `discover()`-derived ID/digest pins | Operator sign-off on §15.3 branch A or B |
| I4 | OP-03 credentials: issuer name + issued key handles (hashes only in evidence) | Operator sign-off on §15.4 |
| I5 | Separate **A6** / **A5** authorization | Charter §O.6–O.7 |

### 15.7 Security and provenance

- Artifact integrity: directory digest + trusted parent roots only; no `test://` in strict profile.
- Provider authenticity: strict env pins match `discover()` descriptor; no LAB fake.
- Secret isolation: qualification `.env` gitignored; fail-closed prod settings without keys.
- Traceability: manifest rows bind SHA, revision id, digest, attestation id — redacted session layout per §14.

### 15.8 Enterprise invariant audit (proposal scope @ `893619d…`)

| Invariant | OP-02 proposal | OP-03 proposal |
|-----------|----------------|----------------|
| Contracts over implementations | Uses `VENV_BUNDLE` + existing resolvers | Uses `ExternalWorkIntegration` Protocol |
| Layer boundaries | Build outside AD coordinator | Adapter in agents/Tier-2 or operator supply; Apps composition only |
| Single semantic owner | AD artifact semantics; Apps consumption | Integration contract in `intergrax/`; impl replaceable |
| Single composition owner | Existing strict serve + profile v1 loader | `compose_governed_contractor_host_runtime` |
| Fail-closed / no bypass | `allow_qualification_locators=false` preserved | Prod without keys / missing pins rejected |
| Governance ≠ execution | N/A (artifact lane) | OP-01 bundle vs provider effect |
| Proposal ≠ permission | This §15 | This §15 |

**No deviation requiring immediate ADR change** if operator accepts §15.2–15.4 as stated.

### 15.9 Operator approval matrix

| Decision | Options | Default recommendation | Blocks until |
|----------|---------|----------------------|--------------|
| D-OP02-1 | Accept external CI/build + DEC-06 handoff | **Accept** | I2 delivered |
| D-OP02-2 | Trusted root paths for qual host | Use §4 / compose approved roots | Attestation recorded |
| D-OP03-1 | Provider branch A vs B | **Operator declares** (no platform default vendor) | I3 |
| D-OP03-2 | Strict provider env pins | Values from approved `discover()` | I3 |
| D-OP03-3 | Credential issuer identity | Operator security role named | I4 |
| D-EXEC | A5 / strict serve / provisioning | **Deny** until I2–I5 + independent audit of provisioning commits | Charter |

### 15.10 Next implementable child tasks (after operator ratification only)

| Child task | Scope | Preconditions |
|------------|-------|---------------|
| `PROD-Q-R1-R7-Q-OP02-EXTERNAL-VENV-BUNDLE-PROVISIONING` | Execute §15.2 build handoff, populate trusted root, manifest + attestation evidence | D-OP02-1/2 accepted |
| `PROD-Q-R1-R7-Q-OP03-PROVIDER-AND-CREDENTIALS-PROVISIONING` | Branch A wiring or authorize Branch B adapter task; issuer-run credential population; discovery pin capture | D-OP03-1/2/3 accepted |
| `PROD-Q-R1-R7-Q-A6-RUNTIME-VERIFICATION` (existing charter) | Strict serve replay, Package B gates | I2–I4 + separate A6 authorization |

### 15.11 FRZ / ECAP (decision proposal only)

| Criterion | Relevance | Status |
|-----------|-----------|--------|
| **FRZ-PRD-01** | Independent qual vs LAB | **OPEN** — external inputs still required |
| **FRZ-PRD-03** | Unsupported config fail-closed | **OPEN** — evidenced negatives; proposal preserves fail-closed |
| **FRZ-PRD-08** | Production bypass count = 0 | **OPEN** — no new bypass in proposal |
| **FRZ-SEC-04** | Auth / isolation | **OPEN** — credentials model defers to operator issuer |
| Supporting | FRZ-OWN/BND/CTR/TYP/REG/REL/REC | No PASS claimed |
| **ECAP-P1/P2/P4** | Assurance protocol | Proposal recorded; **no** execution evidence |

**Tracked freeze debt:** DIM6, DIM17, full PROD-Q-R1-R7-Q qualification — unchanged.

### 15.12 Unresolved findings (classified)

| Finding | Class |
|---------|--------|
| No immutable `file://` bundle on qual host | **IN-SCOPE BLOCKER** (until OP-02 child) |
| No approved external-work provider instance | **IN-SCOPE BLOCKER** (until OP-03 child) |
| No credential issuer authorization | **IN-SCOPE BLOCKER** (until OP-03 child) |
| Docker engine unavailable @ §14 audit | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** (re-check on provisioning host) |
| DIM6 / DIM17 not qualified | **TRACKED FREEZE DEBT** |

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
