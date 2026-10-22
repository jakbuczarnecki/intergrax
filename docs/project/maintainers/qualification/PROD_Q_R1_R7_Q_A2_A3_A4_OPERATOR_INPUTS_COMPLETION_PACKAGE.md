# PROD-Q-R1-R7-Q — A2/A3/A4 operator inputs completion package

| Field | Value |
|-------|-------|
| **Disposition** | **READY FOR AUDIT — APPROVED STATIC CONFIGURATION ONLY** |
| **Task** | `PROD-Q-R1-R7-Q-A2-A4-APPROVED-STATIC-CONFIGURATION` — record operator-approved **static identifiers and topology** for OP-01 / OP-02 / OP-03 (**not** provisioning, ALLOW policy, activation, or qualification) |
| **Operator authorization** | `PROD-Q-R1-R7-Q-A2-A4-APPROVED-STATIC-CONFIGURATION` (2026-10-22) — **D1** identifiers, **D2** container root + named volume, **D3** gitignored `.env` injection + strict prod/context_only/unauth=false; **no** effective Governance ALLOW, bundle publication, artifact attestation, credential issuance, **A5**, strict serving, or production traffic |
| **Branch** | `development` |
| **START_HEAD** | `3b740f929badd97a586c96470dfc6c98511244bc` |
| **Parent @ START_HEAD** | `5e8e97a8b704e39037992873e4a91ad84a3299de` |
| **Production code delta** | **0** (documentation only) |
| **Canonical charter** | [`PROD_Q_R1_R7_Q_QUALIFICATION_ENVIRONMENT_CHARTER.md`](PROD_Q_R1_R7_Q_QUALIFICATION_ENVIRONMENT_CHARTER.md) §C–§G, §O.1–O.9 |
| **Related evidence** | [`PROD_Q_R1_R7_Q_C_PACKAGE_B_OPERATOR_PRODUCTION_PROFILES_QUALIFICATION_EVIDENCE.md`](PROD_Q_R1_R7_Q_C_PACKAGE_B_OPERATOR_PRODUCTION_PROFILES_QUALIFICATION_EVIDENCE.md) · [`PROD_Q_R1_R7_Q_QUALIFICATION_INVENTORY_AND_GATE_PLAN.md`](PROD_Q_R1_R7_Q_QUALIFICATION_INVENTORY_AND_GATE_PLAN.md) · [`PROD_Q_R1_R7_Q_SECTION_C_REMAINING_GATES_RECONCILIATION.md`](PROD_Q_R1_R7_Q_SECTION_C_REMAINING_GATES_RECONCILIATION.md) |

**Per-OP preparation status (not verification):**

| OP | Status |
|----|--------|
| **OP-01** | **STATIC IDENTIFIERS APPROVED / EFFECTIVE ALLOW POLICY & PUBLICATION PENDING** |
| **OP-02** | **CONTAINER ROOT & VOLUME TOPOLOGY APPROVED / PROVISIONING & ATTESTATION PENDING** |
| **OP-03** | **INJECTION MECHANISM & STRICT SETTINGS APPROVED / CREDENTIAL ISSUANCE PENDING** |

**Explicitly not claimed:** `OP-01/02/03 CONFIGURATION VERIFIED`, DIM6/DIM17 **QUALIFIED**, R7-Q **CLOSED**, PROD-Q **PASS**, global **FRZ-*** PASS delta **0**.

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
| Policy `version`, `issued_at`, rule IDs, ALLOW rules | — | **PROVISIONING BLOCKED** (publication / mutation authority not approved) |

**Provenance record (out-of-repo):** approver identity, UTC timestamp, qualification scope statement, file SHA256, `bundle_id`, `canonical_digest`, approved mutation list.

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

**Still blocking provisioning (security / authority / materialization):**

1. Approve and publish **OP-01** effective bundle: finite ALLOW mutation list, rule IDs, `version`, `issued_at`, secured path, provenance SHA256 — **outside Git**.
2. Approve **OP-01** bundle filesystem path and Docker bind mount for published JSON.
3. **OP-02** — attest host/container root list beyond approved topology; materialize immutable venv bundle(s) + manifest digests.
4. **OP-03** — issue API key(s); select external-work provider ID and descriptor digest; populate secrets in operator `.env`.
5. Verify **no** `GOVERNED_CONTRACTOR_BACKEND_BOOTSTRAP_*` on qual host.
6. Authorize **A5** activation window separately (not part of static-configuration approval).

Until outstanding items 1–5 are satisfied, status remains **PROVISIONING BLOCKED** (static configuration alone does **not** authorize execution).

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
| **FRZ-PRD-01** | Static qual identifiers recorded; execution still blocked | **OPEN** |
| **FRZ-PRD-03** | Approved container root + volume topology; attestation pending | **OPEN** |
| **FRZ-PRD-08** | Approved gitignored `.env` mechanism; credentials not issued | **OPEN** |
| **FRZ-SEC-04** | Approved prod/context_only/unauth=false posture; no secrets in Git | **OPEN** |
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

Per [`PROD_Q_R1_R7_Q_QUALIFICATION_ENVIRONMENT_CHARTER.md`](PROD_Q_R1_R7_Q_QUALIFICATION_ENVIRONMENT_CHARTER.md) **§O.6–O.7**: operator completes **outstanding** decisions in §7 (effective OP-01 policy publication, OP-02 materialization/attestation, OP-03 credential/provider issuance), provisions OP-01…03 on the qual host, then authorizes **A6** runtime verification and **A5** activation window; only then re-run Package B operator-profile gates and capture **OP-04**. Static configuration (D1–D3) does **not** close §7.

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
