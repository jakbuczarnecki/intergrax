# ADR-SFX-003: Scenario Studio stack, authentication and hosting

| Field | Value |
|-------|-------|
| **Status** | **Proposed** (T-IGX-037-C, SFX-P2; **not Accepted** - acceptance is OD-SFX-06) |
| **Date** | 2026-11-12 |
| **Deciders** | Operator (OD-SFX-02 stack choice; OD-SFX-05 Node toolchain; OD-SFX-06 acceptance) |
| **Reference SHA** | `eadb1c2116fe` (`origin/development`); every `path:line` below is at this SHA |
| **Related** | SFX master plan R1 §7.8, §10 W3/W5, §13, §16 (`governance/tickets/plans/SFX-MASTER-PLAN.md`; session draft `.tmp/session/sfx/SFX_MASTER_PLAN_R1.md`) · [`T-IGX-037-C`](../../../../../../governance/tickets/T-IGX-037-C.md) · [`ADR-SFX-001`](ADR-SFX-001.md) (placement, ME-1) · [`ADR-SFX-004`](ADR-SFX-004.md) (parity, CLI, MCP) · [`ADR-SEF-001`](../2026-11-05/ADR-SEF-001.md) |

## Context

OD-SFX-02 (operator, 2026-11-12) fixes the Scenario Studio stack: a React + TypeScript + Vite single-page application in the repository, under the Tier-3 host `applications/scenario_studio_application/web/`, with a FastAPI backend built on `intergrax/fastapi_core`. This ADR does not reopen that choice. It records how the choice fits the existing Tier-3 patterns, what it costs (S-1..S-7), which evidence is still missing (ME-1..ME-5), and why the alternatives were rejected.

### Verified facts at `eadb1c2116fe`

**Backend factory.**
- `create_app(config: ApiConfig)` is the single FastAPI factory (`intergrax/fastapi_core/app_factory.py:29`).
- In PROD it disables `/docs`, `/redoc` and `/openapi.json` (`app_factory.py:39-47`).
- Auth wiring accepts an explicit `config.auth_provider` override (`app_factory.py:119-121`) and the API-key provider (`app_factory.py:123-125`), and composes them (`app_factory.py:117-135`).
- `ApiConfig.cors_allow_origins` is an explicit allow-list where empty means disabled (`intergrax/fastapi_core/config.py:37-38`). `create_app` itself adds no CORS middleware; hosts do.

**JWT provider.**
- `intergrax/fastapi_core/auth/providers/jwt/` exists (`provider.py`, `verifier.py`, `extractor.py`) and is unwired: no module under `intergrax/`, `agents/` or `applications/` outside that package imports it.
- `JwtAuthProvider` defaults to `NoOpJwtVerifier` when no verifier is passed (`intergrax/fastapi_core/auth/providers/jwt/provider.py:22-27`).
- `NoOpJwtVerifier` is a "stub verifier accepting any token" (`intergrax/fastapi_core/auth/providers/jwt/verifier.py:30-40`).
- The existing identity contract is `IdentityProviderBackend.verify_token` (`intergrax/integrations/contracts/identity_provider.py:44-49`).

**Tier-3 host pattern.**
- `applications/local_workspace_application/host/factory.py` builds `ApiConfig` and calls `create_app` (`:230-237`). It applies the OpenAPI override (`:239-246`) and adds `CORSMiddleware` only when origins are configured (`:248-255`). It mounts routes through `mount_*_routes` (`:257`) and optionally couples MCP through `couple_harness_host_fastapi_with_mcp` (`:405-431`).
- `applications/legal_application/host/factory.py` follows the same pattern (`:159-168`, `:170-177`), with CORS at `:179-186` and routes at `:188`.
- Both hosts set `allow_credentials=True` and `allow_methods=["*"]`, `allow_headers=["*"]` (`legal_application/host/factory.py:183-185`; `local_workspace_application/host/factory.py:252-254`).
- No module under `intergrax/` or `applications/` uses `StaticFiles` or an SPA fallback.

**Repository tooling.**
- The root uv workspace lists application members explicitly (`pyproject.toml:109-120`); there is no glob.
- Application package discovery reads `applications/<pkg>/package.json` at the app root only (`scripts/maintenance/check_application_package.py:90`).
- The tier boundary test enumerates apps by `pyproject.toml` and scans only `*.py` files (`tests/unit/architecture/test_tier_dependency_boundaries.py:24-38`).
- The IIAC G0 inventory includes every git-tracked `.json` file (`scripts/governance/iiac_g0_inventory.py:18-22`, `:31-49`). It excludes `uv.lock` exactly as a vendor pin (`:28`, `:53-57`), and `node_modules/` only as a repository-root prefix (`:24-25`, `:51`, `:223`).
- `.gitignore` ignores `dist/` (`.gitignore:8`) but has no `node_modules` entry.
- `.github/workflows/unit-tests.yml` uses only `astral-sh/setup-uv` (for example `:97`) and has no `setup-node` step. No workflow under `.github/workflows/` installs Node.
- The repository's own note says that `pull_request` workflows run the file from the default branch (`unit-tests.yml:10-11`).

**UI rules.**
- "UI must not define the architecture" (`docs/project/architecture/satellites/PLATFORM_FOUNDATION_extended_depth.md:1198`, §8.6).
- "Prefer Simple Internal UI", with DX-MAINT-04: "product UI belongs to Tier-3 hosts or Phase K, not the DX control plane" (`docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md:563-567`, §39.7).

## Alternatives

| Variant | Summary | Disposition |
|---------|---------|-------------|
| **A - In-repo SPA in a Tier-3 host (selected, OD-SFX-02)** | React + TS + Vite under `applications/scenario_studio_application/web/`; FastAPI backend on `fastapi_core`; contract-first client generated from the Tier-2 schemas | **Selected** |
| **B - Separate frontend repository** | Studio UI in its own repo and release train, calling the API | **Rejected.** It breaks PC-1 (one schema source, generated client checked by a drift test in the same tree, ADR-SFX-004). It splits the ticket-first DevOS trail (one `Ref:` ticket per batch) and the IIAC/FRZ inventory across two repos. It adds cross-repo version skew between API v1 and the client. It also contradicts OD-SFX-02 ("in the repo"). |
| **C - Python UI (Streamlit, NiceGUI or server-rendered Jinja)** | UI rendered by Python inside the host | **Rejected.** The 12-stage Studio lifecycle (approval inbox, version history, side-by-side diff, e2e walk; §10 SFX-P10-05/06) needs a client-state UI that these frameworks support poorly. A server-rendered UI tends to merge presentation with service calls, which works against PC-2 ("doors hold no validation logic"). It would still need its own auth/session story. It contradicts OD-SFX-02. |

## Decision

Adopt **Variant A** under the following binding rules.

1. **Placement and tiers.** `web/` lives inside the Tier-3 host `applications/scenario_studio_application/`. This is product UI in a Tier-3 host, as DX-MAINT-04 requires (§39.7). The SPA is a **client of API v1 only**. It holds no domain logic, and its types are generated from the Tier-2 JSON schemas (PC-1, ADR-SFX-004), so the UI does not define the architecture (§8.6). Tier-0/1 (`intergrax/`) is not changed by this ADR.
2. **Backend.** The host factory follows the verified Tier-3 pattern: `ApiConfig` → `create_app` → `mount_*_routes`, with optional MCP coupling (ADR-SFX-004). The only new backend modules are host-local: `serving/api_v1.py`, `serving/openapi_export.py` (SFX-P10-01) and `host/spa_mount.py` (SFX-P10-07).
3. **Authentication (S-2).**
   - Studio v1 authenticates with **JWT bearer** through the existing `JwtAuthProvider`, passed as `ApiConfig.auth_provider` (`app_factory.py:119-121`). This needs no `fastapi_core` change.
   - The host **must** inject a real `JwtVerifier`. It adapts the existing `IdentityProviderBackend.verify_token` contract and introduces no new vendor integration. The host **fails closed at startup** when no verifier is configured. `NoOpJwtVerifier` is forbidden in every Studio profile, and a host test asserts this.
   - There is **no cookie session** and no server-side session store. The token is held in memory by the SPA and sent as `Authorization: Bearer`.
4. **CORS (S-2).**
   - Origins come from an explicit list in typed host settings (environment variables), through `ApiConfig.cors_allow_origins`. `*` is rejected at settings validation.
   - Because no cookies are used, the Studio host sets `allow_credentials=False`. It restricts `allow_methods` and `allow_headers` to what API v1 uses (at least `Authorization` and `Content-Type`). This is a **deliberate deviation** from the `allow_credentials=True` / `"*"` settings of the reference hosts.
   - In PROD the SPA is served same-origin from the build mount, so the CORS list may be empty.
5. **Hosting (S-3).**
   - In development, the Vite dev server proxies `/v1` (and `/mcp` when enabled) to the host.
   - In production, the host mounts the built `web/dist/` with an SPA fallback in `host/spa_mount.py`. The fallback never shadows `/v1`, `/mcp` or `/health`.
   - No change to `intergrax/fastapi_core` is permitted for SPA hosting.
6. **OpenAPI (S-4).** OpenAPI stays disabled in PROD (`app_factory.py:44-46`). The schema is exported at build time by `serving/openapi_export.py`. The client is generated from that file and from the Tier-2 schemas. The CI checks drift between the export and the committed client.
7. **Node CI (S-1).**
   - A separate, path-filtered workflow `.github/workflows/scenario-studio-web.yml` (SFX-P10-04) runs lint, typecheck, unit tests, build, e2e and `npm audit`. It triggers only on `applications/scenario_studio_application/web/**` and on the workflow file itself.
   - `unit-tests.yml` stays Node-free.
   - Workflow edits are governed paths, and the Node version, package manager and audit policy are **OD-SFX-05**.
   - Per `unit-tests.yml:10-11`, PR runs of a new workflow take effect only after it reaches the default branch. The rollout must account for this.
8. **Repository hygiene (S-5, S-6, S-7), delivered by SFX-P5-06.**
   - **S-5.** Add `node_modules/` to `.gitignore`. Commit a lockfile (`web/package-lock.json`); §13 requires one for `web/`.
   - **S-5, IIAC.** The IIAC G0 inventory will enumerate the lockfile as a tracked `.json` file (`iiac_g0_inventory.py:31-49`). Its treatment is decided under OD-SFX-05 / OD-SFX-09:
     - either it is excluded like `uv.lock` (`:53-57`), which is a `scripts/` change under C-AF-01 and is not in the SFX-P5-06 file list;
     - or it is accepted as an enumerated neutral row.
   - **S-6.** Keep the npm manifest strictly at `web/package.json`. The app-root `package.json` remains the `ApplicationPackage` read by `check_application_package.py:90`. A test asserts that `web/` is never treated as an application package.
   - **S-7.** Add `applications/scenario_studio_application` to the explicit workspace member list (`pyproject.toml:111-120`).

### Consequences S-1..S-7

| ID | Consequence | Handling in this ADR | Task / decision |
|----|-------------|----------------------|-----------------|
| S-1 | No Node in CI | Separate path-filtered `scenario-studio-web.yml`; `unit-tests.yml` unchanged | SFX-P10-04; OD-SFX-05 |
| S-2 | No browser auth wired | JWT bearer through `JwtAuthProvider` plus a real verifier (fail closed, no NoOp); no cookies; explicit CORS, no `*`, `allow_credentials=False` | SFX-P10-01 |
| S-3 | No `StaticFiles` / SPA fallback | Vite dev proxy; build mount in `host/spa_mount.py`; no `fastapi_core` change | SFX-P10-07 |
| S-4 | OpenAPI disabled in PROD | Build-time export plus drift check | SFX-P10-01 |
| S-5 | `.gitignore` lacks `node_modules/`; lockfile enters IIAC G0 | `.gitignore` entry; lockfile committed; IIAC treatment decided | SFX-P5-06; OD-SFX-05; OD-SFX-09 |
| S-6 | `package.json` name collision | npm manifest only under `web/`; discovery-exclusion test | SFX-P5-06 |
| S-7 | Workspace member needed | Explicit member entry | SFX-P5-06 |

## Consequences

### Positive

- One repository, one ticket trail and one schema source for API, client and MCP (PC-1).
- No Tier-0/1 change. All Studio code is Tier-3 (host and `web/`) or Tier-2 (service), within the boundaries enforced by `test_tier_dependency_boundaries.py`.
- The UI is swappable: it consumes API v1 only and owns no semantics.

### Negative

- A second toolchain (Node, npm, Playwright) with its own supply-chain surface: `npm audit` and an allowlist (SFX-P9-03, OD-SFX-14).
- New governed CI surface and a rollout caveat for new workflows.
- A real JWT verifier must be composed per environment; until then Studio cannot run outside LAB.
- Tension with §39.7 ("Do not build a polished SaaS frontend at this stage"). It is resolved only if DX-MAINT-04's Tier-3 carve-out covers Studio (ME-3).

### Missing evidence

| ID | Question | Owner | Closing task |
|----|----------|-------|--------------|
| ME-1 | Is a new `applications/<x>` after ARCH-FREEZE Class A ("new domain module attached via contracts")? | Architect (ADR-SFX-001 review); operator at OD-SFX-06 | ADR-SFX-001 review, before GATE-SFX-ENTRY |
| ME-2 | Does adopting a JS toolchain need its own ADR? | Operator | OD-SFX-06. Default: this ADR covers it. If the operator rejects that, a dedicated toolchain ADR is created before SFX-P10-04. |
| ME-3 | The Phase K definition referenced by DX-MAINT-04 (§39.7) | Architect | SFX-P2 gap consolidation (`SFX_GAP_ANALYSIS.md`); confirmed in the OD-SFX-06 review |
| ME-4 | Do shipped-product manifest checks require a catalog entry for the new app? | Implementer | SFX-P5-06 (host build plus `check_application_package` run) |
| ME-5 | Do the Node toolchain and Playwright work on the original Windows environment (C10/C15 restriction)? | Tester | SFX-P10-04 (local Windows run recorded under `.tmp/proof/`), decided with OD-SFX-05 |

### Risks

- **R-003-01 NoOp verifier.** The default verifier accepts any token. Mitigation: fail-closed startup plus a host test, under Decision 3.
- **R-003-02 Credentialed CORS copied from the reference hosts.** Mitigation: a settings validator rejects `*`, and a host test asserts `allow_credentials=False`.
- **R-003-03 Nested `node_modules/` not covered by the IIAC root-prefix exclusion** (`iiac_g0_inventory.py:223`). It is harmless only while the directory is untracked. Mitigation: the `.gitignore` entry, plus a pre-commit or CI check that no `node_modules` path is tracked.

## Related

- Compliance: tier boundaries unchanged; no secrets or production endpoints in source; env-sourced origins and IdP settings, failing closed when missing (env/vendor qualification rule); no Docker migration and no new vendor integration (C10/C15).
- Revision triggers: any proposal to change `intergrax/fastapi_core` for Studio; cookie or OIDC session auth; a Python or remote UI; a monorepo-wide Node workflow.
- Verification when implemented: host tests (CORS, verifier fail-closed, SPA fallback precedence), OpenAPI drift, `check_application_package`, `test_tier_dependency_boundaries`, the `scenario-studio-web.yml` CI run.
- ADR status stays **Proposed** until OD-SFX-06.
