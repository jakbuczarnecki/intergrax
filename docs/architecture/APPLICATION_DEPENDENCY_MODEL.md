# Application Dependency Model

**Status:** Canonical architecture (platform packaging)  
**Plan (1:1):** [`plan/APPLICATION_DEPENDENCY_MODEL.md`](../plan/APPLICATION_DEPENDENCY_MODEL.md)  
**Last updated:** 2026-07-23

---

## 1. Outcome

```text
Intergrax platform package (root)
├── minimal required platform dependencies
├── capability / provider extras
└── no Tier-3 application-specific dependencies

Tier-3 application package (applications/<app>/pyproject.toml)
├── depends on Intergrax (workspace)
├── selects required platform extras
├── owns application-only dependencies
└── syncs into an application-scoped dependency set

uv workspace (monorepo phase)
├── local path resolution between application and Intergrax
├── one shared root uv.lock
└── isolated sync via --project applications/<app>
```

## 2. Ownership rules

| Dependency kind | Belongs in | Rule |
|-----------------|------------|------|
| Platform base | root `[project.dependencies]` | Mandatory `intergrax/` execution path imports it |
| Platform capability | root `[project.optional-dependencies]` | Optional provider / capability under `intergrax/` |
| Application-only | `applications/<app>/pyproject.toml` | Only application code needs it |
| Agent (Tier-2) | remain source packages on `PYTHONPATH=agents/` | Not application distributions; not workspace members in this migration |

## 3. Workspace resolution

Root `pyproject.toml` declares:

```toml
[tool.uv.workspace]
members = [
  "applications/attestation_demo",
  "applications/dispute_sim_application",
  # ... every real Tier-3 application
]
```

Each application declares:

```toml
dependencies = [
  "Intergrax-ai[<selected-extras>]",
]

[tool.uv]
package = false

[tool.uv.sources]
Intergrax-ai = { workspace = true }
```

`package = false` means the application is a dependency project: source stays importable via
`PYTHONPATH=applications/` (existing host layout). Hatch does not need to wheel the application tree.

## 4. Isolated application environments

Canonical commands (repository root):

```bash
uv sync --project applications/<app>
uv run --project applications/<app> python -m <app>.host.main
```

Three isolation levels (do not conflate):

| Level | What is isolated | Current monorepo behavior |
|-------|------------------|---------------------------|
| **Declaration** | Each app owns `applications/<app>/pyproject.toml` extras / app-only deps | Yes |
| **Dependency graph** | Resolver installs only the selected project's tree | Yes (`uv export --project …`, Docker `--project`) |
| **Physical environment directory** | Separate `.venv` per application | **Not default** — one workspace root `.venv` unless `UV_PROJECT_ENVIRONMENT` points elsewhere |

Verified behavior (`uv` 0.8.x):

* `uv sync --project applications/<app>` installs **only** that application's dependency tree
  (Intergrax base + selected extras + app-only deps), not the union of all applications.
* Default environment location remains the workspace root `.venv` unless
  `UV_PROJECT_ENVIRONMENT` points elsewhere.
* Switching projects removes packages not required by the newly selected project (exact sync).
* `uv pip show` against the shared root `.venv` is **not** durable isolation evidence;
  prefer `uv export --frozen --project … --no-emit-workspace` and per-image Docker import checks.

Platform / CI gate (root only):

```bash
uv sync --extra dev-ci --frozen
```

does **not** install application-selected extras such as `integrations-slack`.

## 5. Shared lock

One root `uv.lock` covers the workspace. Do not create per-application lockfiles during the monorepo phase.

## 6. Local development

```bash
cp applications/<app>/.env.example applications/<app>/.env
uv sync --project applications/<app>
uv run --project applications/<app> python -m <app>.host.main
```

Application `.env` files remain application-owned. Root `.env.example` is platform-only.

## 7. Docker builds

Dockerfiles synchronize the **application project**:

```dockerfile
COPY pyproject.toml uv.lock ./
COPY intergrax/ ./intergrax/
COPY applications/ ./applications/   # dockerignore keeps sibling trees out; keeps */pyproject.toml
RUN uv sync --frozen --no-dev --project applications/<app>
```

Capability selection is **not** expressed as Dockerfile `--extra` flags.

## 8. CI

| Gate | Command intent |
|------|----------------|
| Platform | `uv sync --extra dev-ci --frozen` + platform tests |
| Application | `uv sync --project applications/<app>` + application tests |
| Scaffold | generate temp app → assert `pyproject.toml` → resolve |
| Lock | `uv lock --check` |

## 9. Future extraction

When an application leaves the monorepo:

1. Publish or path-pin `Intergrax-ai` (and selected extras).
2. Replace `{ workspace = true }` with a versioned index / path source.
3. Optionally introduce an application-local lockfile.

Ownership of extras and application-only deps does not change.

## 10. Migration guidance

1. Add `applications/<app>/pyproject.toml` with Intergrax + extras.
2. Register the folder in `[tool.uv.workspace].members`.
3. Point Docker / scripts / docs at `--project applications/<app>`.
4. Regenerate the shared `uv.lock`.
5. Prove isolation: app with Slack includes `slack-sdk`; app without does not.
