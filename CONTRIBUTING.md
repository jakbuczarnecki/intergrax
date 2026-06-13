# Contributing to Intergrax

Thank you for your interest in Intergrax. This document explains how to contribute effectively to the **Agent OS and Harness AI** platform.

**Canonical documentation:** [`docs/`](docs/) — navigation and update rules in [README.md — Documentation index](README.md#documentation-index).

---

## Project status

Intergrax is under **active private R&D**. The **harness platform is complete** — the default implementation queue is [§6.1 maintenance](docs/intergrax_runtime_architecture.md#61-harness-platform-maintenance-default--band-1) only. Business agents (Phase K) are [end of plan](docs/intergrax_runtime_architecture.md#63-end-of-plan--deferred-product-work-only) until explicit product prioritization.

---

## Before contributing

### Required reading

| Document | When |
|----------|------|
| [docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) | Always — strategic goal and work cycle |
| [docs/intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md) | Architecture changes |
| [docs/intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md) | Status, phases, gates |
| [docs/guides/AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) | Agent or application work |
| [AGENTS.md](AGENTS.md) | AI coding agents working in this repo |

### Work cycle (mandatory for significant changes)

```text
ANALIZA
  → OCENA ARCHITEKTURY (zgodność z celem Harness AI)
  → OCENA PLANU WDROŻENIA
  → PROPOZYCJA USPRAWNIEŃ
  → AKTUALIZACJA DOKUMENTACJI (strategia → kanon → plan)
  → IMPLEMENTACJA
  → WERYFIKACJA (gate + getattr audit where harness touched)
  → WNIOSKI
```

Think as a **Harness AI architect** first, then as an engineer.

---

## Development setup

### Prerequisites

- Python 3.12
- [`uv`](https://github.com/astral-sh/uv) package manager
- Git

### Install

```bash
git clone https://github.com/jakbuczarnecki/intergrax.git
cd intergrax
uv sync --extra dev
```

### Verify platform health

```bash
uv run pytest -m gate -q
python scripts/check_harness_no_getattr.py
uv run intergrax doctor
```

### Local infrastructure (optional)

```bash
cd infra && ./manage.sh up redis qdrant postgresql
```

See [infra/README.md](infra/README.md) and [docs/guides/HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md).

---

## What to contribute

### Welcome contributions

- **Harness maintenance** — bug fixes, regression tests, CI improvements (§6.1)
- **Documentation** — corrections and clarifications in `docs/` (one source of truth per topic)
- **Tier-2 agents** — new specialized agents following [AGENT_CREATION_GUIDE](docs/guides/AGENT_CREATION_GUIDE.md)
- **Tier-0 plugins** — integrations, tools, skills via [EXTENSION_AUTHOR_GUIDE](docs/guides/EXTENSION_AUTHOR_GUIDE.md)
- **Tier-3 applications** — deployable environments following `applications/USAGE.md`
- **Test coverage** — meaningful tests for real behavior (not trivial assertions)

### Requires prior discussion

- Changes to `intergrax/runtime/` (Nexus Agent OS)
- New universal platform mechanisms (check Tier-0 reuse rule first)
- Phase K business agents
- Phase W-ADAPT (Adaptive Harness Intelligence) runtime work
- Breaking API or contract changes

Open an issue or contact the maintainer before starting large architectural changes.

---

## Architecture rules

### Four-tier dependency boundaries

```text
intergrax/       MUST NOT import from agents/ or applications/
agents/          MUST NOT import from applications/
applications/    MAY import from agents/ and intergrax/
```

### Reuse rule

Tier-1/2/3 work is **composition and wiring** of existing Tier-0 modules — not parallel universal mechanisms. See [architecture §5.2](docs/architecture/PLATFORM_FOUNDATION.md#52-platform-reuse-and-no-redundancy-principle).

### Agent creation

- Use scaffold: `python -m intergrax.scaffold new-agent <name> --capability domain.action`
- **Do not modify `intergrax/runtime/`** for agent-specific needs
- Agents consume integrations through Nexus `ToolRuntime`, not direct SDK imports

---

## Pull request process

### 1. Branch

```bash
git checkout -b feature/short-description
```

### 2. Implement with minimal scope

- Smallest correct diff
- Match existing code conventions
- No unrelated changes

### 3. Update documentation

| Change type | Update |
|-------------|--------|
| Strategy / goal | `docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md` |
| Architecture | `docs/intergrax_runtime_architecture.md` + sync plan §0 |
| Phase status | `docs/intergrax_runtime_architecture.md` |
| Agent workflow | `docs/guides/AGENT_CREATION_GUIDE.md` |
| Integration catalog | `docs/architecture/INTEGRATIONS.md` |
| Tool catalog | `docs/architecture/TOOLS.md` |
| Skill catalog | `docs/architecture/SKILLS.md` |

### 4. Verify

```bash
# Always (fast gate)
uv run pytest -m gate -q

# If harness/runtime touched
python scripts/check_harness_no_getattr.py

# If agent touched
uv run pytest agents/<agent>/tests/ -q

# Full local suite (optional)
scripts\test.bat unit
```

### 5. Open PR

Use the [pull request template](.github/PULL_REQUEST_TEMPLATE.md). Include:

- What changed and why (Harness AI alignment)
- Which docs were updated
- Test evidence (gate pass count)
- Phase/task reference from implementation plan (if applicable)

### 6. Review criteria

- Aligns with Harness AI strategic goal
- Respects tier boundaries and reuse rule
- Documentation updated in correct canon file
- Gate tests pass
- No secrets committed

---

## Testing guidelines

| Marker | Purpose | CI |
|--------|---------|-----|
| `gate` | Deterministic regression gate | Yes — always |
| `unit` | Fast unit tests | Yes |
| `integration` | Component wiring | Selective |
| `agent_os` | Agent OS acceptance | Yes (gate) |
| `network` | Real external APIs | No — local only |
| `e2e` | End-to-end flows | Rare |

```bash
uv run pytest -m gate -q          # pre-PR minimum
uv run pytest -m unit -q          # unit only
uv run pytest agents/my_agent/tests/ -q
```

---

## Code style

- Python 3.12
- Formatting/linting: `ruff` (see `pyproject.toml`)
- Type checking: `pyright`
- Copyright header on new files:

```text
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
```

---

## Reporting issues

Use [GitHub Issues](https://github.com/jakbuczarnecki/intergrax/issues) with the appropriate template:

- **Bug report** — reproducible defect
- **Feature request** — new capability (include Harness AI alignment rationale)
- **Documentation** — canon correction or gap

For security vulnerabilities, see [SECURITY.md](SECURITY.md) — **do not** open public issues.

---

## GitHub repository metadata (maintainer setup)

**Source of truth:** [`.github/repo-management/repository-metadata.json`](.github/repo-management/repository-metadata.json)

**Setup guide:** [`.github/repo-management/README.md`](.github/repo-management/README.md) (token, `.env`, sync commands, CI)

Edit the manifest for the public **description**, **homepage**, and **topics** (GitHub allows up to 20 topics). On push to `main`, the workflow [`.github/workflows/sync-repository-metadata.yml`](.github/workflows/sync-repository-metadata.yml) applies the manifest to GitHub.

**Manual sync** (from repository root):

```bash
.github/repo-management/sync-github-metadata.bat           # push to GitHub (Windows)
.github/repo-management/sync-github-metadata.bat check    # dry run only

./.github/repo-management/sync-github-metadata.sh          # push to GitHub (Linux/macOS)
./.github/repo-management/sync-github-metadata.sh check    # dry run only
```

Store `GH_TOKEN` in `.env` (see `.env.example`). The sync script loads it automatically.

Keep [`pyproject.toml`](pyproject.toml) `description` and `keywords` aligned with the manifest for PyPI-style discovery and packaging consistency.

---

## License

Intergrax is proprietary software. See [LICENSE](LICENSE). Contribution does not grant additional rights beyond what the maintainer agrees to in writing.

---

## Contact

- **Maintainer:** Artur Czarnecki
- **Email:** jakbu.czarnecki.83@gmail.com
- **Repository:** https://github.com/jakbuczarnecki/intergrax
