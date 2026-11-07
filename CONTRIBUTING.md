# Contributing to Intergrax

Thank you for your interest in Intergrax. This document explains how to contribute effectively to the **Agent OS and Harness AI** platform.

**Canonical documentation:** [`docs/`](docs/) - navigation and update rules in [DOCUMENTATION_MAP.md](docs/project/technical/DOCUMENTATION_MAP.md).

---

## Project status

Intergrax is **proprietary**, **source-visible**, and under **active R&D**. No general license is granted. Platform maturity is **capability-specific** - see current public evidence in [docs/project/proofs/PROOFS.md](docs/project/proofs/PROOFS.md). The default harness implementation queue is [§6.1 maintenance](docs/project/architecture/intergrax_runtime_architecture.md#61-harness-platform-maintenance-default--band-1) only. Business agents (Phase K) are [end of plan](docs/project/architecture/intergrax_runtime_architecture.md#63-end-of-plan--deferred-product-work-only) until explicit product prioritization.

Intergrax is **proprietary, source-visible** software under [LICENSE](LICENSE). Public access is not permission to run, modify, develop software based on the protected Intergrax code, redistribute, sublicense, host, or commercially use materials owned by the Licensor. No open-source license is granted. GitHub's Terms of Service may govern viewing and forking through the Service. A fork or clone does not by itself permit deployment, operation, commercialization, or distribution outside those platform permissions, a prior license for that copy, a separate written agreement, or rights that applicable law does not allow to be restricted. Submitting a pull request or merging code does not transfer copyright.

Read [LICENSE](LICENSE) and [docs/project/community/COLLABORATION.md](docs/project/community/COLLABORATION.md) before offering any material. Opening an issue or pull request does not license protected Intergrax materials and does not transfer copyright in them. Any contribution license requires a separate written agreement. Large or architectural submissions require **prior discussion** with the maintainer. Discussion is not authorization to install, run, or modify the software.

---

## Before contributing

This document does not grant permission to clone beyond GitHub's Terms of Service, install dependencies, run the software, modify it, or prepare a fork for execution. Evaluation, production use, commercial use, hosted services, redistribution, and incorporation into products or services require separate explicit written permission from the Licensor, unless an earlier license covers that copy or applicable law does not allow the restriction.

### Required reading

| Document | When |
|----------|------|
| [docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) | Always - strategic goal and work cycle |
| [docs/project/technical/guides/SYSTEM_INVARIANTS.md](docs/project/technical/guides/SYSTEM_INVARIANTS.md) | Always - cross-domain rules you must not break |
| [docs/project/technical/guides/LAYER_COMPLETION_MODE.md](docs/project/technical/guides/LAYER_COMPLETION_MODE.md) | Full harness layer closeout sessions |
| [docs/project/architecture/intergrax_runtime_architecture.md](docs/project/architecture/intergrax_runtime_architecture.md) | Architecture changes |
| [docs/project/architecture/intergrax_runtime_architecture.md](docs/project/architecture/intergrax_runtime_architecture.md) | Status, phases, gates |
| [docs/project/technical/guides/AGENT_CREATION_GUIDE.md](docs/project/technical/guides/AGENT_CREATION_GUIDE.md) | Agent or application work |
| [AGENTS.md](AGENTS.md) | AI coding agents working in this repo |

### Work cycle (mandatory for significant changes)

```text
ANALYSIS
  → ARCHITECTURE REVIEW (Harness AI goal alignment)
  → PLAN REVIEW
  → IMPROVEMENT PROPOSAL
  → DOCUMENTATION UPDATE (strategy → canon → plan)
  → IMPLEMENTATION
  → VERIFICATION (gate + getattr audit where harness touched)
  → CONCLUSIONS
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
uv sync
```

The normal repository environment provides the baseline development tools:
`pytest`, `ruff`, and `pyright`. Use the optional `dev` extra only when
heavyweight development integrations are needed:

```bash
uv sync --extra dev
```

### Verify the baseline toolchain

```bash
uv run pytest --version
uv run ruff --version
uv run pyright --version
```

### Verify platform health (optional development integrations)

```bash
uv sync --extra dev
uv run pytest -m gate -q
python scripts/maintenance/check_harness_no_getattr.py
uv run intergrax doctor
```

### Local infrastructure (optional)

```bash
cd infra && ./manage.sh up redis qdrant postgresql
```

See [infra/README.md](infra/README.md) and [docs/project/technical/guides/HARNESS_ENVIRONMENT.md](docs/project/technical/guides/HARNESS_ENVIRONMENT.md).

---

## What to contribute

### Welcome contributions

- **Harness maintenance** - bug fixes, regression tests, CI improvements (§6.1)
- **Documentation** - corrections and clarifications in `docs/` (one source of truth per topic)
- **Tier-2 agents** - new specialized agents following [AGENT_CREATION_GUIDE](docs/project/technical/guides/AGENT_CREATION_GUIDE.md)
- **Tier-0 plugins** - integrations, tools, skills via [EXTENSION_AUTHOR_GUIDE](docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md)
- **Tier-3 applications** - deployable environments following `applications/USAGE.md`
- **Test coverage** - meaningful tests for real behavior (not trivial assertions)

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

Tier-1/2/3 work is **composition and wiring** of existing Tier-0 modules - not parallel universal mechanisms. See [architecture §5.2](docs/project/architecture/PLATFORM_FOUNDATION.md#52-platform-reuse-and-no-redundancy-principle).

### Agent creation

- Use scaffold: `python -m intergrax.scaffold new-agent <name> --capability domain.action`
- **Do not modify `intergrax/runtime/`** for agent-specific needs
- Agents consume integrations through Nexus `ToolRuntime`, not direct SDK imports

---

## Pull request process

### 1. Fork or branch

A GitHub fork is subject to the GitHub Terms of Service. It does not grant permission to run, test, modify, host, or redistribute Intergrax, and other people may not treat that fork as permission to do so. Permission to run or modify the software has to come from a separate written agreement, an earlier license that covers that copy, or applicable law. Opening a pull request does not transfer copyright.

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
| Strategy / goal | `docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md` |
| Architecture | `docs/project/architecture/intergrax_runtime_architecture.md` + sync plan §0 |
| Phase status | `docs/project/architecture/intergrax_runtime_architecture.md` |
| Agent workflow | `docs/project/technical/guides/AGENT_CREATION_GUIDE.md` |
| Integration catalog | `docs/project/architecture/INTEGRATIONS.md` |
| Tool catalog | `docs/project/architecture/TOOLS.md` |
| Skill catalog | `docs/project/architecture/SKILLS.md` |

### 4. Verify

```bash
# Always (fast gate)
uv run pytest -m gate -q

# If harness/runtime touched
python scripts/maintenance/check_harness_no_getattr.py

# If agent touched
uv run pytest agents/<agent>/tests/ -q

# Full local suite (optional)
scripts\ci\test.bat unit
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
| `gate` | Deterministic regression gate | Yes - always |
| `unit` | Fast unit tests | Yes |
| `integration` | Component wiring | Selective |
| `agent_os` | Agent OS acceptance | Yes (gate) |
| `network` | Real external APIs | No - local only |
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
Intergrax is proprietary, source-visible software. No general use license is granted.
See LICENSE. Third-party components and contributor-owned materials remain under their own rights.
```

---

## Reporting issues

Use [GitHub Issues](https://github.com/jakbuczarnecki/intergrax/issues) and pick the matching issue template (bug report, proof path feedback, integration proposal, design partner interest, documentation feedback). Do not include secrets or credentials in public issues.

For security vulnerabilities, see [SECURITY.md](SECURITY.md) - **do not** open public issues.

For commercial licensing, production use, or partnership requests, contact the maintainer directly (see [docs/project/community/COLLABORATION.md](docs/project/community/COLLABORATION.md)) - do not use public issues for those topics.

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

Intergrax is proprietary, source-visible software under [LICENSE](LICENSE). No open-source license is granted. Public access does not grant a general right to run, modify, develop software based on the protected Intergrax code, redistribute, sublicense, host, or commercially use it.

## Contributions and feedback

Opening an issue, discussion, or pull request does not license protected Intergrax materials and does not transfer copyright in them. Merging code does not transfer copyright.

This document does not grant the maintainer a license in contributor-authored code, documentation, or feedback. Any contribution license requires a separate written agreement. Do not submit confidential information or trade secrets.

Material submitted while the Intergrax Evaluation and Collaboration License 1.0 was the published LICENSE stays under that earlier text, for a copy or submission obtained under it, together with applicable law. This page leaves those earlier grants in place.

Patent rights are not granted by a submission. The maintainer may decline material until a separate written agreement exists. This file does not set out that agreement.

---

## Contact

- **Maintainer:** Artur Czarnecki
- **Email:** jakbu.czarnecki.83@gmail.com
- **Repository:** https://github.com/jakbuczarnecki/intergrax
