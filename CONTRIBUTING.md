# Contributing to Intergrax

Thank you for your interest in Intergrax. This document explains how to contribute effectively to the **Agent OS and Harness AI** platform.

**Canonical documentation:** [`docs/`](docs/) — navigation and update rules in [README.md — Documentation index](README.md#documentation-index).

---

## Project status

Intergrax is under **active private R&D**. The **harness platform is complete** — the default implementation queue is [§6.1 maintenance](docs/project/architecture/intergrax_runtime_architecture.md#61-harness-platform-maintenance-default--band-1) only. Business agents (Phase K) are [end of plan](docs/project/architecture/intergrax_runtime_architecture.md#63-end-of-plan--deferred-product-work-only) until explicit product prioritization.

Intergrax is **public and source-available proprietary** under the [Intergrax Evaluation and Collaboration License 1.0](LICENSE). You may clone, install locally, make evaluation modifications, create forks, submit patches, and open pull requests as described in [COLLABORATION.md](docs/project/community/COLLABORATION.md). This is **not** an open-source license grant. Production use and commercial use require **explicit written permission**.

Contributions are welcome under the evaluation and collaboration model. Read [COLLABORATION.md](docs/project/community/COLLABORATION.md) and [LICENSE](LICENSE) before starting work. Large or architectural contributions require **prior discussion** with the maintainer.

---

## Before contributing

You may clone the Official Repository, install dependencies, run evaluation paths, make private evaluation modifications, create a GitHub fork, and prepare patches or pull requests under the [Intergrax Evaluation and Collaboration License 1.0](LICENSE). Production use, commercial use, hosted services, redistribution, and incorporation into products or services require separate explicit written permission.

### Required reading

| Document | When |
|----------|------|
| [docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) | Always — strategic goal and work cycle |
| [docs/project/technical/guides/SYSTEM_INVARIANTS.md](docs/project/technical/guides/SYSTEM_INVARIANTS.md) | Always — cross-domain rules you must not break |
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
uv sync --extra dev
```

### Verify platform health

```bash
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

- **Harness maintenance** — bug fixes, regression tests, CI improvements (§6.1)
- **Documentation** — corrections and clarifications in `docs/` (one source of truth per topic)
- **Tier-2 agents** — new specialized agents following [AGENT_CREATION_GUIDE](docs/project/technical/guides/AGENT_CREATION_GUIDE.md)
- **Tier-0 plugins** — integrations, tools, skills via [EXTENSION_AUTHOR_GUIDE](docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md)
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

Tier-1/2/3 work is **composition and wiring** of existing Tier-0 modules — not parallel universal mechanisms. See [architecture §5.2](docs/project/architecture/PLATFORM_FOUNDATION.md#52-platform-reuse-and-no-redundancy-principle).

### Agent creation

- Use scaffold: `python -m intergrax.scaffold new-agent <name> --capability domain.action`
- **Do not modify `intergrax/runtime/`** for agent-specific needs
- Agents consume integrations through Nexus `ToolRuntime`, not direct SDK imports

---

## Pull request process

### 1. Fork or branch

You may create a public or private GitHub fork as an **Authorized Fork** for evaluation and contribution, or branch from the Official Repository. Other evaluators may clone and test your Authorized Fork for permitted evaluation and pull-request collaboration. An Authorized Fork is not an independent distribution and does not permit Production Use or Commercial Use.

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
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
```

---

## Reporting issues

Use [GitHub Issues](https://github.com/jakbuczarnecki/intergrax/issues) and pick the matching issue template (bug report, proof path feedback, integration proposal, design partner interest, documentation feedback). Do not include secrets or credentials in public issues.

For security vulnerabilities, see [SECURITY.md](SECURITY.md) — **do not** open public issues.

For commercial licensing, production use, or partnership requests, contact the maintainer directly (see [COLLABORATION.md](docs/project/community/COLLABORATION.md)) — do not use public issues for those topics.

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

Intergrax is source-available proprietary software under the [Intergrax Evaluation and Collaboration License 1.0](LICENSE). It is not open source. Production use and commercial use require explicit written permission.

## Contribution License

By participating in the project, you agree to the contribution terms in [LICENSE](LICENSE). The following summarizes the workflow; the LICENSE text controls.

### Code and documentation contributions

When you submit a **Code Contribution** or **Documentation Contribution** (for example, a pull request, patch, test, configuration file, documentation file, diagram, example, or tutorial clearly offered for inclusion), you confirm that:

- you authored the material or otherwise have the right to submit it;
- you retain copyright in your Code Contribution or Documentation Contribution;
- you grant Artur Czarnecki a perpetual, worldwide, non-exclusive, royalty-free, and irrevocable license to use, reproduce, modify, create derivative works from, distribute, publicly display, publicly perform to the extent applicable, sublicense, and include your Code Contribution or Documentation Contribution in Intergrax, including under current and future licensing models for the project;
- you will not submit material that you do not have the right to contribute or that is subject to an incompatible license.

Submitting material creates the license grant described in [LICENSE](LICENSE). The contribution license takes effect when the material is submitted or clearly offered for inclusion, whether or not the maintainer later accepts or merges it. Acceptance or merge is a separate maintainer decision.

Submitting a Code Contribution or Documentation Contribution does not transfer copyright to the maintainer. The maintainer may accept, reject, or request changes to any submission. Submission does not create support obligations, compensation, partnership terms, production rights, or commercial rights.

### Feedback

**Feedback** includes issue comments, suggestions, bug reports, feature proposals, and architecture observations that are not clearly offered for direct inclusion as a project file. Feedback is welcome and is governed by the simpler Feedback license in [LICENSE](LICENSE). Posting Feedback does not automatically make it a Code Contribution or Documentation Contribution, and does not transfer copyright in the literal text of your comment.

Do not submit confidential information or trade secrets as Feedback.

### Potential CLA requirement

The maintainer may require a separate Contributor License Agreement before accepting substantial, corporate, or potentially patent-relevant Code Contributions or Documentation Contributions. Lack of a separate CLA may be a reason to decline a large contribution.

### Patent-related contributions

The current contribution process does not transfer patent rights and does not grant an automatic patent license. If you know that a contribution may be covered by your patent or your employer's patent, disclose that when submitting substantial technical material. The maintainer may request a separate written agreement for potentially patentable mechanisms.

---

## Contact

- **Maintainer:** Artur Czarnecki
- **Email:** jakbu.czarnecki.83@gmail.com
- **Repository:** https://github.com/jakbuczarnecki/intergrax
