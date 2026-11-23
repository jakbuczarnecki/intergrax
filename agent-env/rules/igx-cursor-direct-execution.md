---
description: Cursor-only direct execution on development — no mandatory factory, integrator or worker worktrees
loading: cursor-always
---

# Cursor Direct Execution

<!-- igx:active:start -->
- Cursor defaults to one autonomous implementation agent: analyze, edit, test and fix in approved ticket scope without mandatory subagents.
- Local commits on the current checkout are allowed with `Ref: governance/tickets/T-IGX-<NN>-<L>.md`; no AI-tool attribution trailers.
- Default branch is `development`. Do not create branches or worktrees unless the Operator explicitly requests another mode.
- Subagents, orchestrator routing, `igx-implementer` worktrees and `igx-integrator` are optional — use only when the Operator or Plan Mode asks for delegation.
- Continue multi-step instructions batch-first; do not ask for go-ahead on routine steps inside approved scope.
- Stop only for real Operator decisions (`ESCALATION:` types in `intergrax-core`) or when contract requires independent G6 review.
- After `uv run python -m scripts.governance.igx_gov ticket` (or equivalent gates) on a clean `development` checkout, publish with non-force `git push origin development` (fast-forward only). If GitHub rejects the push or requires a PR, report the blocker — do not bypass branch protection.
- PR creation, remote merge and GitHub settings outside that publish path stay human; hooks and branch protection are not bypassed.
- Enterprise ticket checks, test budget, and Claude Code role policy (`role_registry.yaml`) remain authoritative for delegated workers.
- Software factory (`/igx-native-run-plan`, `igx_factory.py`) is voluntary tooling only — never required for normal Cursor work.
- Plan Mode may specify roles and delegation; that is session-local, not a global requirement.
- Enterprise publish batches on the main checkout: on-demand rule [`igx-cursor-enterprise-delivery.md`](igx-cursor-enterprise-delivery.md) (venv tiers, autonomous commit/push/PR when mandated).
<!-- igx:active:end -->