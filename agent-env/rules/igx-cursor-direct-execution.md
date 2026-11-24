---
description: Cursor-only direct execution on development — no mandatory factory, integrator or worker worktrees
loading: cursor-always
---

# Cursor Direct Execution

<!-- igx:active:start -->
- Cursor defaults to one implementation agent: execute the approved scope, edit, test, fix and commit without mandatory subagents. In ChatGPT Direct mode, architecture, scope, approvals and acceptance are external to Cursor.
- Local commits on the current checkout are allowed with `Ref: governance/tickets/T-IGX-<NN>-<L>.md`; no AI-tool attribution trailers.
- Default branch is `development`. Do not create branches or worktrees unless the Operator explicitly requests another mode.
- Subagents, orchestrator routing, `igx-implementer` worktrees and `igx-integrator` are optional — use only when the Operator or Plan Mode asks for delegation.
- Continue multi-step instructions batch-first; do not ask for go-ahead on routine steps inside approved scope.
- In ChatGPT Direct mode, execute the complete contract; no model approval, planning, independent review or G6 is a prerequisite to candidate publication. Preserve all actual authorization, test and security controls; report only genuine execution failures after completing permitted work.
- After `uv run python -m scripts.governance.igx_gov ticket` (or equivalent gates) on a clean `development` checkout, publish with non-force `git push origin development` (fast-forward only). If GitHub rejects the push or requires a PR, report the blocker — do not bypass branch protection.
- ChatGPT Direct candidate publication: use `python <updated-development-checkout>/scripts/cursor/igx_publish_candidate.py --wave <authorized-wave-id>` from the assigned isolated worktree after commit and required local gates. The authority registers the exact branch, base, target, ticket and write set in `.cursor/publication/<wave>.json` on canonical `development` before dispatch; never author or alter that record yourself. No Grok, factory, integrator, reviewer or audit packet is required for draft publication.
- The candidate publisher permits only non-force publication to the authorized worker ref and a draft PR to the authorized target. Target advancement is not a refusal if the authorized base remains an ancestor. Direct `git push` outside the existing development path, direct `gh` mutations, ready-for-review, merge and GitHub settings remain prohibited in this mode. Independent acceptance and merge controls are unchanged.
- Use the publisher from the synchronized trusted development checkout, not from a stale train or candidate checkout. Keep the candidate worktree as the current directory. Do not copy the publisher into the product write set.
- Enterprise ticket checks, test budget, and Claude Code role policy (`role_registry.yaml`) remain authoritative for delegated workers.
- Software factory (`/igx-native-run-plan`, `igx_factory.py`) is voluntary tooling only — never required for normal Cursor work.
- Plan Mode may specify roles and delegation; that is session-local, not a global requirement.
<!-- igx:active:end -->