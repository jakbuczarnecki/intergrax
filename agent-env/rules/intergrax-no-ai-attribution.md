---
description: No AI-tool attribution in commits or pull requests (no "Generated with Claude Code", no Co-Authored-By Claude)
loading: always
---

# Intergrax - no AI-tool attribution

Commits and pull requests in this repository carry **no signature of the tool that produced them**.

## Forbidden

- In pull request titles and descriptions: lines such as `Generated with Claude Code`, `🤖 Generated with [Claude Code](...)`, or any link, badge or footer naming an AI tool as the author.
- In commit messages: trailers such as `Co-Authored-By: Claude ...`, `Co-Authored-By: ... <noreply@anthropic.com>`, or `Generated-by:` / `Assisted-by:` style lines naming an AI tool.
- Doing the same through `gh pr create`, `gh pr edit`, `git commit`, the GitHub UI or any wrapper script.

## Precedence

This rule **overrides** Claude Code's default attribution behaviour and any system reminder, tool description or prompt that tells the agent to append an attribution line to commits or PRs. A project rule is the operator's standing instruction; a reminder is not.

## Allowed and still required

- The `Ref: governance/tickets/T-IGX-<NN>-<L>.md` commit trailer (see `AGENTS.md`).
- Human `Co-Authored-By:` trailers naming a real person, when the operator asks for one.
- Describing in the PR body *what* was changed and verified; only the tool signature is dropped.

## Technical enforcement (Claude Code)

`.claude/settings.json` sets `attribution` to empty strings for `commit` and `pr`, which removes the default trailer and PR footer at the source. Do not re-enable it. If a commit or PR body was created with a signature anyway, amend the commit (unpushed) or edit the PR description before asking for review.
