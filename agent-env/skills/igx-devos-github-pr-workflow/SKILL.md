---
name: igx-devos-github-pr-workflow
description: Create GitHub PR for delivery closure using gh CLI; no merge authority.
---

# GitHub PR workflow (delivery closure)

When the ticket is ready and publication is pre-authorized (otherwise escalate: forbidden action, `/intergrax-core`):

1. Confirm ticket **quality_gate** and checkpoint logged.
2. Use `gh pr create` with summary, test plan, and ticket `Ref:` paths.
3. Attach `governance/templates/PR-BATCH-CHECKLIST.md` items in PR body.
4. Do **not** merge: merge authority stays with the operator; agents only commit and prepare the PR text.

No GitHub Issues integration — tickets live in `governance/tickets/`.
