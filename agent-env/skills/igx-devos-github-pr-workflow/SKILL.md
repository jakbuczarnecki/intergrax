---
name: igx-devos-github-pr-workflow
description: Create GitHub PR for delivery closure using gh CLI; no merge authority.
---

# GitHub PR workflow (delivery closure)

On operator request only:

1. Confirm ticket **quality_gate** and checkpoint logged.
2. Use `gh pr create` with summary, test plan, and ticket `Ref:` paths.
3. Attach `governance/templates/PR-BATCH-CHECKLIST.md` items in PR body.
4. Do **not** merge unless operator explicitly asks.

No GitHub Issues integration — tickets live in `governance/tickets/`.
