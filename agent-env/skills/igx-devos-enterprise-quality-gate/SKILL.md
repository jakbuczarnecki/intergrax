---
name: igx-devos-enterprise-quality-gate
description: Enterprise quality gate — igx_gov ticket, hooks, and CI tier mapping before push.
---

# Enterprise quality gate

```powershell
uv run python -m scripts.governance.igx_gov ticket
```

Use `--profile smoke` when diff touches `intergrax/runtime/`, `agents/`, or broad tests (see DEVELOPER-QUALITY-PROCESS.md).

Emergency bypass: `INTERGRAX_SKIP_PUSH_GATE` / `INTERGRAX_SKIP_TICKET_REF_GATE` only with operator approval.
