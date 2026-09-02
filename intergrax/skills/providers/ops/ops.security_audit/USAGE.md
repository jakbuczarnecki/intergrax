# `ops.security_audit`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

**Security scanning and workspace review** for agent-produced artifacts: run container/repo scans, search shadow workspace for secrets patterns, alert on violations. Use in CI gates, lab sign-off probes, and pre-release harness checks.

## How it works

1. `security.scan` invokes configured scanner (`trivy`, `semgrep`, `snyk`) via `ToolWiringContext.security_scanner`.
2. `workspace.search` greps agent drafts for policy violations.
3. `notify.send` alerts security channel on findings.
4. Complements `harness.reliability_smoke` with workspace-aware audit path.

## How to use

```python
from intergrax.skills.providers.ops.manifests import OPS_SECURITY_AUDIT

AgentContract(id="security_probe", skills=[OPS_SECURITY_AUDIT], ...)
```

Enable `security_scanner` integration + shadow workspace on host.

## What you get

| Benefit | Detail |
|---------|--------|
| **Agent output governance** | Scan code/files agents write to workspace |
| **Scanner swap** | Change Trivy→Semgrep via integration profile |
| **Alerting built-in** | Notify channel on failed scan |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `security.scan` | Image/repo security scan |
| `workspace.search` | Search agent artifacts |
| `notify.send` | Alert on violations |

## Related skills

- `harness.reliability_smoke` - includes `security.scan` in harness context
- `workspace.authoring` - produces artifacts to audit
