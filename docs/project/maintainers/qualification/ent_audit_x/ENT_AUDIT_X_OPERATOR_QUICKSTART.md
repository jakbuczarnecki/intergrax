# ENT-AUDIT-X — Operator quickstart

Run from repository root (requires `uv` on PATH).

## FAST (non-certifying)

```bat
scripts\enterprise_audit\audit.bat fast
```

```bash
./scripts/enterprise_audit/audit.sh fast
```

## FULL (certifying; long-running)

```bat
scripts\enterprise_audit\audit.bat full
```

```bash
./scripts/enterprise_audit/audit.sh full
```

Reports land under `.tmp/proof/ent-audit-x/local-<sha>-<run-id>/`.

## CHANGED-SINCE

```bat
scripts\enterprise_audit\audit.bat changed <since-sha>
```

## FREEZE

Requires accepted baseline or operator bootstrap per runbook. Local launcher does **not** auto-bootstrap.

## CI required checks

See [ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md](./ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md) (GitHub branch protection is operator-owned OD-ENT-02).
