# Developer quality process (InterGRAX DCP)

Maps Delivery DevOS **quality ticket** to existing InterGRAX tooling.

## Operator entrypoint

```powershell
uv run python -m scripts.governance.igx_gov bootstrap   # once per clone
uv run python -m scripts.governance.igx_gov doctor      # hooks + control plane
uv run python -m scripts.governance.igx_gov ticket      # before risky push / batch close
```

Optional wrapper: `scripts/dev/igx_quality.ps1 ticket`

## What `ticket` runs

1. `verify_commit_ticket_refs.py` — commits since upstream merge-base touching product paths need `Ref: governance/tickets/T-IGX-NNN-L.md`.
2. `run_regression_gate_ci.py --profile fast` (same default as pre-push).

When the push diff touches `intergrax/runtime/`, `agents/`, or `tests/`, also run:

```powershell
uv run python -m scripts.governance.igx_gov ticket --profile smoke
```

for parity with GitHub **Regression gate** (`ci_smoke`).

## Hooks (technical enforcement)

| Hook | Behavior |
|------|----------|
| pre-commit | Ruff + diff-scoped gates ([`run_pre_commit_gate.py`](../../../scripts/ci/run_pre_commit_gate.py)) |
| pre-push | Diff-scoped gates + **ticket Ref** + regression profile ([`run_push_gate.py`](../../../scripts/ci/run_push_gate.py)) |

Install: [`ENTERPRISE_QUALITY_WORKFLOW.md`](../guides/ENTERPRISE_QUALITY_WORKFLOW.md) — `uv run python scripts/dev/install_git_hooks.py`.

## DCP rollout checklist (operator)

Complete before claiming SERVER enforcement on `development`:

1. **Hooks** — `uv run python -m scripts.governance.igx_gov bootstrap` (installs hooks + push-gate readiness).
2. **Local gate** — `uv run python scripts/dev/verify_push_gate_readiness.py` green.
3. **Governance doctor** — `uv run python -m scripts.governance.igx_gov doctor` green @ recorded SHA.
4. **Branch protection** — follow [`governance/rollout/BRANCH-PROTECTION-CHECKLIST.md`](../../../../governance/rollout/BRANCH-PROTECTION-CHECKLIST.md); apply via [`scripts/dev/apply_github_branch_protection.ps1`](../../../scripts/dev/apply_github_branch_protection.ps1) after `gh auth login`; verify with `gh api repos/:owner/:repo/branches/development/protection` (do not claim live without API proof).
5. **CI** — PR shows green `governance-fast` + smoke; treat `governance-policy-pin` as **ADVISORY** until promoted.

Track EXTERNAL_BLOCKED items in ticket checkpoint when GitHub admin auth is unavailable.

## Emergency bypass (operator only)

| Variable | Effect |
|----------|--------|
| `INTERGRAX_SKIP_PUSH_GATE` | Skips regression pre-push |
| `INTERGRAX_SKIP_TICKET_REF_GATE` | Skips Ref trailer check |
| `INTERGRAX_SKIP_PRE_COMMIT_GATE` | Skips pre-commit gate |

## Related

- [`LOCAL_PUSH_GATE.md`](../guides/LOCAL_PUSH_GATE.md)
- [`governance/tickets/README.md`](../../../../governance/tickets/README.md)
