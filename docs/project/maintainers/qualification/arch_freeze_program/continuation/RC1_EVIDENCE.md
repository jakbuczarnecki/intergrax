# RC-1 evidence dossier

Worktree `D:/Projekty/_wt/intergrax/af-rc1`, branch `af2/rc1-continuation`. Train base `28c27cb896d02e0185b9fa02047411fa152f6469` plus RC-0 dossier commits.

No G2, FCS, M3, or ARCH_FREEZE_CLOSED claim. `OD-AF-RC-KR08` is recorded as `PENDING_OPERATOR` in `OD-AF-RC-KR08.md`. PR #680 was not merged.

## Packages executed

| Package | Status | Artifact |
| --- | --- | --- |
| EDER-DEC | done (read-only map) | `EDER_DECISION_MAP.json` — 141 rows; 34 mapped to published DP/OD; 107 without `plan_decisions_waiting` |
| KR-UNKNOWN | done (proposals only) | `KR_UNKNOWN_RC1.json` — 17 owner proposals; register not edited |
| T-AF2-CLOSED-WORLD | replay (diagnostic) | `CLOSED_WORLD_RC1_SUMMARY.json`; full output under `.tmp/` |
| T-AF2-TYP-MEASURE | BLOCKED_EVIDENCE | no tracked `typ_measure.py`; command unchanged from `RC0_EVIDENCE.md` |
| T-AF2-BND-CLOSURE | BLOCKED_EVIDENCE | untracked session script only |
| T-AF2-RW-PLAN-AUDIT | BLOCKED_EVIDENCE | separate independent session (plan J roadmap) |
| PT-G01 / PT-G02 | OPEN | draft tickets `T-IGX-046-F`, `T-IGX-046-G`; no implementation PR without plan approval |
| PT-V04 (partial) | measured, not PASS | `render_ebh7_f10_normative_matrix.py --check` → exit 1 (`STALE`) |

## Closed-world replay

```text
uv run python scripts/governance/arch_freeze_b1_closed_world_inventory.py --report-only \
  --output-dir .tmp/session/arch-freeze/continuation/rc1/closed-world
```

`strict_pass` in summary: true; `mandatory_undisposed_count`: 7; `extractor_unknown_nodes`: 21. This is not certification.

## RC-1 exit vs plan section J

| Criterion | Result |
| --- | --- |
| SCOPE nameable as valid | partial — RW-PLAN-AUDIT still missing; closed-world replay improves evidence |
| G01/G02 merged or PR | **blocked** — draft tickets only |
| KR-08 mandate yes/no | **blocked** — operator decision pending |

## Next

RC-2: #680 and `051-N` review/PR, product backlog B.2, after operator answers `OD-AF-RC-KR08` if merge is required.