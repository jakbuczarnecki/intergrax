# Enterprise Quality Oversight - Campaign A (EQO-A) dossier

Read-only, exact-SHA baseline audit of the Intergrax Enterprise program at `BASELINE_SHA = 258bbf6abb6fdbb92ef0185d058527fbb1e47f04`.

**This is not a certificate.** EQO-A issues no certificate, promotes no FRZ criterion, closes no stage and changes no canonical status. The independent audit (session with role `independent-quality-auditor`) issued the verdict **PASS_SCOPED on the dossier** (see `audit_replay/EQOA_INDEPENDENT_AUDIT_REPORT.md`, version 3). The verdict concerns the dossier, not the platform; the platform is not declared defect-free.

## Layout

| Path | Content |
|---|---|
| `deliverables/` | D01-D11: stage inventory, certification evidence matrix, FRZ reconciliation matrix, architecture baseline, drift matrix, audit coverage gap matrix, residual risk register, domain completeness matrix, baseline audit report, closure matrix, Campaign B/C handoff |
| root `*.md` / `*.json` | charter, source registry, authority conflict register, scope exclusions, findings register, tool run log, test execution record, read log, `BASELINE.json`, `MANIFEST.json` |
| `data/`, `findings/`, `A1`..`A9`, `extracts/`, `shards/`, `briefs/`, `tests/`, `logs/` | data sources (JSON/JSONL), sub-agent shards and briefs, tool and test logs |
| `tools/` | stdlib-only parsers, consolidators, renderer and the checker `verify_campaign_a.py` (non-product evidence tooling). Published as `*.py.txt` (byte-identical to the audited `*.py`; the repository's pre-commit ruff gate lints every `.py` under `docs/`, and this evidence tooling is not product code) |
| `audit_replay/` | the independent auditor's report, IIAC-style manifest and handoff record |
| `handoff/` | the audit handoff and the corrections log of the two corrective loops |

## Replay

Create a detached worktree at the baseline, copy this folder to a scratch directory, restore the tool names (`cd tools && for f in *.py.txt; do cp "$f" "${f%.txt}"; done`; the restored files must match the sha256 values in `MANIFEST.json`), then run `python -I tools/verify_campaign_a.py --wave W5` with `EQOA_WS=<scratch copy>`, `EQOA_WT=<worktree>` and `EQOA_REPLAY=1`. Checks C01-C15 must pass and the manifest digests (C12) must match. Files in `handoff/`, `audit_replay/` and `logs/checker/` are outside the manifest scope by design.
