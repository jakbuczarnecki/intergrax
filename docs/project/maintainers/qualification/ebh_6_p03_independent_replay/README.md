# EBH-6 — Protocol 03 independent replay bundle

Auditor-accessible machine exports @ **`TEST_EXECUTION_SHA`** in [`MANIFEST.json`](MANIFEST.json).

| Artifact | Contents |
| --- | --- |
| [`p03-handoff.json`](p03-handoff.json) | 133 A–J groups, 23 families, 8 L9 blockers, tier census, certification banner |
| [`pre-cert-export.json`](pre-cert-export.json) | 133 pair rows + 19 T2 bundle rows with full pytest nodeids |
| [`pytest-transcript.txt`](pytest-transcript.txt) | Full `tests/qualification/ebh_6/` verbose transcript @ manifest SHA |

**Regenerate** (after SSOT changes, on intended qual HEAD):

```powershell
git checkout development
uv run pytest tests/qualification/ebh_6/ -v --tb=no 2>&1 | Tee-Object -FilePath docs/project/maintainers/qualification/ebh_6_p03_independent_replay/pytest-transcript.txt
uv run python scripts/qualification/export_ebh6_p03_independent_replay_bundle.py
```

Reconciliation narrative: [`EBH_6_PROTOCOL_03_EXACT_FINDINGS_RECONCILIATION.md`](../EBH_6_PROTOCOL_03_EXACT_FINDINGS_RECONCILIATION.md).

**Agent forbidden:** Protocol 03 PASS · EBH-6 CLOSED · E02-AUD-01 ACCEPT.
