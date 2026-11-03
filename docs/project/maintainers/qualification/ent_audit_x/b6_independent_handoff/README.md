# ENT-AUDIT-X — B6 independent handoff (B5 evidence tree)

Tracked qualification artifacts @ **TEST_EXECUTION_SHA** `b8bb88c9db1060153ec0a6a29bbb79ba16e00078`.

| Path | Role |
| --- | --- |
| `qualification_at_<TES>/l3-cli-full/` | Certifying FULL closed-world bundle + `evidence_manifest.v1.json` |
| `qualification_at_<TES>/qual_wave_report.json` | L1→L2→semantic 85→ADV31 orchestration record |
| `qualification_at_<TES>/publication_index.v1.json` | SHA256 index for independent replay |

Republish: `uv run python scripts/enterprise_audit/publish_b5_evidence_bundle.py --qualification-sha <TES>`.

**EVIDENCE_PUBLICATION_SHA** is the git commit that first lands this tree (see SHA ladder); not the MANIFEST self-hash field.
