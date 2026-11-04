# ENT-AUDIT-X — B6 independent handoff (B5 evidence tree)

Tracked qualification artifacts @ **TEST_EXECUTION_SHA** (see `ENT_AUDIT_X_SHA_LADDER.md` canonical row).

| Path | Role |
| --- | --- |
| `qualification_at_<TES>/l3-cli-full/` | Certifying FULL closed-world bundle + `evidence_manifest.v1.json` |
| `qualification_at_<TES>/qual_wave_report.json` | L1→L2→semantic 85→ADV31→L3 FULL orchestration record |
| `qualification_at_<TES>/publication_index.v1.json` | SHA256 index for independent replay (strict closed-world verify) |

**Historical defect (ENT-W8-B6-001):** R7 tree `qualification_at_87203c1b…` @ `d18f01f22` — preserved; see
[`R7_DEFECT_ENT-W8-B6-001.md`](./R7_DEFECT_ENT-W8-B6-001.md).

**Current corrective tree:** `qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` (W8 wave).

**Formal B6 package (draft):** [ENT_AUDIT_X_B6_FORMAL_ACCEPTANCE_PACKAGE_DRAFT.md](../ENT_AUDIT_X_B6_FORMAL_ACCEPTANCE_PACKAGE_DRAFT.md) · trust chain [ENT_AUDIT_X_B6_TRUST_CHAIN_MATRIX.md](../ENT_AUDIT_X_B6_TRUST_CHAIN_MATRIX.md)

Republish: `uv run python scripts/enterprise_audit/publish_b5_evidence_bundle.py --qualification-sha <TES>` (fails closed via `verify_b5_publication_tree()`).

**EVIDENCE_PUBLICATION_SHA** is the git commit that first lands a handoff tree (see SHA ladder); not the MANIFEST self-hash field.
