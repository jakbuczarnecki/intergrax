# ROADMAP-REPLAY-X — SHA ladder

| Field | SHA / value |
| --- | --- |
| **START_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **AUDITED_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **IMPLEMENTATION_SHA** | `b7bf014ee03e32cd3510aa3a509eed6a25e750e2` |
| **W1 lineage** | `6fbc4a827c72e75f2c7a3374581a309a7b6beee8` (replay HEAD SSOT @ 91649e5 + STATE/TRACE remediations) |
| **TEST_EXECUTION_SHA** | `b7bf014ee03e32cd3510aa3a509eed6a25e750e2` |
| **EVIDENCE_PUBLICATION_SHA (W4 bundle)** | `8515728649a280d435b04c23c8700fbd169495f5` — committed W4 qualification narrative + dossier @ `git rev-parse HEAD` when pinned |
| **PRE_B6_RECONCILIATION_PUBLICATION_SHA** | `31c462dc7c4c4ac03a10ed69254c8822b2322910` — doc-only PRE-B6 dossier reconciliation @ W4 base `8515728649a280d435b04c23c8700fbd169495f5` |
| **INDEPENDENT_ACCEPTANCE_BASELINE_SHA** | **OPEN** (B6) |
| **OPERATOR_DECISION_SHA** | **OPEN** |
| **FINAL_PUBLICATION_SHA** | **OPEN** (W10) |

**Preflight @ PRE-B6 session:** W4 **`EVIDENCE_PUBLICATION_SHA`** `8515728649a280d435b04c23c8700fbd169495f5`; reconciliation published @ **`PRE_B6_RECONCILIATION_PUBLICATION_SHA`** `31c462dc7c4c4ac03a10ed69254c8822b2322910`; `851572864…` is ancestor of PRE-B6 commit.

**Immutable qual logs (EVIDENCE_EXPORT_REQUIRED — not in Git):** `.tmp/proof/roadmap-replay-x-immutable-6fbc4a8/` — SHA256 table in [`ROADMAP_REPLAY_X_QUALIFICATION_EVIDENCE_RECORD.md`](ROADMAP_REPLAY_X_QUALIFICATION_EVIDENCE_RECORD.md). Sanitized PRE-B6 archive: `.tmp/proof/roadmap-replay-x-pre-b6-archive-8515728/` + `MANIFEST.sha256`.
