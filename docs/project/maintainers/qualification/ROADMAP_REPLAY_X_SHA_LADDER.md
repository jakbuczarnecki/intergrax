# ROADMAP-REPLAY-X — SHA ladder

| Field | SHA / value |
| --- | --- |
| **START_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **AUDITED_HEAD** | `5f54ba76d68e9065256ac4597bc8e8d9e012b6d2` |
| **IMPLEMENTATION_SHA** | `b7bf014ee03e32cd3510aa3a509eed6a25e750e2` |
| **W1 lineage** | `6fbc4a827c72e75f2c7a3374581a309a7b6beee8` (replay HEAD SSOT @ 91649e5 + STATE/TRACE remediations) |
| **TEST_EXECUTION_SHA** | `b7bf014ee03e32cd3510aa3a509eed6a25e750e2` (B6 L1 integrity @ ladder pin) · historical L2/L3 @ `22fbc0e837e819fe2a068295df5f527612dcf307` |
| **B6_R2_L1_EXECUTION_SHA** | `5f54ba76d68e9065256ac4597bc8e8d9e012b6d2` — log `f87aff225ef44d7610e353ee23c056c90cc0516ac03971bd2c2109e6dfc8fda4` @ `.tmp/proof/roadmap-replay-x-b6-r2-exec-5f54ba76/` |
| **B6_R2_CORRECTION_SHA** | `6abe074eacc4be957b90a57bece0e4d6d4d102f1` — R2 Python + docs @ `development` (`5f54ba76d…` → `6abe074ea…`) |
| **B6_R2_TEST_EXECUTION_SHA** | `6abe074eacc4be957b90a57bece0e4d6d4d102f1` — L1 **32 passed**, exit **0**; log SHA256 `3b3a7706f9d5d542315ffd4dbdafaa275c79de3201f6a7395b30eaf98315218c` @ `.tmp/proof/roadmap-replay-x-b6-r2-immutable-6abe074/` |
| **EVIDENCE_PUBLICATION_SHA (W4 bundle)** | `8515728649a280d435b04c23c8700fbd169495f5` — committed W4 qualification narrative + dossier @ `git rev-parse HEAD` when pinned |
| **PRE_B6_RECONCILIATION_PUBLICATION_SHA** | `31c462dc7c4c4ac03a10ed69254c8822b2322910` — doc-only PRE-B6 dossier reconciliation @ W4 base `8515728649a280d435b04c23c8700fbd169495f5` |
| **INDEPENDENT_ACCEPTANCE_BASELINE_SHA** | `5a3a05347aaea93a8e4086fc90bc08e6d5eeaf92` |
| **OPERATOR_DECISION_SHA** | **PENDING_W10_PUBLICATION** |
| **FINAL_PUBLICATION_SHA** | **PENDING_W10_PUBLICATION** |

**Preflight @ PRE-B6 session:** W4 **`EVIDENCE_PUBLICATION_SHA`** `8515728649a280d435b04c23c8700fbd169495f5`; reconciliation published @ **`PRE_B6_RECONCILIATION_PUBLICATION_SHA`** `31c462dc7c4c4ac03a10ed69254c8822b2322910`; `851572864…` is ancestor of PRE-B6 commit.

**B6 R2 roles:** `b7bf014ee03e32cd3510aa3a509eed6a25e750e2` = Python SSOT + anti-tautology implementation; `5f54ba76d68e9065256ac4597bc8e8d9e012b6d2` = ladder **AUDITED_HEAD** + doc pin (no product delta). L2/L3 heavyweight replay remains @ immutable `22fbc0e` logs unless invalidation.

**Immutable qual logs (EVIDENCE_EXPORT_REQUIRED — not in Git):** `.tmp/proof/roadmap-replay-x-immutable-6fbc4a8/` — SHA256 table in [`ROADMAP_REPLAY_X_QUALIFICATION_EVIDENCE_RECORD.md`](ROADMAP_REPLAY_X_QUALIFICATION_EVIDENCE_RECORD.md). B6 R2 post-publication L1: `.tmp/proof/roadmap-replay-x-b6-r2-immutable-6abe074/` (preliminary @ `5f54ba76`: `.tmp/proof/roadmap-replay-x-b6-r2-exec-5f54ba76/`). Sanitized PRE-B6 archive: `.tmp/proof/roadmap-replay-x-pre-b6-archive-8515728/` + `MANIFEST.sha256`.
