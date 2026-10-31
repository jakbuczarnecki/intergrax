# EBH-7 — Operator decision worksheets (Wave A4)

**Status:** Worksheets only — **no approvals inferred**. OD-EBH7-01 scope unchanged.

| DECISION_ID | Question | Options | When needed | Blocks |
| --- | --- | --- | --- | --- |
| OD-RPL-G1 | Authorize INTEGRATIONS-2E cutover program (remaining categories) | Approve full wave / phased by category / defer | Before Wave B bulk | 13+ BLOCKED RPL rows |
| OD-RPL-G2 | FIXED substrate vs second provider (singles + external_work) | Per-concern ADR / fund sandbox | Before B2 | 7 BLOCKED rows |
| OD-RPL-VECTOR | chromadb / vector-chroma extra for operational qual | Approve dev-ci extra / Testcontainers / defer PARTIAL | Before C2 | vector_store PARTIAL |
| OD-RPL-LLM | External LLM dual qual sandboxes | Approve credentials / lab-only / defer | Before C3 | LLM PARTIAL rows |
| OD-PLUGINHOST | Minimal plugin host harness for TEN-13 / XDM-018 | Activate harness / scoped N/A with evidence | Before D2 | 2 scenario rows |
| OD-EBH7-INVOKER-RECON | Accept RETAIN reconciliation addendum | Accept / request consolidation ADR | Before final L3 | Optional |
| OD-EBH7-SCOPE-RESIDUAL | Scoped CLOSED with documented non-PROHIBITED residuals | Allow / deny | Pre B6 if residuals remain | B6 scope |
| B6-AUTH | Independent reviewer + independence attestation | Named reviewer / org process | Pre B6 | B6 |
| OD-EBH7-08 | Parent closure after B6 PASS | Sign / reject | Post B6 | Roadmap CLOSED |

**Evidence required per row:** RPL concern matrix row, Package C v2 pytest log @ TEST_EXECUTION_SHA, disposition SSOT excerpt, ECAP C5 section.
