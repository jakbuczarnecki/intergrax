# MCP-SERVER-X — SHA ladder

| Role | SHA | Notes |
| --- | --- | --- |
| BASELINE_SHA | `c228799785db8f35826f47ba0790d6a21d97fbe3` | Pin `origin/development` @ campaign start |
| ROADMAP_AMENDMENT_SHA | `bd5d04d1a3244ccb9c4464007b147dc54fc0cfd6` | PR #450 merge @ `origin/development` |
| P1_PUBLICATION_SHA | `fb7ccd7807b570e4afd75dfcbb4ca2a6188c61ad` | PR #451 merge |
| P2_IMPLEMENTATION_SHA | `e21453caf1bba45f94e21781f4872fb1ac58f806` | PR #452 |
| P2_PUBLICATION_SHA | `d46d9a0de316fea7b77f76c847028c9c932a9f3f` | PR #452 merge |
| P3_IMPLEMENTATION_SHA | `bee38a76b415cd4ca701fcc592e41b1f56e37f59` | PR #453 |
| P3_PUBLICATION_SHA | `4385c5e975492d0ee0d0049bd1c2af1ff785ea68` | PR #453 merge |
| P4_IMPLEMENTATION_SHA | `a21a6629a6f8aa25519d4e25ac0004b38426669b` | PR #454 |
| P4_MERGE_SHA | `7562a0692d7f33ce4423baf76988ced008e870fa` | PR #454 merge |
| P4_CORRECTIVE_PRODUCT_SHA | `27e19f1feb31522c3103e6fce6f15a8c9395f380` | F-MCP-ACT-N11-001 |
| P4_PUBLICATION_SHA | `48181df7f99a0f9840e699dd774532672d4ffff3` | PR #455 merge |
| P4_TERMINAL_PUBLICATION_SHA | `7930c256fdbe6bb878d9493670361d3cbe4b07bb` | P4 terminal |
| P5_BASELINE_SHA | `7930c256fdbe6bb878d9493670361d3cbe4b07bb` | T-IGX-035-F cut |
| P6_BASELINE_SHA | `e48106740a4a243204cadccb6a85db9a8710d6af` | T-IGX-035-G cut |
| P6_IMPLEMENTATION_SHA | `5ff8606b089828b334ebac133518ba47c9f6f7ce` | Product batch |
| P6_MERGE_SHA | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` | PR #459 product merge. Relabelled: this value was previously recorded as `P6_PUBLICATION_SHA` (conflated with the P4 convention `P4_MERGE_SHA` vs `P4_PUBLICATION_SHA`); value unchanged. |
| P6_PUBLICATION_SHA | `66dcd1bfe260b33a41881c7b9911b2a2abb752f9` | Merge commit of the activation PR (branch `mcp-p6-gov-signing`); not PR #460 — see `P6_RECOVERY_EVIDENCE_PR460_MERGE_SHA` |
| P6_RECOVERY_EVIDENCE_PR460_MERGE_SHA | `8ed2bdcee6d244c128f3c5d95be3d7817418433b` | PR #460 merge (recovery evidence; merged with failing `governance-fast`, F-MCP-P6-GOV-003) |
| P6_TEST_EXECUTION_SHA | `9b8c19d75d5df0f4b7c87d2707811f3715805102` | IIIC replay |
| P6_G6_AUDIT_EXECUTION_SHA | `9b8c19d75d5df0f4b7c87d2707811f3715805102` | G6 replay |
| P6_G6_AUDIT_SHA | `9b8c19d75d5df0f4b7c87d2707811f3715805102` | G6 audit SHA (same value as `P6_G6_AUDIT_EXECUTION_SHA`; tree-identical to `P6_MERGE_SHA`) |
| P6_IIIC_AUDIT_SHA | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` | ACT-N11 `audit_execution_sha` for the P6 re-bind; replay tree `9b8c19d75` (`P6_TEST_EXECUTION_SHA`) |
| P6_G6_EVIDENCE_PUBLICATION_SHA | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` | G6 + IIIC evidence phase-1 |
| P6_IIIC_EVIDENCE_PUBLICATION_SHA | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` | ACT-N11 P6 re-audit |
| P6_EVIDENCE_PUBLICATION_SHA | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` | Combined G6 + IIIC evidence publication (same value as the two rows above) |
| P6_IIIC_CERTIFICATION_PUBLICATION_SHA | `276c64305799fa4d7f1a25018cc1cccf061fbafd` | Certifier phase-2 pin |
| P6_ACTIVATION_SHA | `92ad3555775649742299826e3680ec8ce703cc8a` | IIAC ACT-N11 re-bind, phase 1 |
| P6_ACTIVATION_PHASE2_SHA | `19c617365cda85d88b474575e77b7d083739b529` | IIAC ACT-N11 re-bind, phase 2 |
| P6_FORMAL_CLOSURE_SHA | `b18a7afc41e25234ca7e2adc903cfea1e0bf63f0` | Formal scoped closure commit (ticket `closed (scoped)`); pinned here in the W7 follow-up, not in itself |
| P6_TERMINAL_CLOSURE_SHA | `6e53fb36f39b4a57331e6c079f2fcbe6094e2786` | Merge commit of the closure PR #468 |
| P5_IMPLEMENTATION_SHA | `8b153422ec1e526c588b7144e934d268055a6df8` | Qualification batch commit |
| P5_PUBLICATION_SHA | `c5989282937f5e592748cb606b2272d04837c601` | PR #456 merge |
| P5_TEST_EXECUTION_SHA | `c5989282937f5e592748cb606b2272d04837c601` | Independent G6 replay |
| P5_G6_AUDIT_EXECUTION_SHA | `c5989282937f5e592748cb606b2272d04837c601` | Post-publication audit |
| P5_G6_EVIDENCE_PUBLICATION_SHA | `7fd899aaea873aac6ec9ae3fe6ce5e3ef9ec9c8c` | G6 closeout docs (amend pin) |
| PLAN_DIGEST_SHA256 | `5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7` | T-IGX-035-A |
| TEST_EXECUTION_SHA | `abbe1525450da77da6801b080619a600c20cd497` | P7 qualification package (branch `mcp-p7-qualification`; clean-checkout replay at the same SHA) |
| P8_IMPLEMENTATION_SHA | `eadb1c2116feb0a2090b234a36fd4182bd6ace8d` | Frozen candidate = P7 publication merge (PR #474) |
| AUDIT_SHA | `eadb1c2116feb0a2090b234a36fd4182bd6ace8d` | P8 independent audit `audit_execution_sha` (`PASS_SCOPED`); evidence publication = PR #477 merge `a618a03a8b245ceefef6ab5a922213aaef46dcd1` |
| CERTIFICATION_SHA | `919e34b8f9fbccfc62a16205d8647d6e6070ce45` | P8 IIAC certification publication (PR #479 merge, `CERTIFIED_PASS_SCOPED`) |
| EVIDENCE_PUBLICATION_SHA | `919e34b8f9fbccfc62a16205d8647d6e6070ce45` | P8 B6 evidence complete on `origin/development` (audit `a618a03a8` + certification `919e34b8f`); terminal verification report published in the closure PR |
| FORMAL_CLOSURE_SHA | `8864cee842a2bc745d26f2a3501dc500e5f3e7df` | Closure commit of the MCP-SERVER-X scoped formal closure (roadmap status), pinned in the follow-up PR |
| P8_TERMINAL_CLOSURE_SHA | `c11184164af3072d0a48102a5e8d62adfec53695` | Merge commit of the closure PR #480 (merged 2026-11-12T12:47:37Z; its five required checks finished `success` between 12:47:31Z and 12:59:31Z, i.e. four of them after the merge, see P8-PROC-02) |
| FINAL_REMOTE_HEAD | `c11184164af3072d0a48102a5e8d62adfec53695` (P8 closure merge; the merge of the pin PR that adds this row is resolved with `git log --first-parent`) | P6: `6e53fb36f39b4a57331e6c079f2fcbe6094e2786` | Head verified by the independent P6 terminal verification (`P6_TERMINAL_VERIFIED`). The merge commit of the PR that publishes the verification report cannot pin itself: resolve it with `git log --first-parent --diff-filter=A --format=%H -- docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_P6_TERMINAL_VERIFICATION_REPORT.md`. History: previously `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` (post–P6 product merge, governance recovery pending) |
