# HIST-01 - QUAL-X -> SKILL-X predecessor reconciliation

Baseline `ba07ebaa34d04633bd60510f68c18346b1ef160d`. Role: inventory and independent verification of published records. HIST-01 did not implement or close any of these stages. Nothing here is acceptance, a PASS or a promotion.

## 1. Chain of facts (all re-checked with git unless marked LEAD)

| Step | SHA | Time (2026-10-23) | Fact |
|---|---|---|---|
| QUAL-X acceptance evidence | `d9ea0d6e87b939e4cefb9971c6bbd4fbcaecc0ce` | 08:03 | ACCEPTANCE_SHA |
| QUAL-X conditional package | `4c5faa89aad381e293da2210ec3e83566c3a8473` | | parent of the closure commit (not a descendant, AUD-QUALX F-07) |
| QUAL-X formal closure | `0d52df6e0da10bb1ec36bcb113b54a4866b9b76f` | 10:52 | doc-only vs d9ea (3 .md files); self-adjudicated A1-A5 table inside the same commit |
| SKILL-X acceptance baseline | `552e379845a9512c275f3ebe520ac51767c97cf8` | 16:46 | |
| SKILL-X publication | `76a98cb3f3e4576ab694f03a5b3e270eb7694f9e` | 16:51 | 5m41s after the record said NOT CLOSED (AUD-SKILLX F-02) |
| TOOL-LIFE-X P0 record | `f5c933097` | 16:56 | 5 min after SKILL-X publication |

Roadmap L480 conditioned SKILL-X entry on "independent acceptance of the QUAL-X formal closure commit". That did not exist on 2026-10-23. The deviation is real and is recorded as such in OD-ROUTE-DS5.

## 2. What regularised it, and what that does and does not do

1. AUD-QUALX (aud_qualx, separate session): PASS_SCOPED at d9ea0d6e8; 630 passed, 0 failed (35-module Dimension A + BND-03 + W2-C). LEAD file: `inputs/AUD-QUALX/`. Ratified as the audit route by OD-ROUTE-DS1.
2. OD-QUALX-ACCEPT-2026-11-07: independent acceptance, Population A only, bound to d9ea0d6e8. Explicitly NOT accepted: global FRZ-REG-01..10, PROD-Q Dimension B, Population E, ARCH-FREEZE, any state after d9ea0d6e8.
3. OD-ROUTE-DS5 (SIGNED_WITH_CONDITIONS): regularises the deviation by audit, effective only from the recording of OD-QUALX-ACCEPT in the repository. If the QUAL-X acceptance is withdrawn or the audit reopened, SKILL-X, TOOL-LIFE-X and DIST-SCALE-X acceptance become conditional again.
4. OD-SKILLX-ACCEPT-2026-11-07 (SIGNED_WITH_CONDITIONS): SKILL-X accepted for EC1-EC11 under the catalog-witness model; conditional on item 3.

**DS-5 effectivity at baseline (documentary):** the signed OD-QUALX-ACCEPT text and signature lines are present in `CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md` at `ba07ebaa3`, and roadmap L2057/L2058 cite both decisions. So the stated trigger ("recording in the repository") is met in form. HIST-01 leaves the question of whether the delegated record is a valid authority to HIST-04 (needs_master N2) and marks every dependent closure AUTHORITY_REQUIRED until that note lands.

Hard limits stated in the records and kept here:
- Assurance is `process_enforced`; no IIAC verifier recognises `operator-delegate`; it is not a cryptographic signature. The delegation record (sha256 `5264bca3...bdab`) is gitignored and was not read by HIST-01.
- The delegation expired 2026-11-07T23:59:59+01:00 (SIGN-5 dossier). Records signed before that stay valid within their bounds; nothing new can be signed on it.
- The regularisation does not certify the current tree: 214 product files changed d9ea..HEAD.

## 3. QUAL-X Population A: authority and exact SHA

| Item | Value |
|---|---|
| Authority | OD-QUALX-ACCEPT-2026-11-07 (delegated). The original 2026-10-23 operator message (A1) was NOT_REPRODUCED by the auditor and is not relied on. |
| Exact SHA | evidence d9ea0d6e8; closure/publication commits are doc-only descendants (verified by AUD-QUALX and the signer; HIST-01 re-ran the d9ea..HEAD diff count, not the d9ea..0d52 diff) |
| Execution evidence | 630 passed at d9ea0d6e8, one invocation (LEAD log in `inputs/`). 535/535 at `9a175177` is an attestation without logs, and it did not cover d9ea (7 runtime files changed in between). |
| Open QUAL-X doc debt | F-04 (32 vs 35 modules) and F-06 (stale READY FOR AUDIT text) remain unfixed in the inventory (H01-F008) |

## 4. QUAL-X-P0 errata and scope boundary

Verified separately (OD-QUALXP0-VERIFIED), not absorbed into the Population A acceptance.
- F-00: three Q01/Q03-cited gates were RED at the P0 tree a70b68cf9 (test_ebh_2a_public_contract_boundary_gate, test_ebh_2i_final_rescan_gate, test_hardening_3_layer_boundary_gate). The P0 "DIRECT" labels were not true at P0. They went green by d9ea0d6e8.
- F-01: P0's own audit-before-P1 gate was bypassed. Cannot be repaired afterwards.
- F-02: `scripts/maintenance/check_architecture_boundary_chaos.py` builds `REPO_ROOT / "scripts" / <child>` (L25). HIST-01 read that line and the file exists at HEAD; the missing child path was reported by the signer and is NOT re-run by HIST-01. Still a live defect candidate.
- F-03: Q03 path is `tests/unit/runtime/architecture/test_ebh_2f_r2_integration_plugin_contract_typing_gate.py`.
- Not covered by the verification: Q05 test_npsc*, AW-7C PHYSQ, Q09/Q10 historical batches, most of the governance/tenant_x/state_x/trace_x trees.

Negative-test position: the Population A evidence includes behavioural negatives (P2 ingress matrix, QINF stale-inventory violations). HIST-01 has not re-executed them at HEAD.

## 5. Which FRZ-REG rows need global current-HEAD replay

See `HIST01_FRZ_REG_PER_ID_MATRIX.json`. Replay: REG-02, 03, 05, 06, 07, 09, 10. Census, not replay: REG-01, REG-04. Not in this campaign: REG-08 live part (PROD-Q Dimension B). A replay on the mutable HIST-01 worktree is not exact-SHA evidence; the exact-SHA run has to come from a clean pinned checkout (needs_master N4).

## 6. SKILL-X scoped certification (validated without expanding it)

Checked against roadmap section 3.0.4 EC1-EC11 and `SKILL_X_PARENT_FORMAL_CLOSURE_RECONCILIATION.md`:

| Check | Result |
|---|---|
| Accepted scope | EC1-EC11 under: catalog-witness supply chain (EC9), no per-certification full GOV-X2 replay (EC2), structural-only registry-less recovery, lifecycle install/bind/enable |
| Code and tests since e00a0aab | `intergrax/skills` and `tests/unit/skills` unchanged (git log shows e00a0aab2 as latest) |
| FRZ promotion | none; reconciliation lists FRZ-CTL-09/TEN-04/STA-06/REC-02/REC-04 as supporting only |
| Imports from agents/applications | none (signer's grep; not repeated) |
| Replay evidence | 117 passed at 552e37984 (AUD-SKILLX, LEAD log); not replayed at HEAD |

**Unsupported lifecycle surfaces (mission item 9).** `intergrax/skills/host_lifecycle.py` exposes `registry_read`, `is_bound`, `binding_metadata`, `enable_installed_skill`, `bind` (plus an install projection elsewhere). There is no disable, remove or upgrade. The roadmap's own text for the lifecycle (L687) says "Certify the full lifecycle **where supported**", and the exit criteria (L721-733) do not name disable/remove/upgrade. HIST-01's reading: these are an excluded capability, not a mandatory requirement, and the F-04 scope note in roadmap L2058 is the correct freeze treatment. Residual: the original closure never stated the supported subset (AUD-SKILLX F-04); the operator should confirm the exclusion. If the operator rules them mandatory, that is a new SKILL-X increment (new implementation), not historical closure, and HIST-01 will not certify it silently.

**EC1-EC11 spot status.** All eleven are marked ACCEPTED (scoped) in the reconciliation. Thin points that the audit already flagged and HIST-01 keeps open: EC3 rests on 3 architecture-gate tests; EC4 determinism tests sit outside the 117-test inventory (F-05); runtime/host-factory edits since e00a0aab were not replayed (F-07); the "external independent review" never existed as an artifact (F-01), so the acceptance stands only on AUD-SKILLX plus the delegated signature.

## 7. Conclusions

1. QUAL-X Population A: independently audited (PASS_SCOPED) and accepted by a delegated, process-enforced record; bound to d9ea0d6e8. All ten FRZ-REG rows stay OPEN.
2. SKILL-X: scoped acceptance is valid as written and is not expanded. It is CONDITIONAL on DS-5 and on the QUAL-X acceptance staying in force. The condition is met in form at baseline; its authority is pending HIST-04.
3. Dependent obligations: SKILL-X formal "unconditional" status = AUTHORITY_REQUIRED (HIST-04). Optional HEAD regression replay of the 117-test inventory = QUALIFICATION_REQUIRED (low priority).
