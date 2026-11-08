# HIST-01 master dossier - QUAL-X / SKILL-X / TOOL-LIFE-X (FRZ-REG-01..10)

Baseline `ba07ebaa34d04633bd60510f68c18346b1ef160d` (branch `hist-cert/hist01-qualification`; no commits, nothing pushed). Role: inventory, independent verification of published records, candidate correctives requested from other sessions. HIST-01 implemented nothing, so there is no implementer/auditor conflict to manage.

Evidence discipline: published repo records and git facts are primary. The AUD-* snapshot in `inputs/` and the HIST-04 notes are LEADS that were re-checked where stated. Two pytest runs were made on the mutable worktree (clean at the baseline) under the heavy lock: they are **not exact-SHA evidence**.

## 1. Final per-stage dispositions

| Stage | Disposition | Basis |
|---|---|---|
| QUAL-X Population A | **PASS_SCOPED (independent), bound to `d9ea0d6e8`**; accepted by a delegated, process_enforced record; **not carried to HEAD** | AUD-QUALX 630 passed at d9ea0d6e8; OD-QUALX-ACCEPT; 214 product files changed since; HIST-04: SIGN-1 effective as signed |
| QUAL-X-P0 | **VERIFIED SEPARATELY (scoped) with errata** F-00, F-01, F-02, F-03 | OD-QUALXP0-VERIFIED; chaos script defect (F-02) still a live candidate |
| FRZ-REG-01..10 | **all OPEN, none promotable now** | per-ID matrix: 7 QUALIFICATION_GAP, 1 EVIDENCE_GAP, 1 PATH_B_LIVE_DEFERRED, 1 VALID_SCOPED_PASS_CANDIDATE; plus the scope-of-promotion authority question H01-F013 |
| SKILL-X | **scoped acceptance valid as written and NOT expanded; CONDITIONAL on DS-5 and on QUAL-X staying accepted** | AUD-SKILLX 117 passed at 552e37984; DS-5 condition met in form (commits 9d0fc1ca1, b0286f769 on HEAD); SKILL-X inventory + TL14 gates green at HEAD (135 passed, mutable worktree) |
| SKILL-X unsupported lifecycle (disable/remove/upgrade) | **excluded capability (ACCEPTED_LIMITATION)**, roadmap says "where supported" (L687), scope note present (L2058); operator confirmation advised | H01-F010 |
| TOOL-LIFE-X | **NOT CLOSED in substance: CONDITIONAL.** The roadmap label "CLOSED - INDEPENDENTLY ACCEPTED" is not supported by a published independent review, and C-1 and C-2 are still red at HEAD | H01-F001..F004, F014, F015 |

## 2. Counts
- Findings: 16 (H01-F001..F016). By class: SSOT_DRIFT 4 (F001, F005, F006, F008), QUALIFICATION_GAP 3 (F002, F003, F004), EVIDENCE_GAP 2 (F009, F014), IMPLEMENTATION_GAP 1 (F015), AUTHORITY_GAP 2 (F011, F013), ACCEPTED_LIMITATION 2 (F007, F010), UNVERIFIED 2 (F012, F016) = 16.
- FRZ-REG: 7 / 1 / 1 / 1 as above. FRZ baseline for this family re-verified: 10 rows, all OPEN, no divergence from the MASTER count.

## 3. Tests run (all on mutable worktree at `ba07ebaa3`, heavy lock held, one `uv run` each)
| Run | Scope | Result | Log |
|---|---|---|---|
| run1 | TLX side-effect-safety module + F3 streamable-HTTP module (`--extra mcp`) | 5 failed, 34 passed, 96 s | `run1_tlx_c1_c2_head.log` |
| run2 | 7 SKILL-X modules + 2 TL14 architecture modules | 135 passed, 70 s | `run2_skillx_tl14_head.log` |
Not run: QUAL-X Dimension A manifest (35 modules, about 18 min, needs Operator budget per the C10/C15 freeze rule), F8 inventory, L3/TL11 set, anything with Docker.

## 4. Which FRZ-REG rows need current-HEAD replay
Replay: REG-02, 03, 05, 06, 07, 09, 10. Census, not replay: REG-01, REG-04. Live part of REG-08 is PROD-Q Dimension B (not ours). One exact-SHA clean-checkout run of the QUAL-X Dimension A manifest on a pinned current SHA covers most of the replay rows at once.

## 5. Key risks stated plainly
1. TOOL-LIFE-X is labelled closed while two mandatory test items are red and a lifecycle defect (unguarded submit) exists at HEAD.
2. Every QUAL-X/SKILL-X statement is bound to older SHAs; the tree moved 214 product files.
3. The delegated-authority chain is process_enforced; Amendment 01 still contradicts itself (header IN FORCE vs body "Not in force"). HIST-04/operator own that.
4. The unpublished correctives sit on local branches; a ticket ID collision (T-IGX-023-A) will bite on publication.

## 6. Out of scope and kept out
ENT-AUDIT-X, FRZ-AUD/DOC/DEBT/FRZ families, ARCH-FREEZE, PROD-Q live, Docker, new features. No file under the main checkout or other worktrees was touched; local `cert-wave/*` branches were read with `git show` / `git log` only.
