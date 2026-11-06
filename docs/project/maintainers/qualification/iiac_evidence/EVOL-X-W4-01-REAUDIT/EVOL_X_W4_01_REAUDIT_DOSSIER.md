# EVOL-X W4-01 consolidated B4 corrective — independent re-audit dossier

**Auditor session:** `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9` · agent role `independent-quality-auditor` · model `claude-sonnet-5-5`
**Audited SHA:** `9f8b04d2b8b670077b1e78ce812bb34b680f3968` (checked out detached in the auditor worktree `stoic-panini-83e1a9`; `origin/development` fetched, contains the pin)
**Baseline:** `5afaad9eadfc1655f4ef1ab59eb022960a924648` · **Policy:** `IIAC-2026-11-05.1` (IN_FORCE, unchanged) · **Ticket:** `T-IGX-019-A`
**Implementer (not me):** `local_4b071fe5-82d0-48bc-b524-146a3daf5c45`. The certifier must be a third session.

This dossier is audit evidence. It is **not** a certification, does not accept B-4, performs no B6, promotes no FRZ row, closes nothing, signs nothing and fills no operator record or W4-F08 template. Historical implementer session ids stay `UNVERIFIED / OPERATOR_ATTESTATION_REQUIRED`.

## 1. Verdict

**PARTIAL — exact blockers: `W401-RA-01`, `W401-RA-02`, `W401-RA-03`** (all MEDIUM). Everything the report and handoff claim to have closed holds up under independent reproduction for the *listed* attack classes, but I found three further ways to obtain a false PASS or to widen scope text that the gates do not see. None of them lets the canonical repository reach PASS today (B-4 / B6-P3 / B-5 stay open), and none is a regression of RA-B4-01..04.

| Area | Verdict |
|---|---|
| (1) DA-01 field repoint, delete+re-add, non-ancestor, fixture fallback, first-publication pin | **PASS** for those paths; **FAIL** via `refs/replace` (RA-01) |
| (2) DA-02 gate-consumed scope sealed | **PASS**; unsealed channels accepted (RA-02) → **FAIL** for "no amendment channel" |
| (3) blocker lifecycle after P2 | **PARTIAL** (RA-04 side-branch hiding) |
| (3) operator-decision parser | **PARTIAL** (RA-05) |
| (3) UTF-8 binding republication | **PASS** |
| (3) LF/CRLF repair via `.gitattributes` | **PASS** (non-material, digest meaning kept) |
| (4) RA-B4-01..04 regression | **PASS** (no regression), but RA-03 shows RA-B4-01's "uncommitted edits stay a mismatch" is only partly true (inherited defect) |
| (5) canonical cannot PASS; no product/material/activation/evidence change | **PASS** |

## 2. Commands run (my own, Windows 11, `core.autocrlf=true`, short basetemp)

| Command | Result | Raw |
|---|---|---|
| `git fetch origin`; `git diff 9f8b04d2b origin/development -- . ':!docs' ':!governance/tickets'` | empty (later commits are docs/tickets only) | `raw/diff_stat_baseline_to_pin.txt` |
| `uv sync --extra dev-ci --frozen` | exit 0 | `raw/uv_sync.txt` |
| `pytest tests/unit/qualification/test_evol_x_closure_preflight_b4.py tests/unit/qualification/test_evol_x_b4_ruff_evidence_exemption.py -q -n 6 --basetemp=D:/w4a/bt` | **423 passed, 0 failed, 11 min 59 s, EXIT=0** (handoff text says 424; collect-only gives 423 = 415 + 8; later docs say 423) | `raw/pytest_b4.txt`, `raw/collect_only_b4.txt` |
| `pytest tests/unit/governance -q -n 4 --basetemp=D:/w4a/bg` | **145 passed, 1 skipped, EXIT=0** | `raw/pytest_governance.txt` |
| `python -m scripts.governance.verify_iiac_policy_activation --require-in-force` | `IN_FORCE @ IIAC-2026-11-05.1`, exit 0 | `raw/verify_activation.txt` |
| `…preflight.py --gate b6-readiness` (canonical) | **FAIL**, exit 1, Gate A has exactly two errors, both `A-18` (B-4, B6-P3), **no A-19** | `raw/canonical_b6_readiness.txt` |
| `…preflight.py --gate formal-closure` (canonical) | **FAIL**, exit 1 (B6 unpublished, 10 FRZ rows OPEN, OD-EVOL-12/OD-EVOLX-08 PENDING, B-4/B-5/B6-P3 open) | `raw/canonical_formal_closure.txt` |
| canonical + `--evidence-root` / `--p2-sha <HEAD>` | refused (`policy mode: INVALID`, A-01 / A-19 override forbidden) | inline |
| Linux container run | **UNVERIFIED** here (Docker daemon: `read-only file system`) — CI substitute, see RA-10 | `raw/linux_container_run.txt`, `raw/gh_run_jobs.txt` |

## 3. Scope of change (check 5 and handoff item 1)

`git diff 5afaad9ea..9f8b04d2b` touches 10 files: `.gitattributes`, `.github/workflows/unit-tests.yml`, the preflight, its B4 test module, `tests/unit/qualification/conftest.py`, `tests/unit/governance/test_evol_x_real_repo_fresh_clone.py`, the new `…BINDING.utf8.json`, the report, the handoff, the ticket. Verified with `git diff --name-only` per path: **0 changed files** under `intergrax/`, `agents/`, `applications/`, `scripts/governance/`, `IIAC_POLICY_ACTIVATION_RECORD.json`, the certification protocol, the ECAP protocol, `iiac_evidence/`, the original binding, the delegation record and the handoff record (blob `4cb3c761e…` identical). The only modified (M) files are the four listed above. No `MATERIAL_GOVERNANCE_PATHS` entry changed. (`raw/scope_of_change.txt`.)

## 4. DA-01 — can the P2 baseline be moved or forged?

Mechanism read in code: `evaluate()` uses the constant `P2_CERTIFICATION_PUBLICATION_SHA` (`ad0877715…`); canonical rejects `--p2-sha`/`--evidence-root` overrides; `_p2_sha()` ignores the manifest; A-19 requires manifest and record fields equal to the anchor, anchor a strict ancestor of HEAD, anchor == first commit that added the record, manifest bytes M0, P2 evidence files + index entries byte-identical.

My own reproductions (fixtures built with the implementer's builders but my own attack scripts, `raw/attack5_own_da01_da02.txt`):

* V1 widen LIM-02 and re-point both fields to the widened commit → rejected (re-point + LIM dropped/added).
* V2 amend a P2 certification evidence file and regenerate the index → rejected.
* V3 FRZ-EVOL-09 SCOPED→UNSCOPED plus re-point → rejected (A-08, A-19).
* Honest control passes.

New path found: **`refs/replace`** (RA-01). With a replace ref on the pinned P2, `git show P2:path` returns forged bytes; a committed widened LIM-01 then passes Gate A. Disabling replace objects restores FAIL. A plain clone does not copy replace refs, so exploitation needs write access to the verifying repository (or an explicit refspec). I did not manage to forge the *blocker* baseline this way (history walk re-surfaced the real blockers). Not reproduced on the real repository chain: UNVERIFIED there.

## 5. DA-02 — can certified scope be widened and still pass?

Gate-consumed scope is sealed: limitation multiset, verdict, FRZ dispositions, evidence-only `notes`, F-C-01/02, every non-lifecycle key (sealed, dropped keys rejected), manifest bytes, P2 evidence files. Reordering limitations is accepted (positive control S7c passes); re-indenting the manifest is rejected by the M0 byte contract (S7b) — a formatter run on the manifest would fail, acceptable by design.

Found (RA-02), all accepted by Gate A: widening text in lifecycle keys (`certification_publication_sha_binding`, `frz_checklist_rows_changed`), an extra *indexed* file in `certification/`, and a duplicate JSON key in the certification record. They are not consumed by any gate, so they cannot flip a gate on their own, but they contradict the report's "no amendment channel exists by design".

Legitimate operations wrongly rejected (RA-06, RA-05): B6 record and promoted FRZ-row formats are exact and undocumented (real FRZ-EVOL-06 note must be `LIM-01,LIM-03,LIM-04; footnote: EVOL_CONSERVATIVE_FULL scope only; not PLATFORM_CONSERVATIVE_FULL`, replacing the Evidence column); non-ASCII operator scope text (Polish letters, em dash) is rejected.

## 6. Other checks

* **Blocker lifecycle:** linear add-then-remove, rename, delete-all, resolved+open → rejected. Side-branch blocker dropped by a `-s ours` merge → accepted (RA-04, new W4-F03 code; baseline had no history walk).
* **Operator-decision parser:** real register rows (`SIGNED / APPROVED`, `OPERATOR APPROVED`) accepted; `PENDING`, `AUTHORIZED`, `DECIDED`, decorated or struck-through statuses rejected; second status line and `Revoked:` key rejected; gaps in RA-05.
* **Dirty working tree (RA-03):** committed widened FRZ row / PENDING register / PENDING governance JSON → Gate B FAIL; the same files corrected only in the working tree → Gate A and B PASS with HEAD unchanged. `_read_bytes` is unchanged from the baseline, so this is inherited, not introduced.
* **UTF-8 binding republication:** original blob is not UTF-8 (`0x97` at offset 954), sha256 `584c504e…5be9` recorded in the new file matches the committed original; decoding the original as cp1252 and parsing equals the republication field for field except the two *added* fields `supersedes` and `republication_ticket`; `verdict_preserved: true`, `not_certification: true`; original untouched; `-text` set on binding, republication, delegation record. **PASS** (wording nit in RA-09).
* **LF/CRLF repair:** activation record `handoff_digest` = `96fcf995…` = sha256 of the CRLF-expanded blob; the committed blob (no CR) hashes to `3b7e901f…`; the working copy in my checkout is CRLF and `verify_iiac_policy_activation` passes. `.gitattributes` adds `text eol=crlf` for that one path (exact path, not a glob); the gate, the activation record and every `MATERIAL_GOVERNANCE_PATHS` file are byte-identical to the baseline; `.gitattributes` is not on the list. The digest still covers exactly the bytes the operator-signed activation bound. **PASS.** Caveat: gate correctness now depends on `.gitattributes` being honoured at checkout; removing the rule fails closed (ACT-N08), it cannot weaken a check.
* **RA-B4-01..04 regression:** the diff of the existing test module changes fixtures only where the stricter contract requires it (FRZ-EVOL-06/07/08 fixture dispositions and promoted rows, `Accepted limitations` tokens, B6 title, explicit `p2_sha`), no assertion removed; the RA-B4 families (`ra_b4_01…04`) are inside the 423 passes. Original-register positives still pass (real data probe).
* **Canonical PASS impossibility:** both gates FAIL on the pinned canonical tree; a canonical run with overrides is INVALID.
* **Residuals challenged:** OD-1 confirmed (RA-08). Fixture-class PASS exit code 0 (RA-07).

## 7. Governance suite

See `raw/pytest_governance.txt` (appended result below).

**145 passed, 1 skipped (symlink creation unavailable on Windows), 0 failed, 13 min 39 s, EXIT=0.** Includes the real-repository fresh clones (autocrlf true/false/input) and the ACT-N08 tamper cases. The implementer's CI figure of 146 passed on Linux is consistent (the skipped test runs there).

## 8. UNVERIFIED / not done

* Linux container run (RA-10). Substitute: Actions run `42214173868` on `49b50fd74` — `governance-fast` and `evol-x-b4-preflight` success; `ent-audit-fast` is a pre-existing separate failure (failing at least since `2ae101411`).
* `refs/replace` attack on the real repository chain; `.git/info/grafts` and shallow-boundary variants; Windows MAX_PATH deep-path behaviour beyond the short basetemp.
* W4-F08 attestation: nothing signed or filled by me. Operator-owned items untouched: W4-F08, OD-1, OD-EVOL-12, OD-EVOLX-08, P3, B6, FRZ promotion, EVOL-X closure.

## 9. Files

`EVOL_X_W4_01_REAUDIT_FINDINGS.json`, `iiac_audit_session_manifest.EVOL-X-W4-01.reaudit_draft.json`, `raw/*`. Attack scripts: `raw/scripts/`.
