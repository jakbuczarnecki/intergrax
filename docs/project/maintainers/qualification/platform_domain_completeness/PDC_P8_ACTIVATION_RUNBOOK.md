# PDC-X P8 activation runbook (PDC-P8-04: ACT-N11 rebind at the global audit SHA)

**DRAFT runbook, not a verdict.** Ordered hand-over for the sessions that run after the global audit (IA-GLOBAL). Written by a read-only research session that is not an implementer, an auditor, a certifier or a signing delegate. It carries no verdict, no signature, no activation line, no rebind, no FRZ change and no closure claim. It does not edit the gate, a verifier, the protocol or the activation record. Drafted 2026-11-15 against `origin/development` `cba32d2479cd134911d2fdf4fb1db4587c8a94d8`.

Legend: **[gate]** read from `scripts/governance/iiac_activation_gate.py`, `iiac_common.py` or the two companion verifiers; **[precedent]** copied from the merged MCP-RC re-pin 2 (PR #554: `32dc342d0`, `88fd679e6`, `eb3b400f3`, `b4d2a3c17`) and its logs; **[record]** quoted from a delegation or decision record; **UNVERIFIED** not derivable from the repository at drafting.

Sources read: `iiac_activation_gate.py` (578 lines), `iiac_common.py`, `verify_iiac_policy_activation.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`, `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`, `IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md`, `IIAC_POLICY_ACTIVATION_RECORD.json`, `mcp_server_x/MCP_SERVER_X_RC_ACTIVATION_RUNBOOK.md`, `..._ACTIVATION_CERTIFIER_REPORT.md`, `OD-MCP-RC-04`, `OD-MCP-RC-05`, `DELEGATION-PDC-STANDING`, `DELEGATION-PDC-P7-ACT`, `DELEGATION-PDC-P7-REPIN`, `DELEGATION-PDC-WAVE2`, `DELEGATION-PDC-CLOSE`, `OD-PDC-IIAC-SCOPE`, `PLAN_T-IGX-036-A.md` rows PDC-P7 to PDC-P9, `PDC_EBH7_REPIN_DOSSIER.md`, `PDC_SHA_LADDER.md`.

---

## 0. Update 2026-11-16 (orchestrator; supersedes the values below where they differ)

The runbook below was drafted against `cba32d247` and names `G`, `E`, `C` symbolically. Since then:

- **`G` is fixed: `969cbd964`** (`development` after PR #593). The last product-path change is PR #594 (`7f1859e0f`, T-IGX-041-P, three files under `intergrax/rag/graph/`); everything after it is tests, `.github`, docs or tickets. The IA-GLOBAL Addendum 2 (PR #595, `4220069c2`) verified this delta at `969cbd964`. The merge hold of section 2.3 therefore runs from `7f1859e0f` until the P8-04 PR is merged. The product set `a1b3053b4`..`G` is no longer the 62 paths of section 1: recompute it with the gate's own helper before writing the manifest `limitations` (**UNVERIFIED** count).
- **Verdict input.** IA-GLOBAL Addendum 3 (PR #597) states the final verdict at `969cbd964` as PASS_SCOPED with the Operator-accepted exceptions in the manifest limitations. The Operator accepted G-05 residual (77 scenario reds), E1-E11, the `trace_bridge.v1` mismatch (G-10 residual) and G-28 (Cypher store) as declared exceptions: `operator_decisions/DELEGATION-PDC-EXCEPTIONS.md` (PR #596, `be051d38c`). The certifier lists them, with the earlier declared exceptions of `DELEGATION-PDC-CLOSE.md` decision 1, in the binding manifest `limitations` (authority: `DELEGATION-PDC-P8-ACT.md` (b)).
- **EBH-7 replay.** The replay at pin `d39a520dd` (run 43738431866) failed; the two test defects are fixed by T-IGX-041-Q (PR #593). By the Operator condition of `DELEGATION-PDC-LOOP2.md` it is **not re-run** and there is no new pin; the failed run is recorded by PR #598 as "not a B6 PASS". Wherever this runbook expects a green replay or a re-pin (sections 2.1, 2.2, 10), it does not: the EBH-7 identity gate stays a known red class and no B6 claim is made.
- **CI.** The runner pin PR #592 (`ubuntu-24.04`) is merged; the known red classes on a docs-only PR are the three IIAC activation files (ACT-N11) and nothing else.
- Authority records now merged: `DELEGATION-PDC-LOOP2.md`, `DELEGATION-PDC-EXCEPTIONS.md`, `DELEGATION-PDC-P8-ACT.md` (a)-(c).

---


## 1. State at drafting (from git, 2026-11-15)

| Item | Value |
| --- | --- |
| Activation record | `status` `IN_FORCE`; `audit_execution_sha` = `evidence_publication_sha` `a1b3053b49c28124a0abde679d733f21a597797f`; `corrective_implementation_sha` `d02c399508c40b4a06d846410334d677f226c18d`; `certification_publication_sha` `dc1e66fa140d81b28c2658e88b4e1aa16eb29ad2`; `effective_from_sha` = `policy_activation_sha` `88fd679e63188303e0590e4033abc81a3c30805b` (phase 1; phase 2 `eb3b400f3`). This is the MCP-RC re-pin 2, merged by PR #554 (`61531828f`). It also covers the PDC-X integrated baseline; no separate PDC-P7-00 activation exists. |
| Signed line in the record | `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ 2026-11-14T08:21:13+01:00` in `IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md`, decision id `IIAC-OD-MCP-RC-ACT-N11-2026-11-13`; written by a designated signing session (file provenance note). |
| Bound artifact set | The `IIAC-GOV-1` set: report digest `c405c65f...7a4e`, scope `e06ce433...`, matrix `d5028f19...`, findings `df3648f9...`; the PDC and MCP audits are only listed in the manifest's extra fields **[precedent]**. |
| Gate now at `origin/development` | `verify_iiac_policy_activation --require-in-force` fails with **62** lines `ACT-N11 product path changed after the audited SHA` and nothing else (my run at `cba32d247`; two further `ACT-N10 companion` lines in my log were environment noise from a system Python without `jsonschema` and without the repo on `PYTHONPATH`, see section 5.4). The count is identical against `a1b3053b4` and against `61531828f`. |
| What the 62 are | Exactly the eight product-bearing T-IGX-041 repair merges (all under `intergrax/`, none under root `agents/` or `applications/`): #568 041-E 11, #569 041-C 14, #570 041-G 7, #571 041-D 4, #572 041-H 8, #573 041-F 4, #574 041-A 3, #575 041-B 11 (sum 62). #576 (041-I), the audit PRs #559 to #566 and the record PRs #567, #577, #578 change no product path. The per-domain audits IA-D1..IA-D8 all audited `c08efe46a74a20595468d8d8bf06eab2dca5f573`, whose product tree equals the pin (`git diff 61531828f c08efe46a` has 0 product paths), so the 62 are post-audit repairs and are the whole delta the global audit must cover. |
| Material governance paths changed after the pin | None (checked against `61531828f..origin/development`). |
| Verdicts of the domain audits | D1, D2, D3, D4, D6, D7, D8 `PARTIAL`; D5 `FAIL` (row `PDC-D5-R015`); manifests carry `CONDITIONAL` (domain audit reports, header tables). |
| Not yet on `development` (at drafting) | tickets T-IGX-041-J, -K, -L (the plans are described only in `DELEGATION-PDC-P7-REPIN.md`); the global audit; the CERT-PDC matrix (`PDC_INDEX.md` row says "Not started"); the P9 drafts. |
| MCP-SERVER-X closure (PDC-P8-01 hard gate) | UNVERIFIED: the roadmap lines I read state the requirement but I did not find a closed status line; the global auditor confirms it itself (plan row PDC-P8-01). |

## 2. What voids an audit (ACT-N11), exactly

The gate evaluates `git diff --name-only <audit_execution_sha> HEAD` (a two-dot **tree** diff; HEAD is the merge ref in CI). It raises an ACT-N11 line for a path only if it is in one of two classes **[gate: `iiac_common.PRODUCT_PATH_PREFIXES`, `iiac_activation_gate.MATERIAL_GOVERNANCE_PATHS`]**:

| Class | Voids the audit | Notes |
| --- | --- | --- |
| Any path starting `intergrax/`, `agents/` or `applications/` | **Yes** | Includes `intergrax/platform/enterprise_audit/impact_map.py` and `changed_since_impact_map.v1.json` (F-41 precedent), `intergrax/applications/_shared/**`, and test files that live inside those trees (for example `applications/local_workspace_application/tests/test_run_artifact_metadata.py`, precedent). |
| `scripts/governance/verify_iiac_policy_activation.py`, `iiac_activation_gate.py`, `iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py` | **Yes** | Six material paths with the protocol. |
| `docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` | **Yes** | Last changed in `7b1c0a4b7` (T-IGX-040-B), an ancestor of the current pin. |
| Root `tests/**` (including `tests/qualification/**`, `tests/unit/**`, `tests/integration/**`) | **No** | Not a product prefix. |
| `docs/**` (including `docs/.../iiac_evidence/**`, the activation record, OD files, this runbook) | **No** | Only the protocol file is material. |
| `governance/**` (tickets, plans, role registry) | **No** | |
| `.github/**` (workflows) | **No** | |
| Other `scripts/**` (`scripts/qualification/**`, `scripts/gates/**`, other `scripts/governance/*.py`), `platform_proofs/**`, `pyproject.toml`, `uv.lock` | **No** | Not gated by ACT-N11 (they can still matter to the EBH-7 identity gate, see below, and to what the auditor chooses to audit). |

Further gate facts:

- Because the diff is between two trees, a product change that is fully reverted before HEAD does not appear. A rename out of a product path is not covered by this reasoning (UNVERIFIED how git reports it); do not rely on it.
- `verify_iiac_role_separation --strict-product-diff` separately requires an **empty** product diff between the manifest's `audited_implementation_sha` and `audit_execution_sha` **[gate]**.
- The ACT-N11 check also runs in CI on the PR merge ref, which contains the current `development`. Any product commit merged to `development` after the global audited SHA `G` therefore breaks the rebind, even if the PR branch itself is clean **[precedent: runbook (c)]**.
- Mechanical freeze check (use it before every step; expected output empty):

```bash
git diff --name-only <G> <HEAD-or-origin/development> -- intergrax/ agents/ applications/ \
  scripts/governance/verify_iiac_policy_activation.py scripts/governance/iiac_activation_gate.py \
  scripts/governance/iiac_common.py scripts/governance/verify_iiac_audit_manifest.py \
  scripts/governance/verify_iiac_role_separation.py \
  docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md
```

(I ran this pathspec form against `a1b3053b4..origin/development`; it returns the same 62 as the gate.)

### 2.1 Already planned changes, classified

| Planned change | Write set (from the records) | ACT-N11 path? |
| --- | --- | --- |
| T-IGX-041-K: classifier entries in `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` | `tests/qualification/**`, ticket files; the plan may name the three UNKNOWN paths (`intergrax/experiments/workflow.py`, `intergrax/integrations/credentials/secrets_store_resolver.py`, `platform_proofs/.../investigator_agent.py`) as **classified subjects only**, not edited | **No**, provided no `intergrax/**` file is edited (the delegation forbids `intergrax/`, `agents/`, `applications/`, `scripts/governance/` in the write set of J, K and L) |
| T-IGX-041-L: re-pin COMPAT-X counters (debt item Z-D1) | tests and docs (`DELEGATION-PDC-P7-REPIN.md`: nothing under the four forbidden prefixes) | **No** |
| T-IGX-041-J: EBH-7 pins: `docs/.../ebh_7_p03_independent_replay/MANIFEST.json`, `tests/qualification/ebh_7/_constants.py`, the three pin lines of `.github/workflows/unit-tests.yml`, roadmap document, renderer script | `PDC_EBH7_REPIN_DOSSIER.md` table | **No** (`.github/`, `tests/`, `docs/` are outside both classes) |
| W2 docs PR (replay run id and artifact digests, "not B6 PASS") | docs only | **No** |
| Held dependency-bot and `uv.lock` PRs | `uv.lock`, `pyproject.toml` | **No** for ACT-N11, but held until after the final re-pin anyway (`DELEGATION-PDC-WAVE2` decision 7) |
| P9 drafts (`PDC_EXCEPTION_REGISTER`, `PDC_DEBT_REGISTER`, `PDC_FRZ_PROMOTION_PROPOSAL`, closure text, final acceptance report, ARCH-FREEZE handoff) | docs | **No** |
| This runbook, the CERT-PDC matrix, the global audit evidence, the P8-04 activation files | `docs/project/maintainers/qualification/**` | **No** |
| Any product repair from a P8 finding (`intergrax/**`) | product | **Yes**: voids the audit; repeats audit delta, re-pin and (per `DELEGATION-PDC-CLOSE` item 2) one more EBH-7 re-pin; allowed once, a third loop returns to the Operator |
| A new ENT-AUDIT impact-map rule (F-41 style) | `intergrax/platform/enterprise_audit/**` | **Yes** |

### 2.2 The EBH-7 identity gate is a second, narrower freeze (not an ACT-N11 path class)

`PDC_EBH7_REPIN_DOSSIER.md` says the executable-tree identity gate allows after the pin `P` only `docs/project/maintainers/qualification/`, `.github/`, the verifier script and nine named files (seven under `tests/qualification/ebh_7/`, the roadmap document and the renderer script); `governance/tickets/` and `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` are outside it (`DELEGATION-PDC-P7-REPIN.md`, Ordering). I did not read the verifier script itself (UNVERIFIED beyond those two records). Consequence for this runbook: every file of PDC-P8-04 lives under `docs/project/maintainers/qualification/`, so the P8-04 PR can merge between `P` and the recorded replay; it must **not** edit `governance/tickets/**` (reference tickets in commit trailers only). If a ticket file must change, do it before `P`.

### 2.3 How to freeze `G`

1. The orchestrator declares a merge hold on every PR that touches `intergrax/`, `agents/`, `applications/` or the six material paths, from the moment `G` is chosen until the P8-04 PR is merged. The only exception is a P8 repair loop, which restarts section 4 at S0.
2. `G` = the `development` tip after the last product merge (today, the tip after #575/#576 is a candidate; the global auditor fixes it). Everything that follows is docs, tests, `.github` or tickets and does not void `G`.
3. The freeze check of section 2 is run by every session at its start and by the orchestrator before merging.
4. A freeze is a declaration, not a control (MCP-RC report, item 8; the first MCP-RC freeze was broken). Nothing in the repository enforces it.

## 3. Preconditions and exact inputs

| # | Input | Source / check |
| --- | --- | --- |
| I1 | `G` (`GLOBAL_AUDITED_SHA`), a `development` commit containing all product repairs and, per plan row PDC-P8-01, the MCP-SERVER-X closure publication | IA-GLOBAL manifest `audited_implementation_sha` and `test_execution_sha` |
| I2 | Global audit report (plan name `PDC_INDEPENDENT_AUDIT.md`, verdict `PASS` or `PARTIAL_EXACT_BLOCKERS` per plan row PDC-P8-02), scope, test matrix, findings register, session manifest, logs | `iiac_evidence/PDC-X/global/` (location proposal following `iiac_evidence/PDC-X/D1..D8/`; names UNVERIFIED) |
| I3 | Evidence publication commit `E` (docs-only, descendant of `G`, product-identical to `G`) | the commit that publishes I2; may equal the auditor's last commit |
| I4 | CERT-PDC publication commit `C`: `PDC_DOMAIN_CERTIFICATION_MATRIX.md` and the certifier preflight, descendant of `E` | section 8 |
| I5 | Findings: no open `BLOCKER`/`CRITICAL` in the bound findings register (ACT-N15). With the kept `IIAC-GOV-1` set this is already true; the global findings are not bound mechanically **[gate, precedent]** | |
| I6 | Product diff `G`..`E`, `G`..`C` and `G`..HEAD empty; no material governance path change after `E` | freeze check of section 2 |
| I7 | The decisions of section 10 answered (merge authorization, plain-form naming, certifier-authored `PASS`) | orchestrator and Operator |
| I8 | Delegation checks of `DELEGATION-PDC-STANDING.md` answer 6 and `DELEGATION-PDC-P7-ACT.md` (c) (section 9) | the activation delegate |

If IA-GLOBAL returns `FAIL` or a verdict the Operator has not already scoped, there is no rebind (ACT-N04/N05): route per `DELEGATION-PDC-CLOSE` item 2 (one repair loop).

## 4. Sessions, files and order

Roles (all distinct sessions; `IMPLEMENTER != AUDITOR != CERTIFIER`; the delegate is none of them) **[record: P7-ACT Separation; CLOSE Separation]**.

| Step | Session | Writes (only these) | Validator or check, expected output |
| --- | --- | --- | --- |
| S0 | Orchestrator | merge hold; chooses nothing in the binding | freeze check empty |
| S1 | IA-GLOBAL (new auditor; not an auditor of D1 to D8; never certifies, never fixes) | audit report, scope, test matrix, findings register, session manifest, logs under `iiac_evidence/PDC-X/global/` and the dossier under `platform_domain_completeness/`; covers the 62 paths, every repaired HIGH/MEDIUM finding of T-IGX-041-A..H and the declared exceptions (`DELEGATION-PDC-CLOSE` item 2), plan checks 1 to 15, ECAP 2.C run | `verify_iiac_audit_manifest <its manifest> --repo . --require-handoff-template` prints `verify_iiac_audit_manifest: PASS` (its own manifest may carry `CONDITIONAL`, precedent) |
| S2 | Evidence publication (the auditor or the integration session) | commit `E` (docs only; "pin-only edit" pattern of `fbe400f06`: one commit that fixes the SHAs the dossier cites) | freeze check empty at `E` |
| S3 | CERT-PDC (certifier; not auditor, not implementer, not the delegate) | commit `C`: `PDC_DOMAIN_CERTIFICATION_MATRIX.md` (section 8), certifier preflight dossier under `platform_domain_completeness/` | its own pre-check log (section 5.2) |
| S4 | CERT-PDC (same session is allowed: roles of one certifier; the delegate must differ) | commit `B`: binding manifest, handoff record (CRLF), pre-check and validator logs under `iiac_evidence/PDC-X/P8/activation/` (path proposal, mirrors `MCP-RC/activation/repin_2/`) | section 5.3 |
| S5 | Activation delegate (new dedicated session, R-2 statements in its first message) | OD file `IIAC_OD_ACTIVATION_<...>.md` and phase 1 (record only) and phase 2 (two lines) | section 6 |
| S6 | Orchestrator | the PR, merge after green CI and head match, **with a merge commit** | section 7 |
| S7 | Verification (orchestrator PDC-P9-06 or a separate session) | report | section 7 |

### 4.1 PR plan (minimum)

**One PR** `p8-04-rebind` carrying `E` (if not already on `development`), `C`, `B`, the OD commit, phase 1 and phase 2, in that order. Only these things are required by the gate:

- the commits must be separate and ordered: phase 2 sets `effective_from_sha` to the **phase-1 commit**, which must differ from HEAD (ACT-N12) and must contain an `IN_FORCE` blob with empty `effective_from_sha` and the same ten binding fields **[gate]**;
- `E`, `C` and phase 1 must all be ancestors of the final HEAD, and the CI clone has full history (`fetch-depth: 0` in `unit-tests.yml`);
- **merge commit only**: a squash or rebase merge destroys the phase-1 SHA and ACT-N12 then reports an unreadable activation blob at `effective_from_sha` (derived from the gate, not observed; PR #554 was merged with a merge commit);
- splitting into several PRs buys nothing and costs CI waits; two PRs are only needed if `E` and `C` should reach `development` earlier (they may: they are docs-only, but they are red-class PRs until the rebind lands, see section 7).

### 4.2 SHA layout to bind

| Record field | Value |
| --- | --- |
| `corrective_implementation_sha` = manifest `audited_implementation_sha` = `test_execution_sha` | `G` |
| `audit_execution_sha` = `evidence_publication_sha` | `E` (precedent `a1b3053b4`: both the same commit) |
| `certification_publication_sha` | `C` (must differ from `G`; must be an ancestor of phase 1; precedent `dc1e66fa1`) |
| `governance_implementation_sha` | unchanged `2ae1014116d353bc7742985e62f2bccc2bfac798` |
| `effective_from_sha` = `policy_activation_sha` | phase-1 commit, set in phase 2 only |

Ancestry chain **[gate, ACT-N09]**: `2ae101411` -> `G` -> `E` -> phase 1; `C` -> phase 1; `G != C`.

## 5. The certifier step (S3 to S4)

### 5.1 Artifacts the gate reads **[gate]**

Keep the five bound artifact paths and digests of the current record (the `IIAC-GOV-1` set): this keeps the signed decision's `Audit report digest: c405c65f...` line valid and needs no digest revision **[precedent]**. All five must stay under `docs/project/maintainers/qualification/iiac_evidence/` (ACT-N14). Binding the global audit report instead would need certifier-authored copies under `iiac_evidence/` and a new decision text citing the new digest.

Binding manifest (`stage_id` `IIAC-GOV-1`, `certification_profile` `iiac-v1`, `policy_version` `IIAC-2026-11-05.1`, `verdict` `PASS`, `assurance_level` `process_enforced`):

| Field | Value |
| --- | --- |
| `implementation_session_id` / `implementation_agent_id` | one id (the PDC-X implementer lane; candidate form of re-pin 2: `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER`, extended with `T-IGX-041-A..I`), `implementer`; the id appears **verbatim** in the handoff text (ACT-N17) |
| `auditor_session_id` / `auditor_agent_id` | the IA-GLOBAL session id, `independent-quality-auditor`; also verbatim in the handoff; differs from the implementer and the certifier (ACT-N10) |
| `certifier_session_id` / `certifier_agent_id` | the CERT-PDC session's own id, `certification-governance-agent` (ACT-N17: non-empty) |
| `audited_implementation_sha`, `test_execution_sha` | `G` |
| `audit_execution_sha`, `evidence_publication_sha`, `certification_publication_sha` | `E`, `E`, `C` |
| `audit_report_digest`, `audit_scope_manifest_digest`, `test_matrix_digest`, `findings_digest` | SHA-256 of the working-tree bytes of the five bound files: `c405c65f...7a4e`, `e06ce433ac1c260390885e19cc60b0687be9d8908d78bd38ae4fb5f1102fe56f`, `d5028f194bb7917d07e8df5264075952dcf8757503975042988caba6e89a240b`, `df3648f926dad3199ff40c7c83c7fae4ed7a26bd1c67701a355bc7130073f831` (recompute; the report digest is in the record) |
| `limitations` | array; must list: scope (62 paths, the eight repairs, `G`, `E`, `C`); that `PASS` is the certifier's own, authored from the global audit's scoped verdict and the D1 to D8 `PARTIAL`/`FAIL` dossiers (only valid if decision (b) of section 10 is answered); the declared exceptions of `DELEGATION-PDC-CLOSE` item 1; stale statements of the signed decision text; CI contexts not verified; "does not close PDC-X, does not promote any FRZ status, no production readiness" |
| extras (schema allows) | `handoff_record_path`, `activation_audit_report_path`, `prior_activation_manifest` (the re-pin 2 manifest), `product_session_manifest_paths` (global and D1 to D8 manifests) |

Handoff record: name `IIAC_SESSION_HANDOFF_RECORD_*.md`, not the template, contains `IIAC-2026-11-05.1`, the implementation session id and the auditor session id; its SHA-256 goes into `handoff_digest`. Write it with CRLF line endings (precedent log: an LF-only file made the tamper test a no-op; `file` on the re-pin 2 handoff reports CRLF; `iiac_evidence/**` is `-text` in `.gitattributes`). Not required by gate code (UNVERIFIED as a rule).

### 5.2 Pre-check (log each result with command, HEAD, UTC, exit code)

Reuse `iiac_evidence/PDC-X/MCP-RC/activation/repin_2/scripts/precheck_repin2.py` as the template (edit SHAs only in the certifier's own copy). It must show:

1. ancestry chain of 4.2; `G != C`;
2. product diff via the gate's own `product_paths_changed_between`: `G`..`E`, `G`..`C`, `G`..HEAD all empty; `a1b3053b4`..`G` = the 62 paths of section 1 (or the recomputed set if a repair loop ran);
3. no material governance path changed after `E`;
4. the five bound artifacts: digests equal the record, bytes at `E` equal working tree (ACT-N11 report check);
5. findings register passes ACT-N15;
6. decision text rules of 6.1 for the file in use.

### 5.3 Validators the certifier runs on its binding manifest

```bash
python -m scripts.governance.verify_iiac_audit_manifest <binding-manifest> --repo . --require-handoff-template
python -m scripts.governance.verify_iiac_role_separation <binding-manifest> --strict-product-diff --repo .
```

Expected: `verify_iiac_audit_manifest: PASS` and `verify_iiac_role_separation: PASS`, exit 0 each **[precedent: repin2_03 log]**.

### 5.4 Environment (observed, not assumed)

The gate starts the two companions as `sys.executable scripts/governance/<script>.py` with `cwd` = repo root. In my run (system Python, no `jsonschema`, no `PYTHONPATH`) they failed with `ModuleNotFoundError` and added two `ACT-N10 companion` lines. Run every validator in the project environment (`uv run --frozen --extra dev-ci python -m ...`, one process at a time per `intergrax-test-process-budget`) with `PYTHONPATH` set to the checkout root (PowerShell: `$env:PYTHONPATH = (Get-Location).Path`), as the re-pin 2 logs did. Exact working form on this workstation: UNVERIFIED.

### 5.5 Certifier unit evidence (precedent)

One pytest process: `tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_g0_inventory.py tests/unit/governance/test_iiac_negative_guards.py` at the phase-2 head; re-pin 2 result 50 passed, 1 skipped, about 322 s. Only after phase 2; log under the evidence folder. The MCP-RC contract run of re-pin 2 is not needed for PDC.

## 6. The activation step (S5)

### 6.1 Operator decision file

Recommended: a **new** file `docs/project/maintainers/qualification/IIAC_OD_ACTIVATION_PDC_X_P8_<YYYY-MM-DD>.md`, decision id `IIAC-OD-PDC-X-P8-ACT-N11-<YYYY-MM-DD>`, modelled on `IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md`. The gate requires **[gate: `_operator_errors`, `resolve_evidence_path`]**:

- file name starts `IIAC_OD_ACTIVATION_`, ends `.md`, contains neither `PACKET` nor `DRAFT` in any case, sits directly under `docs/project/maintainers/qualification/`;
- whole lines: `Decision ID: <id>` (equal to the record's `operator_decision_id`), `Policy version: IIAC-2026-11-05.1`, a line containing `Decision scope: IIAC policy activation`, `Decision status: AUTHORIZED`, `Audit report digest: c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e` (the record's digest);
- the exact line `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <ISO-8601 with offset, from the delegate's own clock>`, byte-identical to the record's `operator_activation_line`;
- must **not** contain a run of sixteen underscores, the capitals `NOT SIGNED`, or `Decision status: PENDING`; must not say `EVOL-X`, `ENT-AUDIT-X` or `ARCH-FREEZE` within 80 characters before `CLOSED` (ACT-N18, also scanned in the record);
- the text carries the disclosure that a designated session wrote the line (commit message and provenance note; P7-ACT "Amendment 01 check"), and lists the scope limits, the declared exceptions and "no stage closure, no FRZ promotion".

Fallback (needs an Operator statement, not a delegate act): reuse the signed MCP-RC line unchanged as re-pin 2 did under `OD-MCP-RC-05` section 1 ("the signature stands"; file digest `0e0aed58...5147`). No equivalent statement exists for PDC-X (UNVERIFIED that the Operator would give it).

### 6.2 Phase 1 commit (only `IIAC_POLICY_ACTIVATION_RECORD.json`)

Use the shape of `repin_2/scripts/phase1_record_edit_repin2.py` (JSON, `indent=2`, `ensure_ascii=False`, trailing newline, UTF-8, `-text` path). Set:

| Field | New value |
| --- | --- |
| `effective_from_sha`, `policy_activation_sha` | `""` |
| `corrective_implementation_sha` | `G` |
| `audit_execution_sha`, `evidence_publication_sha` | `E` |
| `certification_publication_sha` | `C` |
| `session_b_verdict` | `PASS` |
| `session_b_manifest_path`, `session_b_manifest_digest` | the binding manifest and its SHA-256 |
| `handoff_record_path`, `handoff_digest` | the handoff and its SHA-256 |
| `operator_decision_path`, `operator_decision_id`, `operator_activation_line` | the new OD file, its id, its line (or the MCP-RC values under the fallback) |
| `operator_decision_class`, `notes` | free text (for example `OD-PDC-X-P8-ACT-N11`; notes name `G`, `E`, `C`, the 62 paths and the previous bindings) |

Unchanged: `status`, `governance_implementation_sha`, `session_b_governance_audit_report_digest`, `audit_report_path`, `audit_scope_path`, `test_matrix_path`, `findings_path`, `policy_version`, `schema_version`. The ten `BINDING_KEYS` must be the same in phase 1 and phase 2 **[gate]**.

Expected at phase 1 (`verify_iiac_policy_activation --require-in-force`), exit 1, exactly four lines **[precedent: repin2_04]**:

```text
IIAC policy activation: FAIL
  - missing activation field: effective_from_sha
  - missing activation field: policy_activation_sha
  - effective_from_sha must be a 40-char lowercase git SHA
  - policy_activation_sha must be a 40-char lowercase git SHA
```

Any ACT-N11, digest, ancestry or companion line here is a defect: stop and report.

### 6.3 Phase 2 commit (two lines)

Set `effective_from_sha` and `policy_activation_sha` to the phase-1 commit SHA, nothing else. Expected, exit 0:

```text
IIAC policy activation: IN_FORCE @ IIAC-2026-11-05.1 from <phase-1 SHA>
```

Gate conditions **[gate]**: the two fields are equal; not equal to HEAD (ACT-N12); the phase-1 blob is `IN_FORCE` with empty `effective_from_sha` and equal binding keys; the working-tree record equals HEAD (commit before running); product diff `E`..HEAD empty; no material path changed.

Commit trailer on every commit: `Ref: governance/tickets/T-IGX-040-C.md` (the closing-wave ticket used by the latest record PRs; the orchestrator may name another), no AI attribution lines (repo rule), one valid `Ref:`. Before pushing a new branch set its upstream to `origin/development` and push with an explicit refspec (user memory: the pre-push ticket-ref gate otherwise scans all history).

## 7. PR, CI, merge, verification

- Open one PR. Pin the head: `gh pr view <n> --json headRefOid` equals the pushed SHA, re-checked after every push (a PR was once merged without its last fix commit).
- Required contexts per the MCP-RC runbook: `ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast`, `ent-audit-pr-gate`; `governance-policy-pin` also exists. `governance-fast` runs `tests/unit/governance/` (includes `test_canonical_policy_is_in_force_and_verifies` and the N12 test) and is red **only** until phase 2 is in the head. Known red classes at `DELEGATION-PDC-CLOSE` item 3: IIAC activation staleness until the rebind and the EBH-7 identity gate until the final re-pin is recorded. `in_progress` is not success (P6 PRs were merged with checks pending). Whether the five contexts are green on any head: UNVERIFIED.
- Merge with a merge commit (section 4.1). Who merges: see decision (a) of section 10.
- After merge: on the merged `development` tip run `verify_iiac_policy_activation --require-in-force` (expect `IN_FORCE ... from <phase-1 SHA>`), `git diff` of product paths between `G` and the tip empty, the freeze lifted only after that, the dependency-bot batch merged afterwards (`DELEGATION-PDC-WAVE2` decision 7).
- A separate terminal verification session is precedent (MCP-RC runbook (e)); whether PDC requires one beyond PDC-P9-06 "verify merged remote HEAD + CI" is UNVERIFIED.

## 8. CERT-PDC session (S3): checklist and matrix skeleton

Role per `OD-PDC-IIAC-SCOPE`: "Consolidates D1 to D8 verdicts into `PDC_DOMAIN_CERTIFICATION_MATRIX`; checks manifests and role separation. Not an auditor and not an implementer." Plan row PDC-P7-CERT: "8 accepted". `PDC_INDEX.md` lists the deliverable as "Not started".

Checklist:

1. State in the first message: new session, dedicated, not a continuation, not an implementation session (R-2); role = certifier; list the sessions it must not equal (implementers of T-IGX-036 and T-IGX-041, IA-D1..IA-D8, IA-GLOBAL, the activation delegate).
2. Read only: D1 to D8 reports and manifests, the global audit set, tickets 041-A..I, `DELEGATION-PDC-CLOSE`, `OD-PDC-IIAC-SCOPE`.
3. Run `verify_iiac_audit_manifest` and `verify_iiac_role_separation --strict-product-diff` on the nine auditor manifests (D1..D8 and global) and record exit codes.
4. Fill the matrix below from the audit reports, the repair tickets and the global audit; never upgrade a verdict that the auditor or an Operator authorization has not issued; the declared exceptions keep their labels and reopen conditions (`DELEGATION-PDC-CLOSE` item 1).
5. Publish it as commit `C` (docs only), with the preflight.
6. Do not: edit an audit file, the gate, a verifier, the protocol, the record; write an `Approved` line; promote an FRZ row; close PDC-X.

Skeleton (values below are what the repository records at drafting; empty cells are for the global audit and the certifier):

| Domain | Audited SHA | Audit verdict (IA-Dn) | Audit PR | Repair ticket / PR | P8 re-verification (IA-GLOBAL) | Declared exceptions (`DELEGATION-PDC-CLOSE` item 1) | Certifier disposition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D1 AGENT-LIFECYCLE | `c08efe46a` | PARTIAL (manifest `CONDITIONAL`) | #560 | T-IGX-041-A / #574 | _pending_ | F-D1-04 (revoked revision without trusted rollback target) | _pending_ |
| D2 CONTEXT-KNOWLEDGE | `c08efe46a` | PARTIAL | #563 | T-IGX-041-B / #575 | _pending_ | none named | _pending_ |
| D3 MODEL-RUNTIME | `c08efe46a` | PARTIAL | #561 | T-IGX-041-C / #569 | _pending_ | four `fixture_strict_adapter_parity` tests (041-C deviation) | _pending_ |
| D4 HOST-RUNTIME | `c08efe46a` | PARTIAL | #566 | T-IGX-041-D / #571 | _pending_ | "no production composition or caller" family (dossier F-04), option-B missing-feature findings | _pending_ |
| D5 EVALUATION | `c08efe46a` | FAIL (row PDC-D5-R015) | #562 | T-IGX-041-E / #568 | _pending_ | remainder of F-D5-02 (MVP promotion gate semantics) | _pending_ |
| D6 RUNTIME-SECURITY | `c08efe46a` | PARTIAL (0 HIGH, 6 MEDIUM) | #565 | T-IGX-041-F / #573 | _pending_ | none named | _pending_ |
| D7 BACKGROUND-SCHEDULING | `c08efe46a` | PARTIAL | #559 | T-IGX-041-G / #570 | _pending_ | F-D7-03 (graceful worker shutdown with host drain); row PDC-D7-R014 rated PASS_SCOPED by Operator authorization | _pending_ |
| D8 ARTIFACT-LIFECYCLE | `c08efe46a` | PARTIAL | #564 | T-IGX-041-H / #572 | _pending_ | none named | _pending_ |

Cross-cutting rows to add: T-IGX-041-I (#576, TENANT-X class-field carriers and carrier-count re-pin, no product path); EBH-7 final re-pin (J, K, L); MCP-RC carry-overs (AF-05, RA-02..RA-04, xfail owner); F-35 and F-56 status. Mapping of the declared exceptions to domains beyond the ones in `DELEGATION-PDC-CLOSE` is by name only; confirm each against the audit report before filling the cell. The per-domain exception text above is from the CLOSE record and the repair ticket titles, not re-derived from the audit rows.

## 9. Where a human or Operator signature is mandatory versus delegable

| Act | Status | Basis (quoted or paraphrased from the record) |
| --- | --- | --- |
| Writing the activation decision file and its line for P8-04 | **Delegable** to a dedicated delegate session | `DELEGATION-PDC-STANDING` answer 6: "A dedicated delegate session, separate from every implementer and auditor, may write the activation lines for PDC-P7-00 ... and PDC-P8-04 (rebind at the global audit SHA)." `DELEGATION-PDC-CLOSE` item 3: "A delegate session writes the IIAC activation record line of P8-04 after confirming that the protocol permits it." Amendment 01 section 0: "S-3 is STRUCK ... No decision class is excluded ... including ... IIAC policy activation." |
| Delegate's mandatory pre-confirmations | Required before writing | STANDING answer 6: no product delta between the audited SHA and the activation HEAD (ACT-N11); `verify_iiac_policy_activation` preconditions and `verify_iiac_role_separation` pass; the P6 evidence is complete; for P8-04 the global audit is published; and "The delegate must first confirm that the IIAC protocol and Amendment 01 permit delegated activation lines. If they do not, the delegate writes nothing and reports to the Operator." |
| Plain `operator @` form of the line | Delegable **only if the commission names it** | Amendment 01 S-2: a designated session "may write the plain form if the commission names it". `DELEGATION-PDC-P7-ACT` (c) names it for PDC-P7-00 only. `DELEGATION-PDC-CLOSE` item 3 says "the activation record line" without naming the form (section 10, (c)). |
| Two-phase record publication | Delegable (P7-ACT (c) assigns it to the activation delegate; the MCP-RC route gave it to the activation certifier) | Choose one split in the commission; both keep roles apart |
| Binding manifest `PASS` authored by the certifier from scoped audits | Precedent only for MCP-RC | `OD-MCP-RC-04` section 3 ("Tak, wzór P6", answer recorded by the orchestrator): the certifier authors `PASS` from `PASS_SCOPED` audits. `DELEGATION-PDC-CLOSE` "Not delegated" lists "the verdict on B6 or certification". Section 10, (b) |
| Merge of the P8-04 PR | **Not covered** by a standing authorization | `DELEGATION-PDC-STANDING` authorization 2 does not count "a change to IIAC protocol or activation files" as docs-only; `DELEGATION-PDC-WAVE2` decision 8: "The implementer never merges its own change to IIAC protocol or activation files, an OD signature, a FRZ line or a closure line." `DELEGATION-PDC-P7-ACT` (d) lifted this for the P7-00 path only, "only for them, and only for a session that did not write the diff". |
| New OD signatures | **Operator only** | STANDING and CLOSE "Not delegated": "OD signatures (new ones)". The activation line under 1 is the delegated exception; a new `Approved:` line elsewhere is not. |
| FRZ promotion; PDC-X closure line (PDC-P9-03, PDC-P9-04) | **Operator only** | CLOSE: "FRZ promotion and the PDC-X closure line (PDC-P9-03, PDC-P9-04)"; "the Operator signs once at the end" |
| Lifting C10/C15; changing the meaning of a gate; edits of hotspot paths outside signed plans; a third repair loop; a finding of a new kind | **Operator only** | CLOSE "Not delegated"; item 2 |
| Auditor and certifier verdicts | Theirs; the orchestrator never edits them | P7-ACT (b) |

## 10. Blockers and Operator decisions found

One question to the Operator (three parts) removes every refusal point I found. Suggested wording, to be put by the orchestrator, not an Operator statement:

- **(a) Merge of the P8-04 PR.** "May the orchestrating session merge the PDC-P8-04 pull request (activation record, OD file, evidence), with a merge commit, after green CI and a head match, although it changes IIAC activation files?" Without it, `DELEGATION-PDC-WAVE2` decision 8 and STANDING authorization 2 forbid the implementer lane to merge, and the MCP-RC runbook treats the merge as "a separate act that needs the operator's instruction at that moment".
- **(b) Bound verdict.** "Does the MCP-RC route (`OD-MCP-RC-04` section 3: the activation certifier authors a binding `PASS` from scoped audits, scope limits in `limitations`) apply to the global PDC audit?" The gate accepts only `PASS` in the binding manifest (ACT-N05); the domain audits are `PARTIAL`/`FAIL` and the global verdict may be `PASS_SCOPED` or `PARTIAL_EXACT_BLOCKERS`; `DELEGATION-PDC-P7-ACT` (c) speaks of a `PASS` audit manifest (the tension named in the re-pin 2 report, open condition 5).
- **(c) Plain form of the line.** "The commission for PDC-P8-04 names the plain form `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>` written by the activation delegate." Needed because Amendment 01 S-2 conditions the plain form on the commission naming it and `DELEGATION-PDC-CLOSE` does not.

If the Operator answers, the orchestrator transcribes the three answers into a short docs-only record (`operator_decisions/DELEGATION-PDC-P8-ACT.md`, under the EBH-7-allowlisted `docs/.../qualification/` tree). Quote answers verbatim with a labelled English rendering.

Other findings:

1. **Stale audit risk is real and mechanical**: any `intergrax/**` merge after `G` fails the rebind in CI, including a late impact-map or classifier fix. Keep K, L, J strictly to tests, `.github` and docs.
2. The signed decision text (if the fallback is used) is stale in the same way as in re-pin 2; the gate does not compare it.
3. The `INTEGRATED_BASELINE_SHA` row of `PDC_SHA_LADDER.md` (`146e7acf1`) is stale for ACT-N11 (it predates the product repairs); `GLOBAL_AUDITED_SHA`, `AUDIT_EXECUTION_SHA`, `EVIDENCE_PUBLICATION_SHA` and `CERTIFICATION_PUBLICATION_SHA` are still `—`. The ladder row edit is docs-only and belongs to PDC-P9-02.
4. The 62 paths include contracts and shared hotspots (`intergrax/contracts/evaluation_contracts.py`, `intergrax/applications/_shared/*`); this is context for the auditor, not a gate rule.
5. D5 has verdict `FAIL` in its dossier; the global audit must show the repaired row `PDC-D5-R015` or record it as a finding. Not decided here.

## 11. UNVERIFIED list

MCP-SERVER-X closed status in the roadmap; the identity-gate allowlist beyond the dossier's text; exact environment command on this workstation; whether the five CI contexts are green on any head; whether a separate terminal verification is required for PDC; names and paths under `iiac_evidence/PDC-X/global/` and `.../P8/activation/`; the content of T-IGX-041-J/K/L (the tickets are not on `development`; write sets are taken from `DELEGATION-PDC-P7-REPIN.md` and `PDC_EBH7_REPIN_DOSSIER.md`); whether the merge-commit requirement is enforced by repository settings; git's reporting of renames out of a product path; per-domain exception mapping beyond the CLOSE text.

## 12. Not done, not claimed

No verdict, no signature, no activation line, no rebind, no PR, no merge, no push, no pytest run, no edit of the gate, a verifier, the protocol, the activation record, a ticket or any audit file. Not stage closure, not FRZ promotion, not production readiness.
