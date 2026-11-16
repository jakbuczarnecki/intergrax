# T-IGX-024-A — ENT-AUDIT-X formal closure under IIAC: W0 reconciliation and W1 corrective batch

## Scope

Master plan: `agent-env/plans/pasted-content-id-00e5-integrax-expressive-bubble.md` (target status
`ENT-AUDIT-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`; this ticket covers waves W0 and W1 only).

**W0 (docs only, branch `t-igx-024-a-docs`)**

1. W0-04: status overlay `docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_IIAC_CLOSURE_MATRIX.md` (status only, requirements by reference) and append-only ladder rows (IIAC activation, EVOL closure/handoff/B6 pins).
2. W0-05: one stale-text batch (SUPERSEDED-BY-IIAC pointers, factual roadmap/CAAP/FRZ-checklist/WBS fixes that do not promote any status). Excludes every path in `MATERIAL_GOVERNANCE_PATHS` and all EVOL-X artifacts.

**W1 (code and tests; lanes in worktrees under `<parent>/_wt/intergrax/`, merged A, then B, then C by PR)**

| Lane | Defect | Files |
|------|--------|-------|
| A | D-1 validator rejects the current disposition (CI red on every PR and the nightly) | `scripts/enterprise_audit/master_plan_ssot_lib.py`, `tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py` |
| B | D-2 baseline promotion is human-only (no IIAC/delegated path; GAP-RC-OD-TRUST, C001/C002/C003/C042/C057/C065) | `intergrax/platform/enterprise_audit/baseline.py` and its tests |
| C | D-3 closure preflight (Gate A/B split; fail-open on `*unset*` pins; wrong EVOL key; manifest path), D-5 OD-ENT-02 verifiers, D-6 status JSON and ladder fields | `scripts/qualification/ent_audit_x_formal_closure_preflight.py`, `scripts/governance/verify_ent_audit_branch_protection.py`, `scripts/governance/verify_od_ent_02_compensating_control.py` and their tests |

Then: requalification at the merged tip (TES'), B5 evidence publication, handoff to the independent IIAC auditor (W2).

**Corrective loop 1 (plan section 15, W2-03; triggered by the first independent audit, verdict CONDITIONAL on `audit_execution_sha` 15347255ac62dbccfb7b64762b4c0f8af7ec38fa, 2 HIGH / 11 MEDIUM / 5 LOW / 2 INFO)**

Fixed in this loop (minimal diffs, `scripts/` and `tests/` only, no `intergrax/` change, so ACT-N11 stays satisfied):

| Finding | Change |
|---------|--------|
| F-02 (HIGH) `promote-baseline` judged the OD-ENT-01 gate against the current working directory | `scripts/enterprise_audit/run_enterprise_audit.py` passes `repo_root` of its own repository (this edit is qualification-sensitive and therefore requires a new TES) |
| F-01 (HIGH) B5 digests only verify on a CRLF checkout | `.gitattributes` marks every new `qualification_at_*` tree `-text` (legacy 89c76eb7 tree waived explicitly); `publish_b5_evidence_bundle.py` writes the index with LF; the ladder-pinned tree must verify strictly (no skip) once it is not the legacy tree |
| F-09 preflight accepted a fabricated manifest and pass record | preflight now runs `verify_iiac_audit_manifest`, requires real ancestor commits, report/evidence-index digests that match the files, and a pass record that cites the verdict and `audit_execution_sha` |
| F-10 compensating route unsatisfiable by default wiring | preflight reads the active branch rules and judges ruleset controls against them |
| F-11 LIVE verifier required context strings GitHub never emits | verifier accepts the bare job names (and the exact `Enterprise audit (ENT-AUDIT-X) / <job>` form) |

Carried as declared limitations (not fixed here; the cause is under `intergrax/`, qualification-sensitive, or a documented design property): F-03, F-04, F-05, F-06, F-07, F-08, F-12, F-13, F-14, F-15, F-16, F-17, F-18, F-19, F-20 (details: audit report `IIAC_ENT_AUDIT_REPORT.md`, `findings.json`). Requalification at the new TES, republication of the B5 tree and a second independent audit follow.

## Constraints (must be honoured)

- No change to any of the six `MATERIAL_GOVERNANCE_PATHS` (`scripts/governance/iiac_activation_gate.py:40-49`, incl. the IIAC protocol) and no change to the IIAC activation record; `verify_iiac_policy_activation --require-in-force` stays IN_FORCE.
- No change to EVOL-X artifacts (history, status JSON, `evol_x_closure_publication_preflight.py`).
- Ladder is append-only. New exact-name `TEST_EXECUTION_SHA` / `EVIDENCE_PUBLICATION_SHA` rows go below the current ones (`parse_sha_ladder_table`: last wins); old rows get a suffix.
- `ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md` must keep: TES pin, `Audited IMPLEMENTATION_SHA`, `b6_independent_handoff`, `## External blockers (GAP-B)`, "no pass declared" (`mechanical_readiness_check.py:368-384`).
- Thresholds are not lowered; functional code is not changed to satisfy documentation.
- The implementer declares no PASS, promotes no FRZ-AUD row, and writes no CLOSED text. No claim of equivalence between LIVE and COMPENSATING branch protection.
- PRs are merged with a merge commit (not squash or rebase), so SHA pins stay valid.
- Verification is serial: one `uv run pytest` per step (`intergrax-test-process-budget`).

## Plan digest

Plan digest: c1e2a3a975f80bbe791aa7a8e208b4e0fe012c9560fa32a9631f32d9213b96ed

(computed by `scripts/governance/plan_digest.py` over the plan file including its section 21 W0 execution log; if the plan file is edited again, recompute and re-record the digest before the operator approval line is written.)

## Approved

(no `Approved:` line exists; none is written by the implementer. Same form as `T-IGX-023-A`: the batch runs on the operator's explicit chat instructions of 2026-11-08. The master plan was approved in chat via plan mode; branch `t-igx-024-a-docs` and this ticket were approved by the operator in chat; the standing authorization below was agreed by the operator in chat on 2026-11-08 ("tak, zgadzam się na tę regułę, zapisz ją i ruszaj"). If a gate requires the literal `Approved: <operator> @ <ISO8601>` line, the operator must write it; it must not be synthesized.)

## Standing authorization (operator, chat 2026-11-08)

Recorded verbatim so that fresh sessions (auditor, certifier, signing session) can see it; sessions cannot see the chat:

> „W zakresie planu ENT-AUDIT-X (T-IGX-024-A) udzielam z góry zgody na: push, PR, merge commit, workflow_dispatch, worktree i subagenty oraz na uruchamianie niezależnych sesji audytora, certyfikatora i podpisującej na podstawie rekordu delegacji. Zatrzymaj się przy FAIL/CONDITIONAL po 2 pętlach, przy każdej zmianie poza planem i przed publikacją zamknięcia."

Interpretation (binding on every session working under this ticket):

1. Covered: `git push` of `t-igx-024-a-*` and `ent-w1-*` branches; PR creation; merge **with a merge commit** (never squash/rebase); `workflow_dispatch` of the ENT workflows; worktrees under `<parent>/_wt/intergrax/`; sub-agents; launching the independent auditor, certifier and signing sessions as separate sub-agents with fresh context.
2. Disclosed to and accepted by the operator: independent sessions run as sub-agents of the orchestrator, so the assurance level is `process_enforced`, not cryptographic. The manifest `limitations` must say so. Session ids in manifests are the real harness ids; none may be invented.
3. **Not covered and not delegable:** the operator's delegation record for the signing session (O-1) and any `operator` / `operator-delegate` signature line. Until the operator signs that record, W3-01..W3-03 and W5 OD-ENT-08 cannot be written. Work up to and including W2 is unaffected.
4. Hard stops regardless of this authorization: a failing gate whose cause is outside this plan; auditor FAIL/CONDITIONAL after the second loop; any change outside the plan (new material governance path, EVOL-X artifact, lowered threshold); anything that would require fabricating a signature, id, timestamp or test result; and the **final publication** (OD-ENT-08, roadmap CLOSED headline, W5 phase 1), where the orchestrator pauses for operator review before the commit.
5. Nothing here promotes a status, certifies an implementation, or waives an independence rule.

## Test contract

```powershell
uv run python scripts/governance/verify_iiac_policy_activation.py --require-in-force
uv run pytest tests/qualification/ent_audit_x/unit tests/qualification/ent_audit_x/integration tests/qualification/ent_audit_x/l3 -q
uv run python scripts/enterprise_audit/validate_post_implementation_master_plan.py
uv run python scripts/enterprise_audit/check_manifest_annex_c_alignment.py
uv run python scripts/enterprise_audit/mechanical_readiness_check.py
uv run python -m scripts.governance.igx_gov ticket
```

W0 (docs only) needs only the first, third and last command. W1 additionally needs a green `Enterprise audit (ENT-AUDIT-X)` run on each PR head (read the actual run) and a nightly `workflow_dispatch` FULL run on TES'. Exact flags are confirmed from the scripts before use.

## Out of scope

IIAC audit and certification (W2), operator decisions and baseline promotion (W3), certifying FREEZE and FRZ-AUD promotion (W4), formal closure (W5), ARCH-FREEZE handoff (W6), Docker, vendor integrations, production configuration, PROD-Q, edits of the IIAC protocol or Amendment 01. PR merges with a merge commit are covered by the standing authorization above.

## Lifecycle state

in_progress
