# PDC-P7-00 - ACT-N11 baseline independent audit dossier (G6)

Stage `PDC-X/P7-00` · policy `IIAC-2026-11-05.1` · auditor branch `p7-00-baseline-audit` · date 2026-11-13

| Item | Value |
|------|-------|
| Audited implementation SHA (= test execution SHA = audit execution SHA) | `fa6ff1c9616a6df28c11b087eb3f1b0dc4097b5f` (branch `mcp-rc`; product tree = merge commit `962c1d670`, `fa6ff1c96` adds only an evidence log) |
| Integrated baseline | `origin/development` `87f4d4f417050fec40c32e4051bf76fb1124d58b` (merge of PR #520) merged into `mcp-rc` |
| Activation pin audited against | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` (`audit_execution_sha` of the `IIAC-2026-11-05.1` activation) |
| Scope | 153 ACT-N11 paths = 23 MCP-RC + 130 PDC-X (derived from git, `iiac_evidence/PDC-X/P7-00/audit_scope.json`) |
| **Dossier verdict** | **PASS_SCOPED** (IIAC manifest value: `CONDITIONAL`, same mapping as the MCP-RC precedent) |
| New findings | 0 HIGH · 6 MEDIUM · 22 LOW · 6 INFO (34) |
| Product change required for the activation rebind | none |
| Test-only change required before merge / CI | 1 (F-06) |

This is the ACT-N11 activation audit only. It is not a certification, not a closure and not one of the eight per-domain audits IA-D1..IA-D8. It issued no `Approved:` line, edited no product, test, ticket or activation record file, did not rebind, did not open a PR or merge, and pushed only `p7-00-baseline-audit`.

## 1. Verdict in one paragraph

No HIGH finding. The 23 MCP-RC paths are byte-identical to the independently re-audited `b0e19bf61` and the MCP-RC contract passes on the integrated tree exactly as claimed (481 passed, 1 strict xfail). The 130 PDC-X paths add or tighten governance rather than bypassing it: the hard-coded signing key fallback is gone, tool input injection defence is wired into the canonical path, retrieval poisoning defence defaults ON, plugin provenance, tenant-scoped evaluation registry, governed artifact read, durable audit trail and strict-config validation are new fail-closed building blocks, and I found no new import of `agents/` or `applications/` from `intergrax/`, no credential, vendor endpoint or new process/network/dynamic-code surface in the added lines. The verdict is scoped, not clean, for four reasons: (a) six MEDIUM design weaknesses in new capabilities (F-01..F-05, and the test regression F-06), none of which breaks an invariant that held at the pin; (b) several hardening functions have no production caller, so their fail-closed behaviour is not yet a property of a composed host (F-04); (c) six declared-open gaps are pinned as strict xfail, one of them a tenant-isolation gap in the memory plane (F-05); (d) the PDC-X suite is not green on the baseline (10 failures: one new regression, two new inventory tests, seven pre-existing). Everything in (a)-(d) is routed to the domain audits, to a test-only repair, or to the orchestrator; none needs a product change before the ACT-N11 rebind.

## 2. Scope and method

1. Scope from git only: `git diff --name-only 2d3eceb6c fa6ff1c96 -- intergrax agents applications` gives 153 paths (24 added, 129 modified, none under `agents/`); `git diff --name-only 6a6279a49 b0e19bf61 -- intergrax applications agents` gives 23 paths, all inside the 153; the remaining 130 are PDC-X (+12,032 / -3,544 lines). 15 non-merge commits touch the 130 paths and every one carries `Ref: governance/tickets/T-IGX-036-{J..T}.md`. The tickets and PDC documents were used as claims to check, not as truth.
2. MCP-RC 23: `git diff b0e19bf61 fa6ff1c96 -- <23 paths>` is empty; the existing delta re-audit (`MCP_SERVER_X_RC_INDEPENDENT_REAUDIT_DOSSIER.md`, PASS_SCOPED) is relied on and its design was not re-audited. The mandatory contract was run once as a regression check.
3. PDC-X 130: my own scans of every added line (imports, credentials/endpoints, process/network/dynamic-code surfaces), a manual read of the highest-risk files (signing, audit trail, plugin admission/discovery, governance pipeline, security wiring, contracts, strict-config), three read-only review helpers over the remaining 34 + 32 + 44 files, auditor verification of every helper claim I rate MEDIUM or that I could cheaply reproduce, 22 throw-away probes, a 12-mutant sample, the PDC-X batch tests, ruff and the repository boundary gates.
4. Findings carry a `verification` field. Only `PROBE_CONFIRMED`, `CODE_READ_BY_AUDITOR` and `TEST_RUN` are verified by me; five LOW findings (F-18, F-24..F-28) come from a review helper and are leads for the domain audits, not facts.

## 3. Own numbers

| Run | Result |
|-----|--------|
| MCP-RC contract (39 files, list passed from Python), `fa6ff1c96` | 481 passed, 1 xfailed (strict, TRANSFERRED), exit 0, 975.51 s - matches the claim (562.72 s on the implementer's machine) |
| PDC-X batches (95 files: `tests/qualification/pdc_x/**` 80, 3 governance tests, 12 existing modified tests) | 10 failed, 1064 passed, 2 skipped, 6 xfailed, 251.15 s; pytest then did not exit (F-09) |
| per directory (own process each) | d1 65 · d2 41 · d3 154+1s · d4 36 · d5 79 · d6 134 · d7 52 · d8 117 · p6_repair 99+1s+2x · reg_p 3 · xd 76+4x |
| the 10 failures at the pin (3 existing files, detached checkout of `2d3eceb6c`) | 13 failed, 110 passed; 7 of the 10 baseline failures are red at the pin too, 4 audit-ideal tests and 2 EBH-5 parity tests were repaired by the delta |
| ruff, 153 product paths + pdc_x tests + 3 governance tests + scripts | all checks passed |
| `uv lock --check`; 9 tier/boundary/maintenance scripts | exit 0 |
| `check_intergrax_no_applications_imports.py` | exit 1, 12 `intergrax/runtime/**` files, identical at the pin, none touched by the delta (F-33) |
| probes | 22, 0 probe errors |
| mutants | 12; 10 killed, 2 survive after the wide re-run (F-29) |

Environment deviation: the shared `.venv` lacks `tzdata`, which the baseline `pyproject.toml` now declares; the runs used a copy of tzdata 2026.3 from `build/dep2-core-gate` through `PYTHONPATH`. Nothing was installed.

## 4. The 23 MCP-RC paths

No finding. Byte-identical to `b0e19bf61`; contract 481 / 1 on the integrated tree. Not covered: anything the delta re-audit listed as not covered (CI contexts, G-DOC-01..12, shared harness modules, production host cancellation after the bound, real cross-loop ingress, real product factory composition AF-05); the open conditions of that dossier (owners for AF-05 and AF-16) are untouched by this audit.

## 5. The 130 PDC-X paths - what the diff does to ACT-N11 invariants

| Invariant | Result |
|-----------|--------|
| Governance / authorization | Strengthened: `AgentRuntimeGovernancePipeline` audits the basic capability denial before re-raising; governed artifact read goes through `authorize_tool`; plugin entry points can require publisher attestation. Weak points: F-01, F-10, F-21 |
| Identity / tenancy | Tenant-scoped evaluation registry, tenant-keyed SQLite artifact store (all SQL parameterized, no path from ids), tenant-scoped audit trail. Weak points: F-01, F-03, F-05 (FINDING-6), F-25 |
| Execution / cancel / deadline | New recurrence, retention sweep, queue retry budget, host lifecycle metrics. Weak points: F-02, F-13, F-14, F-15, F-16 |
| Trace / audit / evidence | Durable evidence-plane audit trail (replaces nothing silently), eval evidence. Weak points: F-20, F-24 |
| Tool runtime / MCP surface | `plan_context_invocation` spills large output only with an artifact store; `storage_presigned_url` now always raises (fail-closed, F-31); `mcp_result_bounds` is MCP-RC. No new tool, no new MCP surface found |
| Import boundary | No new import of `agents`/`applications` from `intergrax/` (own scan plus three helpers, including lazy imports); the pre-existing violations in 12 untouched files are F-33 |
| Secrets | Removed the `harness-dev-signing-key` fallback; `raw_secret_gate` added (heuristic, F-30); `SecurityConfigurationError` never carries a secret (probe S1, test d6). Demo seed stays reachable for non-strict profiles (F-23) |
| New network/process surfaces | None added (no subprocess, socket, eval/exec, pickle, new importlib use beyond moving `entry_points` into `select_entry_points`); `HostDeploymentProfile` now accepts unknown flat keys (F-22) |
| Hardcoded credentials / vendor endpoints | None found in the added lines |
| Disabled / skipped / loosened tests | No skip, `importorskip` or non-declared xfail added; assertions were retargeted (F-32); 6 strict xfails are declared open gaps (F-05); one previously green gate is now red (F-06) |

## 6. Findings

Full evidence, verification class and disposition per finding: `iiac_evidence/PDC-X/P7-00/iiac_audit_p7_00_findings.json` (real `findings` array).

| ID | Sev | Area | Finding | Verification | Disposition |
|----|-----|------|---------|--------------|-------------|
| F-01 | MEDIUM | D8 | Governed artifact read: workspace scoping optional; tenant-level capability reads any artifact of the tenant | probe | IA-D8 |
| F-02 | MEDIUM | D8 | Retention sweep: audit optional (worker bootstrap passes none), delete removes versions written after the expiry selection | probe | IA-D8 |
| F-03 | MEDIUM | D7 | Recurrence occurrence id has no tenant; conflict swallowed, chain ends | code + probe | IA-D7 |
| F-04 | MEDIUM | cross | Signing, audit-trail, strict-config, plugin-provenance, tenant-scoped evaluation and skill lifecycle have no production caller; fail-closed only on opt-in | code | IA-D6, orchestrator |
| F-05 | MEDIUM | cross | Six declared-open gaps as strict xfail incl. tenant-bound memory plane serving a foreign-tenant identity | test run | IA-D2/D3 |
| F-06 | MEDIUM | tests | EBH-5 `sqlite_closed_world_accountability` red (`39 == 38`), green at the pin | test run | test-only fix |
| F-07 | LOW | tests | New PDC-X inventory tests red (3 unmapped documents) | test run | owner regenerates |
| F-08 | INFO | tests | 7 failures pre-existing at the pin (README x5, audit-ideal x2) | test run | accept as pre-existing |
| F-09 | LOW | tests | pytest does not exit after `test_xd_03_agent_governance.py` | test run | test owner |
| F-10 | LOW | D8 | Governed read ignores the returned authorization decision | probe | fix `_guard` |
| F-11 | LOW | D1 | Skill upgrade not restored on `BaseException` | probe | IA-D1 |
| F-12 | LOW | D1 | Skill upgrade ordering ignores pre-release (`1.2.3`→`1.2.4-rc.1` admitted) | probe | IA-D1 |
| F-13 | LOW | D4 | Retry budget redelivers crashed tasks with `idempotency_store=None` (opt-in) | code | IA-D4 |
| F-14 | LOW | D7 | Recurrence materialization in `finally` for dead tasks; failure only logged | code | IA-D7 |
| F-15 | LOW | D2 | Memory retention task: producer-controlled `as_of`, unbounded thread join | code | IA-D2 |
| F-16 | LOW | D4 | Lifecycle metric publish unbounded in shutdown path | code | IA-D4 |
| F-17 | LOW | D2 | Raising reindex hook breaks retrieval | probe + code | IA-D2 |
| F-18 | LOW | D2 | Freshness identity from mutable metadata, absolute path stamped | helper only | IA-D2 verify |
| F-19 | LOW | D2 | Egress filter after required-source validation | code | IA-D2 |
| F-20 | LOW | D6 | Audit trail lists oldest 1000, value secrets accepted, regions are bookkeeping | probe | IA-D6 |
| F-21 | LOW | D6 | Provenance bound to mapping key, installed digest not checked | probe | IA-D6 |
| F-22 | LOW | D4 | `HostDeploymentProfile` `extra="allow"`, typo accepted, wide bind accepted | probe | IA-D4 |
| F-23 | LOW | D6 | Demo seed implicit for non-strict profiles | probe | IA-D6 |
| F-24 | LOW | D5 | Eval evidence digests self-attested, comparisons caller-supplied | helper only | IA-D5 verify |
| F-25 | LOW | D5 | Eval registry id oracle and lock removal | helper only | IA-D5 verify |
| F-26 | LOW | D1 | Upgrade admission gaps | helper only | IA-D1 verify |
| F-27 | LOW | D8 | Bundle builder follows symlinks, publishes digest | helper only | IA-D8 verify |
| F-28 | LOW | D2/D8 | Store/retention edge cases | helper only | IA-D8 verify |
| F-29 | LOW | tests | Two redundant tenant checks unpinned (mutants P6, P9 survive) | test run | IA-D6/D8 |
| F-30 | INFO | D6 | Secret gate evasion matrix | probe | accept |
| F-31 | INFO | contracts | Stricter sha256 validators; presign tool always raises | probe | accept |
| F-32 | INFO | tests | Retargeted assertions; MVP gate docstring vs empty G0/G2 symbol tuples | code | accept |
| F-33 | INFO | boundary | Boundary gate red at pin and baseline in 12 untouched files; chaos gate path bug | test run | pre-existing |
| F-34 | INFO | env | Shared `.venv` lacks tzdata | test run | note in activation record |

## 7. Test-weakening check for the delta

Compared the 27,676-line test diff and the 29 modified existing test files. No skip, `importorskip`, `--deselect` or `noqa`-hidden assertion was added; the 44 removed `assert`/`pytest.raises` lines are formatting reflows or retargeting (EBH-5 derived sets instead of the literal `56`, GR-10 AST gates retargeted to the development commits `6700ff756`/`716ed746f`, README asset roots) and, for MCP, the MCP-RC delta already reviewed. Six new strict xfails are declared gaps (F-05). One existing gate was left red by the delta (F-06). `scripts/gates/check_mvp_promotion_gates.py` now only denies and requires a governance ticket `Approved:` line; its docstring claims symbol checks for G0/G2 whose symbol tuples are empty (F-32).

## 8. Open conditions and recommendations

For the orchestrator / operator (none blocks the ACT-N11 rebind by itself):

1. F-06 is a test-only repair; it should land before this baseline is merged or CI contexts are read.
2. Route F-01..F-05 and the per-domain leads in the findings JSON (`routed_to_domain_audits`) to IA-D1..IA-D8 as mandatory inputs; F-04 (host composition wiring) needs an orchestrator decision on whether it is a PDC-X claim.
3. Track the held B-P6-R item 11 write-set amendment (FINDING-6) and the ENT-AUDIT impact map gap (FINDING-8): until the impact map covers the PDC-touched modules, a change to them selects no domain check.
4. Operators on Windows must sync the environment (tzdata) before running the suite (F-34).
5. Any product or test change after `fa6ff1c96` voids this audit for the changed paths.

## 9. What this audit does NOT cover

- The eight per-domain PDC-P7 audits IA-D1..IA-D8: findings routed to them are leads; domain completeness, requirement closure and replay are theirs.
- CI contexts / GitHub Actions, the enterprise and nightly profiles, `igx_gov ticket`, the full `tests/unit` tree, EBH replays, any network or live-provider test.
- P8 (MCP-SERVER-X) and any FRZ change (counter stays 225); documentation-only changes and G-DOC-01..12 (the 344 changed docs and the 71 tickets were used only as claims).
- The 23 MCP-RC paths' design (relied on the existing delta re-audit) and anything that audit lists as not covered.
- Files outside the 153 paths (scripts, agent-env, `.claude` skills, tests) except as stated in sections 5 and 7: `pyproject.toml`/`uv.lock` were read (tzdata only), `scripts/gates/check_mvp_promotion_gates.py` and two maintenance checks were read, nothing else.
- Concurrency, performance, durability and crash behaviour beyond what the probes and tests above exercise; Windows only; no PostgreSQL provider path; no real object storage or exporter.
- Findings F-18 and F-24..F-28 were reported by read-only review helpers and not reproduced.
- Full-code review of all 130 files: the three helpers read a subset fully and the rest by diff (their file lists are in their reports, not archived); my own full reads are limited to the files named in section 2.
- Mutation coverage: 12 mutants are a sample, not a measurement.
- `certifier_session_id`, `evidence_publication_sha` and `certification_publication_sha` in the manifest are placeholders for a later certifier.

## 10. Independence and provenance

Not the implementer of MCP-RC or of any PDC-X batch, not an earlier MCP-RC auditor or certifier, not a signing delegate. All 15 product-touching PDC-X commits and the MCP-RC corrective commits are by the same git author, so commit authorship says nothing about session independence; independence rests on session separation as stated by the orchestrator.

## 11. Evidence index

`docs/project/maintainers/qualification/iiac_evidence/PDC-X/P7-00/`: `audit_scope.json`, `test_matrix.txt`, `iiac_audit_p7_00_findings.json`, `iiac_audit_session_manifest.PDC-X.p7_00.json`, `EVIDENCE_MANIFEST.txt`, `logs/` (contract, PDC-X run, pin comparison, hang bisect, ruff, gates, probes, mutation, verifier outputs, `NOTES.txt`), `adversarial_scripts/` (scope derivation, runners, probes, mutation sample, bisect).
