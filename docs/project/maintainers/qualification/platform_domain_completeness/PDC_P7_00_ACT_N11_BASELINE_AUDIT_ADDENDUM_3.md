# PDC-P7-00 - ACT-N11 baseline independent audit, addendum 3 (G6)

Stage `PDC-X/P7-00/addendum_3` · policy `IIAC-2026-11-05.1` · auditor branch `p7-00-baseline-audit` · date 2026-11-14
Parents: [`PDC_P7_00_ACT_N11_BASELINE_AUDIT_DOSSIER.md`](PDC_P7_00_ACT_N11_BASELINE_AUDIT_DOSSIER.md) (F-01..F-34), [`..._ADDENDUM_1.md`](PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_1.md) (F-35..F-40) and [`..._ADDENDUM_2.md`](PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_2.md) (F-41..F-48).

| Item | Value |
|------|-------|
| Delta audited | `324d9dba37cb54e1b74a79a05148d2dcc05e6d09` .. `0fc8badd81c4c3e46a3d0ce262c08e8efba41898` (`mcp-rc`, merge of development `146e7acf1` after B-P6, development frozen on product paths) |
| Activation record at this head | two-phase MCP-RC rebind (phase 1 `92d730821`, phase 2 `9949fe2c3`); audited tree `324d9dba3`; previous pin `2d3eceb6c` |
| Product paths in the delta | 1 (PR #537 = T-IGX-036-X, commit `f209febbe`): `intergrax/integrations/providers/embedding_provider/ollama/register.py` (+25 / -3) |
| **Addendum verdict** | **PASS_SCOPED** (IIAC manifest value `CONDITIONAL`) |
| **Combined verdict** (first audit + addenda 1-3) | **PASS_SCOPED** |
| New findings | 0 HIGH · 1 MEDIUM · 2 LOW · 4 INFO (F-49..F-55) |
| Product change required for the ACT-N11 rebind | none; one decision for the orchestrator/operator (F-49), one carried (F-41) |

Audit only: no product, test, ticket, activation-record or ruleset edit; the only merge is `origin/mcp-rc` into my own audit branch (`a5c1fd514`), no PR, no rebind, no `Approved:` line; only `p7-00-baseline-audit` is pushed.

## 1. Verdict

The one product path does what T-IGX-036-X says and no more. The `ollama` catalog row's factory is now wrapped so that the embedding category builds the embedding integration and every other category goes to the owning model-serving factory with unchanged keyword arguments. Across all 201 (slug, category) pairs of the full default catalog exactly one resolve result differs between `324d9dba3` and `0fc8badd8`: `ollama` as `embedding_provider` (a `TypeError` before, the embedding integration now). The wrapper adds no import beyond the embedding bundle, no credential, tenant, endpoint, process or network handling; the catalog revision generation and digest are identical to `324d9dba3`; ruff, pyright and the tier and boundary gates are clean; the delta's own tests, the catalog and registry unit tests and the MCP-RC contract pass. F-42 (the red EBH-5 embedding cases) and F-07 (inventory tests) are resolved; F-41, F-35 and F-09 are unchanged.

The addendum is scoped, not clean, because (a) the sibling shared-slug row `openai` still has exactly the defect that was fixed for `ollama` and no test pins it (F-49, decision needed), (b) the wrapper's `enabled` and keyword-argument handling are not pinned by tests (F-50), (c) the two-step registration leaves a narrow window (F-52), and (d) F-41 (impact rules for the eight `applications/*/mcp/server.py` modules), F-35 and F-09 remain open at this head.

## 2. Scope verified from git

- `git diff --name-only 324d9dba3 0fc8badd8 -- intergrax agents applications` lists exactly the announced path.
- The 23 MCP-RC paths are byte-identical to `b0e19bf61`.
- No path under `scripts/governance`, `scripts/maintenance`, `scripts/enterprise_audit`, `.github` or `agent-env` changed, and not the G0 matrix. The activation record, its operator decision, binding manifest and certifier report changed as the activation certifier's rebind commits; the rest of the range is docs evidence.
- Other non-docs changes, all read (F-54): `uv.lock` (dependabot opentelemetry bump, `pyproject.toml` unchanged, `uv lock --check` exit 0), `scripts/qualification/pdc_x_inventory.py` and `pdc_x_wbs.py` (outside the ACT-N11 prefixes), `tests/qualification/ebh_5/_package_c_v2_primitives.py` (+5, owning row registered first, no assertion touched), the new `tests/qualification/pdc_x/p6_ecapfix` (7 tests), tickets and plans of T-IGX-036-W/-X. No commit of the range carries an AI-attribution trailer.
- `verify_iiac_policy_activation.py` at this head fails with exactly one line, `ACT-N11 product path changed after the audited SHA: ...ollama/register.py`, as expected. My first attempt evaluated the main checkout instead of this tree (an MSYS-form `PYTHONPATH`); the log is kept as superseded.
- **Combined count: 158, not 159.** `git diff --name-only 2d3eceb6c 0fc8badd8 -- intergrax agents applications` is 158 paths, the same set as at `324d9dba3`: the path was already in it because B-REG-P (`c3d5b7005`) changed it earlier. The delta changes the content of one audited path; this addendum re-audits it. The other 157 paths are unchanged since they were audited.

## 3. The dispatch, probed

| Question | Answer (evidence in `addendum_3/logs/`) |
|----------|----------|
| Does any non-embedding category resolve differently? | No. Of 201 (slug, category) pairs of the full default catalog one differs: `ollama|embedding_provider` (`probes_head.json` vs `probes_base.json`). The factory called directly with enum, string, `None`, absent and unknown category values forwards to the owning factory with unchanged keyword arguments (spy probe P3). |
| Identity, tenant, credential handling? | None added. The embedding integration is catalog-only (docstring: runtime binding pending B3), builds no client, reads no environment, makes no network call. |
| Default endpoints? | None. `base_url` stays `None`, `enabled` defaults to `False`. A configured `base_url` or key is silently dropped, and `enabled` goes through `bool()` (`'false'` becomes `True`); both are carried over from the pre-B-REG-P embedding factory (F-51). |
| Error paths? | Embedding contribution without the owning row still raises `UnknownIntegrationError` and leaves the catalog empty; a second embedding registration raises the same `ValueError` as before; the isolated registration sink leaves the global catalog untouched; all identical to `324d9dba3`. |
| Catalog revision? | Unchanged: generation 0 to 1 to 2 for owning row then embedding, 201 / 205 for full and overriding full bootstrap, in both trees; the digest excludes factories. |
| Atomicity? | The registration is two writes; between them the row advertises the embedding category with the owning factory and resolve raises the old `TypeError` (probe P6, F-52). A 200-round concurrent soak did not hit it. |
| Is the B-REG-P `ollama` regression the same item? | Yes. `7aa9cfa18` registered a second `ollama` row (bootstrap collision), `c3d5b7005` (B-REG-P) merged the rows by augmenting the owning one but left the owning factory in charge of both categories, `f209febbe` adds the per-category dispatch. The probe at `324d9dba3` reproduces the problem statement of the T-IGX-036-X plan exactly. The lineage at the older commits rests on code read; my probe hit an import cycle there. |
| Is F-42 resolved? | Yes, both layers: the `[integration.embedding_provider]` cases pass at `0fc8badd8`; mutants M2 (no dispatch) and M12 (no owning-row registration in the harness) each turn them red again. |
| Is the defect class closed? | No: row `openai` (managed_retrieval + embedding_provider) fails the same way (F-49). |

## 4. Mutation sample

12 mutants of `register.py` and the harness primitive, control green, verdicts only on new failures. Killed: inverted condition, no dispatch, always-embedding, no wrapper registration, registration without `override`, model-serving routed to embedding, harness without the owning row. Survived: embedding branch hard-wired to `enabled=True`, default `enabled=True`, `enabled` ignored, non-embedding branch called without its keyword arguments; one mutant is equivalent (F-50). The first run was wrong (`-k` deselected most tests) and is kept as superseded.

## 5. Own numbers

| Run | Result |
|-----|--------|
| One pytest process, 86 entries: MCP-RC contract (39 files), p6_ecapfix, reg_p, p6_impact, XD-12, EBH-5 package C v2 + 3 gate files, EBH-7 int2e, integration catalog / registry / provider unit tests, rag bootstrap, dep1, pdc_x governance unit tests, F-35 status files | 231 failed, 5176 passed, 1 xfailed (known TRANSFERRED), 1230 s, pytest exited by itself; no MCP test failed; delta tests all pass |
| The 231 ids in short-path checkouts: head / `324d9dba3` / pin | 222 failed (same set) / 222 / 218 of 229 |
| Attribution | 12 new since the pin (F-35 x10, F-41 x2), 210 red at the pin as well, 9 order dependent (F-53) |
| Mutation sample | 7 killed, 4 survived, 1 equivalent |
| ruff, pyright, tier and boundary gates, `uv lock --check` | clean; `check_intergrax_no_applications_imports` exit 1 with the same output hash as the earlier audits |
| F-09 probe (xd_03 alone) | 9 passed, process still alive after 90 s, killed |

## 6. Status of earlier findings at `0fc8badd8`

| ID | Status |
|----|--------|
| F-41 impact rule for the eight `applications/*/mcp/server.py` | **still absent**; `impact_map.py` is unchanged in this range, the eight modules select only the floor, the XD-12 case and `p6_impact` coverage test are red |
| F-42 EBH-5 embedding cases | **RESOLVED** by this delta |
| F-35 ten COMPAT-X event-evolution gates | **open, unchanged** (same 10 fail, pass at the pin) |
| F-09 pytest exit hang | **open, reproduced** with xd_03 alone (listed as debt in T-IGX-036-X) |
| F-07 inventory tests | **RESOLVED** (`pdc_x_inventory.py` classifies ARTIFACT_LIFECYCLE; the three governance unit files pass) |
| F-06 | resolved, unchanged (EBH-5 gate files pass) |
| F-05, others | unchanged |

## 7. New findings

Full evidence: `iiac_evidence/PDC-X/P7-00/addendum_3/iiac_audit_p7_00_addendum_3_findings.json`.

| ID | Sev | Finding | Verification | Disposition |
|----|-----|---------|--------------|-------------|
| F-49 | MEDIUM | The row `openai` has the defect fixed for `ollama`: `resolve(EMBEDDING_PROVIDER, 'openai')` goes through the managed-retrieval factory (config error without a key, `TypeError` with one); out of scope of T-IGX-036-X, not covered by the EBH-5 harness catalog (ollama, hf only); the RAG runtime binder does not use this path, the generic resolve and health path does | probe, code read | decision: fix needs a product path after the freeze (another pin) or accept as debt for IA-D3 with a pinned test |
| F-50 | LOW | `enabled` and keyword pass-through of the wrapper are not pinned by tests (4 of 12 mutants survive) | mutation | test-only addition; route to IA-D3; accept for the rebind |
| F-51 | INFO | `enabled` coerced with `bool()`, other keywords silently dropped in the embedding branch; carried over from the pre-B-REG-P factory; catalog-only integration | probe, code read | accept; address with runtime binding (B3) |
| F-52 | LOW | Two catalog writes leave a narrow window with the embedding category and the owning factory | probe | accept; single-write form if the file is touched again |
| F-53 | INFO | Widened selection: 231 failures, 12 new since the pin (F-35, F-41), 210 red at the pin (provider cutover/migration, agent distribution, duckdb), 9 order dependent; none caused by the delta; the provider unit tests are partly red at the pin and detect less sharply | test run, 3 checkouts | route the 210 to their owners |
| F-54 | INFO | Positive: other non-docs changes (lock file, qualification scripts, harness primitive) weaken no gate | code read | accept |
| F-55 | INFO | Positive: no boundary, credential, endpoint, process or network surface; exactly one resolve result changes; catalog revision unchanged | probe, code read, gates | accept |

## 8. Combined coverage statement

The four audits together cover the **158 unique ACT-N11 paths** that differ between the previous pin `2d3eceb6c` and the audited head `0fc8badd8`: 153 at `fa6ff1c96` (23 MCP-RC + 130 PDC-X), 3 first seen in addendum 1, 2 first seen in addendum 2 (the impact map), and 0 new in this addendum, which re-audits the changed content of one already counted path (`ollama/register.py`). The 23 MCP-RC paths are byte-identical to `b0e19bf61`. Every product or test change after `0fc8badd8` voids the audit for the changed paths; a different re-pin voids it for the whole delta.

## 9. What this addendum does not cover

- The eight per-domain audits IA-D1..IA-D8; F-49 is routed to IA-D3.
- CI contexts, `igx_gov ticket`, the full `tests/unit` tree, `tests/e2e/rag` (needs services), network and live-provider tests; P8 and FRZ (counter 225); documentation-only changes.
- Whether F-41's two tests pass on development alone; the pdc_x suites other than p6_ecapfix, reg_p, p6_impact, XD-12 and the three governance unit files.
- The 210 failures red at the pin and the order-dependent ones were attributed, not diagnosed.
- The `ollama` lineage at the commits before `c3d5b7005` was not exercised (import cycle of my probe there).
- Windows only; the shared `.venv` lacks tzdata (F-34) and duckdb.
- `certifier_session_id`, `evidence_publication_sha` and `certification_publication_sha` in the manifest are placeholders.

## 10. Open conditions for the orchestrator / operator

1. **F-49**: fix (a mirrored dispatch in `embedding_provider/openai/register.py`, a product path, another pin) or accept as recorded debt with a pinned test.
2. **F-41** is unchanged: a map rule for `applications/*` (product path) or the accepted known-red class; do not narrow the test.
3. F-35, F-09 and (optional) F-50 are test-side items; route F-53's 210 to the owners of the integration provider tests.
4. The rebind needs no product change; `verify_iiac_policy_activation` at this head lists the one audited path and nothing else.

## 11. Evidence index

`docs/project/maintainers/qualification/iiac_evidence/PDC-X/P7-00/addendum_3/`: `audit_scope.json`, `test_matrix.txt`, `iiac_audit_p7_00_addendum_3_findings.json`, `attribution_of_regression_failures.json`, `iiac_audit_session_manifest.PDC-X.p7_00.addendum_3.json`, `EVIDENCE_MANIFEST.txt`, `logs/`, `adversarial_scripts/`.
