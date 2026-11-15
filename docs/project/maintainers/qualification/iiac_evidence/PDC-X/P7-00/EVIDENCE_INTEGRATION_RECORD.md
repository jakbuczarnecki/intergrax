# PDC-P7-00 evidence integration record (T-IGX-038-F)

Role: evidence integration only. This record is written by a session that is neither the implementer, the auditor nor a certifier; it carries no audit verdict, no certification, no activation or rebind, no FRZ change (counter stays 225) and no `Approved:` line. It follows the choices documented in `iiac_evidence/PDC-X/MCP-RC/EVIDENCE_INTEGRATION_RECORD.md`.

## Commit ladder on branch `mcp-rc-evidence-2`

| Step | SHA | Content |
|------|-----|---------|
| Base (`origin/mcp-rc`) | `086c087bd` | RC contract evidence on the re-baselined tree; product tree is merge `19d552bee` |
| Evidence publication commit | `6b195980d20067449ade0d2c2fd2280781c5a25d` | 56 files imported byte-for-byte, no content change |
| Pin-only edits + this record | the commit that adds this file (its own SHA cannot be stored in it; read it from `git log`) | six pin-only field edits below, this record |

## Imported evidence (evidence publication commit)

Source: `origin/p7-00-baseline-audit` at `6b89a368ef2d667c91b78fa8fcbbb7a2f51370fc`. That branch contains merges of `mcp-rc`, so it was not merged. The import is the set of files it adds relative to `19d552bee`, the last `mcp-rc` commit it merged (`2bbcca89c`, second parent). Verified: 56 paths, all additions, none outside `docs/`, none already present on `origin/mcp-rc`, blob identity with the source branch for all 56.

| Content | Audit |
|---------|-------|
| `platform_domain_completeness/PDC_P7_00_ACT_N11_BASELINE_AUDIT_DOSSIER.md`, evidence under `iiac_evidence/PDC-X/P7-00/` | first audit of `fa6ff1c96` (dossier verdict `PASS_SCOPED`, manifest verdict `CONDITIONAL`) |
| `platform_domain_completeness/PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_1.md`, evidence under `iiac_evidence/PDC-X/P7-00/addendum_1/` | addendum 1, delta `fa6ff1c96..19d552bee` (dossier verdict `PASS_SCOPED`, manifest verdict `CONDITIONAL`) |

## Pin-only edits (second commit)

Only these six field values changed; every other byte of both auditor-authored manifests is untouched. Neither manifest is an input of any digest it records (the digests cover the audit dossier, scope file, test matrix and findings file, none of which was edited); all eight recorded digests were re-computed with `sha256sum` before the edit and match. The addendum's `prior_audit_manifest` field only names the first manifest by path, no digest is recorded.

| File | Field | Old | New |
|------|-------|-----|-----|
| `iiac_evidence/PDC-X/P7-00/iiac_audit_session_manifest.PDC-X.p7_00.json` | `certifier_session_id` | `PENDING-CERTIFIER-PDC-P7-00` | `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` |
| same | `evidence_publication_sha` | `0000000000000000000000000000000000000000` | `6b195980d20067449ade0d2c2fd2280781c5a25d` |
| same | `certification_publication_sha` | `0000000000000000000000000000000000000000` | `6b195980d20067449ade0d2c2fd2280781c5a25d` |
| `iiac_evidence/PDC-X/P7-00/addendum_1/iiac_audit_session_manifest.PDC-X.p7_00.addendum_1.json` | `certifier_session_id` | `PENDING-CERTIFIER-PDC-P7-00` | `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` |
| same | `evidence_publication_sha` | `0000000000000000000000000000000000000000` | `6b195980d20067449ade0d2c2fd2280781c5a25d` |
| same | `certification_publication_sha` | `0000000000000000000000000000000000000000` | `6b195980d20067449ade0d2c2fd2280781c5a25d` |

(Paths above are relative to `docs/project/maintainers/qualification/`.)

Choices, identical to the MCP-RC record and the P6 precedent (`MCP_SERVER_X_P6_TERMINAL_DOSSIER.md`, "Pin-only certifier edits"):

- `evidence_publication_sha` is the evidence publication commit above.
- `certification_publication_sha` cannot hold the SHA of the commit that edits it, so it carries the evidence publication SHA.
- `certifier_session_id` is the placeholder designation `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13`. Neither the manifest schema nor `validate_manifest_dict` requires anything but a non-empty string that differs from the implementation and auditor session ids; it does. **No certifier preflight for PDC-P7-00 has been performed:** the value reserves a designation, it does not attest a session. A real certifier preflight must either adopt this id or replace it by its own pin-only edit.

## Known residue (left unchanged on purpose)

- Each manifest's `LIM-publication-pending` limitation still reads "certifier_session_id is a placeholder ... 40 zeros until a certifier publishes". It is auditor-authored prose, outside the pin-only scope, and is now historical.
- `LIM-pre-activation` in both manifests stays true: `verify_iiac --require-in-force` is expected to fail until the PDC-P7-00 rebind. It was not run here.

## Verifier run after the pin-only edits

Run from the worktree root, one process at a time (log: `.tmp/session/mcp-rc-evidence-2/verify_iiac_manifests_after_pin.log`, not committed).

```text
## first audit verify_iiac_audit_manifest
verify_iiac_audit_manifest: PASS            exit=0
## first audit verify_iiac_role_separation --strict-product-diff
verify_iiac_role_separation: PASS           exit=0
## addendum_1 verify_iiac_audit_manifest
verify_iiac_audit_manifest: PASS            exit=0
## addendum_1 verify_iiac_role_separation --strict-product-diff
verify_iiac_role_separation: PASS           exit=0
```

No test suite, mutation harness or activation gate was run; the audits' own results stay as published by the auditor.

# Round 3: addendum 2 (branch `mcp-rc-evidence-3`)

Additive section; rounds 1 and 2 above are unchanged. Same role limits as above: evidence integration only, no verdict, no certification, no rebind, no FRZ change (counter stays 225), no `Approved:` line.

## Commit ladder

| Step | SHA | Content |
|------|-----|---------|
| Base (`origin/mcp-rc`) | `14e1322df` | product tree identical to the audited head `324d9dba3` (`git diff --name-only 324d9dba3 14e1322df -- intergrax agents applications` is empty; `324d9dba3` is an ancestor of `14e1322df`) |
| Evidence publication commit | `29d66ae3fcc7a06a919d7e07541740841e8abfe8` | 27 files imported byte-for-byte, no content change |
| Pin-only edit + this section | the commit that adds this section (its own SHA cannot be stored in it; read it from `git log`) | three pin-only field edits below, this section |

## Imported evidence

Source: `origin/p7-00-baseline-audit` at `3fd4d9160`. It contains merges of `mcp-rc`, so it was not merged. The import is the set of files it adds relative to `324d9dba3`, the last `mcp-rc` commit it merged (second parent of `bbe54afbb`): 27 paths, all additions, none outside `docs/`, none present on `origin/mcp-rc`, blob identity with the source branch verified for all 27. The set equals the files the branch adds relative to `origin/mcp-rc` itself. It consists of the addendum 2 dossier (`platform_domain_completeness/PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_2.md`, delta `19d552bee..324d9dba3`, dossier verdict `PASS_SCOPED`, manifest verdict as authored) and the `iiac_evidence/PDC-X/P7-00/addendum_2/` folder. The first-audit and addendum 1 manifests on the auditor branch and on `mcp-rc` are byte-identical (the pin-only versions of round 2), so nothing of rounds 1 and 2 was re-imported.

## Pin-only edits

Only these three field values changed; every other byte of the manifest is untouched. The manifest is not an input of any digest it records; the four digests (dossier, scope file, test matrix, findings file) were re-computed with `sha256sum` before the edit and match.

| File | Field | Old | New |
|------|-------|-----|-----|
| `iiac_evidence/PDC-X/P7-00/addendum_2/iiac_audit_session_manifest.PDC-X.p7_00.addendum_2.json` | `certifier_session_id` | `PENDING-CERTIFIER-PDC-P7-00` | `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` |
| same | `evidence_publication_sha` | `0000000000000000000000000000000000000000` | `29d66ae3fcc7a06a919d7e07541740841e8abfe8` |
| same | `certification_publication_sha` | `0000000000000000000000000000000000000000` | `29d66ae3fcc7a06a919d7e07541740841e8abfe8` |

(Path relative to `docs/project/maintainers/qualification/`.)

Choices are the same as in rounds 1 and 2: `evidence_publication_sha` is this round's evidence publication commit; `certification_publication_sha` cannot name the commit that edits it and carries the same SHA; `certifier_session_id` keeps the designation `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13`, which is a reserved designation and not an attested session (no certifier preflight of addendum 2 has been performed here; the certifier is expected to add its own and to adopt this id or replace it by its own pin-only edit). The designation differs from the implementation and auditor session ids of the manifest.

Residue left on purpose: the manifest's `LIM-publication-pending` still describes the placeholders and is now historical; `LIM-pre-activation` stays true (`verify_iiac --require-in-force` was not run here).

## Verifier run

Run from the worktree root, one process at a time (log: `.tmp/session/mcp-rc-evidence-3/verify_iiac_manifests_after_pin.log`, not committed).

```text
## addendum_2 verify_iiac_audit_manifest
verify_iiac_audit_manifest: PASS            exit=0
## addendum_2 verify_iiac_role_separation --strict-product-diff
verify_iiac_role_separation: PASS           exit=0
```

No test suite, mutation harness or activation gate was run; the audit's own results stay as published by the auditor.

# Round 4: addendum 3 (branch `mcp-rc-evidence-4`)

Additive section; rounds 1 to 3 above are unchanged. Same role limits as above: evidence integration only, no verdict, no certification, no rebind, no FRZ change (counter stays 225), no `Approved:` line.

## Commit ladder

| Step | SHA | Content |
|------|-----|---------|
| Base (`origin/mcp-rc`) | `0fc8badd8` | merge of `development` `146e7acf1` (B-P6 finished, development frozen on product paths); tip of `origin/mcp-rc` at integration time, equal to the audited head of addendum 3 |
| Evidence publication commit | `d5c51df0ef2ae63730129e673b426937867c1568` | 44 files imported byte-for-byte, no content change |
| Pin-only edit + this section | the commit that adds this section (its own SHA cannot be stored in it; read it from `git log`) | three pin-only field edits below, this section |

Product delta of the base over the previously audited tree `324d9dba3`: exactly one path under `intergrax/ agents/ applications/`, `intergrax/integrations/providers/embedding_provider/ollama/register.py`.

## Imported evidence

Source: `origin/p7-00-baseline-audit` at `793cce5c8`. It contains merges of `mcp-rc`, so it was not merged. The import is the set of files it adds relative to `0fc8badd8`, the last `mcp-rc` commit it merged (second parent of `a5c1fd514`): 44 paths, all additions, none outside `docs/`, none present on `origin/mcp-rc`, blob identity with the source branch verified for all 44. The set equals the files the branch adds relative to `origin/mcp-rc` itself. It consists of the addendum 3 dossier (`platform_domain_completeness/PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_3.md`, delta `324d9dba3..0fc8badd8`, dossier verdict `PASS_SCOPED`) and the `iiac_evidence/PDC-X/P7-00/addendum_3/` folder (6 top-level files, 9 adversarial scripts, 28 logs).

## Pin-only edits

Only these three field values changed; every other byte of the manifest is untouched. The manifest is not an input of any digest it records; the four digests (dossier, scope file, test matrix, findings file) were re-computed with `sha256sum` before the edit and match.

| File | Field | Old | New |
|------|-------|-----|-----|
| `iiac_evidence/PDC-X/P7-00/addendum_3/iiac_audit_session_manifest.PDC-X.p7_00.addendum_3.json` | `certifier_session_id` | `PENDING-CERTIFIER-PDC-P7-00` | `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` |
| same | `evidence_publication_sha` | `0000000000000000000000000000000000000000` | `d5c51df0ef2ae63730129e673b426937867c1568` |
| same | `certification_publication_sha` | `0000000000000000000000000000000000000000` | `d5c51df0ef2ae63730129e673b426937867c1568` |

(Path relative to `docs/project/maintainers/qualification/`.)

Choices are the same as in rounds 1 to 3: `evidence_publication_sha` is this round's evidence publication commit; `certification_publication_sha` cannot name the commit that edits it and carries the same SHA; `certifier_session_id` keeps the reserved designation `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13`, which is not an attested session (no certifier preflight of addendum 3 has been performed here). The designation differs from the implementation and auditor session ids of the manifest.

Residue left on purpose: the manifest's `LIM-publication-pending` still describes the placeholders and is now historical; `LIM-pre-activation` stays true (`verify_iiac --require-in-force` was not run here). Note that in this manifest `audited_implementation_sha` equals `audit_execution_sha`, so the `--strict-product-diff` check compares an identical pair and is trivially satisfied; the product delta over `324d9dba3` is covered by the auditor's own dossier, not by this verifier.

## Verifier run

Run from the worktree root, one process at a time (log: `.tmp/session/mcp-rc-evidence-4/verify_iiac_manifests_after_pin.log`, not committed).

```text
## addendum_3 verify_iiac_audit_manifest
verify_iiac_audit_manifest: PASS            exit=0
## addendum_3 verify_iiac_role_separation --strict-product-diff
verify_iiac_role_separation: PASS           exit=0
```

No test suite, mutation harness or activation gate was run; the audit's own results stay as published by the auditor.

# Round 5: addendum 4 (branch `mcp-rc-evidence-5`)

Additive section; rounds 1 to 4 above are unchanged. Same role limits as above: evidence integration only, no verdict, no certification, no rebind, no FRZ change (counter stays 225), no `Approved:` line.

## Commit ladder

| Step | SHA | Content |
|------|-----|---------|
| Base (`origin/mcp-rc`) | `d02c39950` | merge of `development` `83d939095` (T-IGX-040-B protocol header, DELEGATION-PDC-P7-ACT); tip of `origin/mcp-rc` at integration time, equal to the audited head of addendum 4 |
| Evidence publication commit | `a1b3053b49c28124a0abde679d733f21a597797f` | 40 files imported byte-for-byte, no content change |
| Pin-only edit + this section | the commit that adds this section (its own SHA cannot be stored in it; read it from `git log`) | three pin-only field edits below, this section |

## Imported evidence

Source: `origin/p7-00-baseline-audit` at `0d4758ca1a185f150c7e0c37ea5dbc1a6eea03cb`. It contains merges of `mcp-rc`, so it was not merged. The import is the set of files it adds relative to `d02c39950`, the last `mcp-rc` commit it merged (second parent of `473133290`): 40 paths, all additions, none outside `docs/`, none present on `origin/mcp-rc`, blob identity with the source branch verified for all 40. It consists of the addendum 4 dossier (`platform_domain_completeness/PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_4.md`, delta `0fc8badd8..d02c39950` covering F-49, F-41 and T-IGX-036-Z, dossier verdict `PASS_SCOPED`) and the `iiac_evidence/PDC-X/P7-00/addendum_4/` folder (6 top-level files, 7 adversarial scripts, 26 logs). The `EVIDENCE_INTEGRATION_RECORD.md` blob on `origin/mcp-rc` equals the one carrying rounds 1 to 4, so this section extends the current text.

## Pin-only edits

Only these three field values changed; every other byte of the manifest is untouched. The manifest is not an input of any digest it records; the four digests (dossier, scope file, test matrix, findings file) were re-computed with `sha256sum` before the edit and match.

| File | Field | Old | New |
|------|-------|-----|-----|
| `iiac_evidence/PDC-X/P7-00/addendum_4/iiac_audit_session_manifest.PDC-X.p7_00.addendum_4.json` | `certifier_session_id` | `PENDING-CERTIFIER-PDC-P7-00` | `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` |
| same | `evidence_publication_sha` | `0000000000000000000000000000000000000000` | `a1b3053b49c28124a0abde679d733f21a597797f` |
| same | `certification_publication_sha` | `0000000000000000000000000000000000000000` | `a1b3053b49c28124a0abde679d733f21a597797f` |

(Path relative to `docs/project/maintainers/qualification/`.)

Choices are the same as in rounds 1 to 4: `evidence_publication_sha` is this round's evidence publication commit; `certification_publication_sha` cannot name the commit that edits it and carries the same SHA; `certifier_session_id` keeps the reserved designation `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13`, which is not an attested session (no certifier preflight of addendum 4 has been performed here). The designation differs from the implementation and auditor session ids of the manifest. Verdicts are untouched: dossier `PASS_SCOPED`, manifest `CONDITIONAL`.

Observations, left unchanged on purpose:

- `test_execution_sha` (`0cb5dd011`) differs from `audited_implementation_sha` and `audit_execution_sha` (`d02c39950`). `git diff --name-only 0cb5dd011 d02c39950 -- intergrax agents applications` is empty, so no product path lies between them.
- In this manifest `audited_implementation_sha` equals `audit_execution_sha`, so `--strict-product-diff` compares an identical pair and is trivially satisfied; the product delta is covered by the auditor's own dossier, not by this verifier.
- `LIM-publication-pending` still describes the placeholders and is now historical; `LIM-pre-activation` stays true (`verify_iiac --require-in-force` was not run here).

## Verifier run

Run from the worktree root, one process at a time (log: `.tmp/session/mcp-rc-evidence-5/verify_iiac_manifests_after_pin.log`, not committed). No pytest was run.

```text
## addendum_4 verify_iiac_audit_manifest
verify_iiac_audit_manifest: PASS            exit=0
## addendum_4 verify_iiac_role_separation --strict-product-diff
verify_iiac_role_separation: PASS           exit=0
```

No test suite, mutation harness or activation gate was run; the audit's own results stay as published by the auditor.

# Round 6: certifier preflight addendum 3 (branch `mcp-rc-evidence-6`)

Additive section; rounds 1 to 5 above are unchanged. Same role limits as above: evidence integration only, no verdict, no certification, no rebind, no FRZ change (counter stays 225), no `Approved:` line.

## Commit ladder

| Step | SHA | Content |
|------|-----|---------|
| Base (`origin/mcp-rc`) | `fbe400f06cd9b77f0fb548ec0d2b2aaeafe10dac` | head of round 5 (addendum 4 evidence and its pin-only edit) |
| Evidence publication commit | `dc1e66fa140d81b28c2658e88b4e1aa16eb29ad2` | 17 files imported byte-for-byte, no content change |
| Record commit | the commit that adds this section (its own SHA cannot be stored in it; read it from `git log`) | this section only |

## Imported evidence

Source: `origin/p7-00-certifier` at `e9d2e2b19e17bd8762d0b30416e017a7d591a4f7` (two commits on `d02c39950`: `3bf093c89` certifier preflight addendum 3, `e9d2e2b19` its round 5 check). The branch is linear and does not contain `fbe400f06`, so it was not merged; the import is the set of files it adds relative to `d02c39950`: 17 paths, all additions, none outside `docs/`, none present on `origin/mcp-rc`, blob identity with the source branch verified for all 17. It consists of the report `platform_domain_completeness/PDC_P7_00_CERTIFIER_PREFLIGHT_ADDENDUM_3.md` and the folder `iiac_evidence/PDC-X/P7-00/certifier/addendum_3/` (result JSON `certifier_preflight_addendum_3_result.json`, `data/`, 9 logs, 5 scripts). Certifier verdict as authored: `CERTIFIER_PREFLIGHT_PASS_SCOPED`; it covers audit addendum 4 (delta `0fc8badd8..d02c39950`) and is not an activation certification, rebind or stage closure.

## Pin-only edits

None. The imported files are certifier records, not audit session manifests; the certifier uses the designation `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` as authored. The strings `PENDING-CERTIFIER-PDC-P7-00` and the 40-zero SHA that occur in `data/addendum4_checks.json` and `logs/round5_checks_at_fbe400f06.log` are the certifier's recorded observations of the auditor's unpinned addendum 4 manifest at the time of its check, not placeholders awaiting a value; rewriting them would falsify the record, so they are left byte-identical. The addendum 4 manifest itself keeps its round 5 pin-only values.

Observation: the certifier's section 9 states that it checked the round 5 pin-only edit directly on git (publication commit 40 of 40 blob-identical, pin-only edit changes exactly the three fields, 12 of 12 digests match, both verifiers PASS) and did not read the round 5 section of this record line by line.

## Verifier run

No new audit manifest is imported in this round, so the two manifest verifiers have no new input. They were run once each on the pinned addendum 4 manifest at the base state as a regression check, one process at a time (log: `.tmp/session/mcp-rc-evidence-6/verify_iiac_manifests.log`, not committed). No pytest was run.

```text
## addendum_4 (pinned, round 5) verify_iiac_audit_manifest
verify_iiac_audit_manifest: PASS            exit=0
## addendum_4 (pinned, round 5) verify_iiac_role_separation --strict-product-diff
verify_iiac_role_separation: PASS           exit=0
```

Not run here: contract suite, mutation harness, `verify_iiac --require-in-force` (the certifier recorded it as failing by design at the base: 7 lines, 6 product paths, one stale protocol governance file).
