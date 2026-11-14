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
