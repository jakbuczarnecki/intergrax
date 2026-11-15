# MCP-RC evidence integration record (T-IGX-038-F)

Role: evidence integration only. This record is written by a session that is neither the implementer, an auditor nor the certifier; it carries no audit verdict, no certification, no activation, no FRZ change (counter stays 225) and no `Approved:` line.

## Commit ladder on branch `mcp-rc-evidence`

| Step | SHA | Content |
|------|-----|---------|
| Base (`mcp-rc` record head) | `292cc6b29d23debf08542c4c733948e0d12fd16e` | implementation and record; last product change `b0e19bf61` |
| Evidence publication commit | `78d4c496071141ca4092b926b838fb80016f1b91` | 81 files imported byte-for-byte, no content change |
| Pin-only edits + this record | the commit that adds this file (its own SHA cannot be stored in it; read it from `git log`) | six pin-only field edits below, this record |

## Imported evidence (evidence publication commit)

Everything the three branches add under `docs/` relative to their merge base with `mcp-rc` (all additions, no modifications, no deletions, no overlapping paths, no path pre-existing on `mcp-rc`, nothing outside `docs/`). Blob identity with the source branch was verified for all 81 paths.

| Source branch | Head | Merge base with `mcp-rc` | Files | Verdict (as authored there) |
|---------------|------|--------------------------|-------|-----------------------------|
| `mcp-rc-audit` | `4a459ed0be798ca13e292920ba081874244ee58e` | `f3078b293` | 33 | dossier `PARTIAL` at `93a089d3c` |
| `mcp-rc-reaudit` | `efa74c54536567d138b27a2d96dbb15b35f103c2` | `f1dde5a3e` | 31 | `PASS_SCOPED` |
| `mcp-rc-certifier` | `66a782254` | `1b5f26f24` | 17 | `CERTIFIER_PREFLIGHT_PASS_SCOPED` |

## Pin-only edits (second commit)

Only these six field values changed; every other byte of both auditor-authored manifests is untouched. Neither manifest is an input of any digest it records (the four digests cover the audit dossier, the scope file, the test matrix and the findings file, none of which was edited), so no digest changed; all eight recorded digests were re-computed with `sha256sum` before the edit and match.

| File | Field | Old | New |
|------|-------|-----|-----|
| `iiac_evidence/PDC-X/MCP-RC/iiac_audit_session_manifest.PDC-X.mcp_rc.json` | `certifier_session_id` | `PENDING-CERTIFIER-MCP-RC` | `MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13` |
| same | `evidence_publication_sha` | `0000000000000000000000000000000000000000` | `78d4c496071141ca4092b926b838fb80016f1b91` |
| same | `certification_publication_sha` | `0000000000000000000000000000000000000000` | `78d4c496071141ca4092b926b838fb80016f1b91` |
| `iiac_evidence/PDC-X/MCP-RC/reaudit/iiac_audit_session_manifest.PDC-X.mcp_rc_reaudit.json` | `certifier_session_id` | `PENDING-CERTIFIER-MCP-RC` | `MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13` |
| same | `evidence_publication_sha` | `0000000000000000000000000000000000000000` | `78d4c496071141ca4092b926b838fb80016f1b91` |
| same | `certification_publication_sha` | `0000000000000000000000000000000000000000` | `78d4c496071141ca4092b926b838fb80016f1b91` |

(Paths above are relative to `docs/project/maintainers/qualification/`.)

Choices, following the P6 precedent (`MCP_SERVER_X_P6_TERMINAL_DOSSIER.md`, "Pin-only certifier edits", commit `276c64305`, where `evidence_publication_sha` and `certification_publication_sha` both received the phase-1 evidence publication SHA):

- `evidence_publication_sha` is the evidence publication commit above.
- `certification_publication_sha` cannot hold the SHA of the commit that edits it; as in P6 it carries the phase-1 evidence publication SHA. The commit that carries the pins is the one that adds this record.
- `certifier_session_id` is the preflight designation recorded in `certifier/certifier_preflight_result.json`. It differs from both auditor session ids and from the implementer session id (rule N-01 and siblings in `validate_manifest_dict`). It designates the preflight certifier, not the activation certifier.

## Known residue (left unchanged on purpose)

- Each manifest's `LIM-publication-pending` limitation still reads "40 zeros until publication". It is auditor-authored prose, outside the pin-only scope, and is now historical. P6 rewrote the equivalent limitation; this session did not, to keep the edit surface to the six fields.
- `certifier/certifier_preflight_result.json` still states `pin_only_edits: []` and check 15 `NONE_MADE`. That is true of the certifier session; the six edits above were made by this integration session, not by the certifier.
- `MCP_SERVER_X_RC_CERTIFIER_PREFLIGHT.md` line 91 describes the placeholders as still present; it describes the state before this integration.

## Verifier run after the pin-only edits

Run from the worktree root, one process at a time (log: `.tmp/session/mcp-rc-evidence/verify_iiac_manifests_after_pin.log`, not committed). Commands as in the certifier log `certifier/logs/certifier_verify_iiac_manifests.log`.

```text
## audit1 verify_iiac_audit_manifest
verify_iiac_audit_manifest: PASS            exit=0
## audit1 verify_iiac_role_separation --strict-product-diff
verify_iiac_role_separation: PASS           exit=0
## reaudit verify_iiac_audit_manifest
verify_iiac_audit_manifest: PASS            exit=0
## reaudit verify_iiac_role_separation --strict-product-diff
verify_iiac_role_separation: PASS           exit=0
```

Not re-run here: the contract suite, the mutation harness and `verify_iiac --require-in-force` (expected exit 1 by design until the PDC-P7-00 rebind; ACT-N11 trigger). Those results remain the certifier preflight's, at `1b5f26f24`; `1b5f26f24..292cc6b29` changes only four docs files and the draft ticket `governance/tickets/T-IGX-039-A.md`; no product, test or script path.
