# Intergrax Platform Audit

docs/audit_results/ is the **single canonical source of truth** for Intergrax platform audit methodology, campaigns, historical results, and remediation status.

| Document | Role |
|----------|------|
| [AUDIT_PROTOCOL.md](AUDIT_PROTOCOL.md) | Canonical instruction for conducting an adversarial platform audit |
| [AUDIT_REMEDIATION_PROTOCOL.md](AUDIT_REMEDIATION_PROTOCOL.md) | Canonical instruction for implementing accepted audit findings |

**Not duplicated here:** target system design lives in docs/project/architecture/; implementation work lives in docs/project/maintainers/plans/. Architecture and plans may reference canonical audit finding IDs; audit history and remediation status belong under docs/audit_results/ only.

---

## Directory model

`	ext
docs/audit_results/
  README.md
  AUDIT_PROTOCOL.md
  AUDIT_REMEDIATION_PROTOCOL.md
  YYYY-MM-DD/
    README.md              # campaign master register
    <LAYER>.md             # immutable per-layer snapshot at exact commit
    ...
  legacy/
    README.md
    YYYY-MM-DD/            # Legacy Audit Protocol v1 campaigns
    ...
`

### Campaign semantics

- **Dated directory** = one audit campaign (YYYY-MM-DD). Repeated same-day campaigns use deterministic suffixes (YYYY-MM-DD_run-2) when needed.
- **Campaign README** = master register: scope, commit SHA, status (IN_PROGRESS / COMPLETE / ABORTED), finding index, remediation rollup.
- **Per-layer files** = immutable audit snapshots bound to an exact repository commit. Completed historical reports are **not rewritten** to pretend a problem never existed.
- **Remediation / closure evidence** is tracked in the campaign register and via [AUDIT_REMEDIATION_PROTOCOL.md](AUDIT_REMEDIATION_PROTOCOL.md).
- **Later periodic audits** create new dated campaigns; they do not silently overwrite prior evidence.

### Legacy vs protocol v2

Results under [legacy/](legacy/README.md) were produced under the superseded **Legacy Audit Protocol v1** (progress.json, orchestrators, generated prompts). They are useful for historical comparison only — **not** evidence of current platform maturity.

**Protocol v2** (this tree, from 2026-08-18) is model-driven and periodic. No 2026-08-18 v2 campaign is persisted until independently reviewed after this protocol migration.

---

## How to run an audit

1. Read this README.
2. Follow [AUDIT_PROTOCOL.md](AUDIT_PROTOCOL.md) end to end.
3. Persist accepted results under a new YYYY-MM-DD/ campaign directory.

## How to remediate findings

1. Read this README.
2. Follow [AUDIT_REMEDIATION_PROTOCOL.md](AUDIT_REMEDIATION_PROTOCOL.md).
3. Update campaign remediation status in the relevant campaign README; do not rewrite immutable per-layer audit text.
