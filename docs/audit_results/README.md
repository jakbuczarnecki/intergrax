# Intergrax Platform Audit

`docs/audit_results/` is the **single canonical source of truth** for Intergrax platform audit methodology, campaigns, historical results, finding status, remediation status, and verification traceability.

| Document | Role |
|----------|------|
| [AUDIT_PROTOCOL.md](AUDIT_PROTOCOL.md) | Canonical instruction for conducting an adversarial platform audit |
| [AUDIT_REMEDIATION_PROTOCOL.md](AUDIT_REMEDIATION_PROTOCOL.md) | Canonical instruction for implementing accepted audit findings |

**Not duplicated here:** target system design lives in `docs/project/architecture/`; implementation work lives in `docs/project/maintainers/plans/`. Architecture and plans may reference canonical finding IDs; audit history, campaign status, and remediation status belong under `docs/audit_results/` only.

**Status authority (no duplicates elsewhere):**

| Artifact | Role |
|----------|------|
| **Root `README.md` (this file)** | Global campaign registry — discover latest campaign, latest `COMPLETE` campaign, active campaign |
| **Campaign `README.md`** | Current source of truth for that campaign — metadata, layer register, finding register, rollup, remediation/verification trace |
| **Per-layer report** (`<LAYER_CODE>.md`) | Immutable historical observation at exact `audited_sha` |
| **Architecture** | Target design |
| **Implementation plan** | Implementation work unit |

---

## Directory model

```text
docs/audit_results/
  README.md
  AUDIT_PROTOCOL.md
  AUDIT_REMEDIATION_PROTOCOL.md
  <CAMPAIGN_DIR>/
    README.md              # REQUIRED — campaign master register AND rollup
    <LAYER_CODE>.md        # immutable per-layer snapshot at exact audited_sha
    ...
  legacy/
    README.md
    plan-audit-history/    # migrated plan-satellite audit registers
    YYYY-MM-DD/            # Legacy Audit Protocol v1 campaigns
    ...
```

`<CAMPAIGN_DIR>` = `YYYY-MM-DD` \| `YYYY-MM-DD_run-2` \| `YYYY-MM-DD_run-3` (examples may show `2026-08-18`).

### Campaign semantics

- **Dated directory** = one audit campaign (`<CAMPAIGN_DIR>`). Repeated same-day campaigns use `YYYY-MM-DD_run-2`, `YYYY-MM-DD_run-3`, etc.
- **Campaign `README.md`** = master register: metadata, layer register, finding register (authoritative current lifecycle), cross-layer rollup, remediation/verification trace. **No separate `CAMPAIGN_SUMMARY.md`.**
- **Per-layer files** = immutable audit snapshots bound to an exact `audited_sha`. Completed historical reports are **not rewritten** to advance remediation status or pretend a problem never existed.
- **Remediation / closure evidence** is tracked in the campaign finding register and via [AUDIT_REMEDIATION_PROTOCOL.md](AUDIT_REMEDIATION_PROTOCOL.md).
- **Later periodic audits** create new dated campaigns; they do not silently overwrite prior evidence.

---

## Campaign registry

| Campaign | started_at | completed_at | status | campaign_start_sha | campaign_end_sha | scope | overall_verdict |
|----------|------------|--------------|--------|--------------------|------------------|-------|-----------------|
| — | — | — | — | — | — | — | — |

_No Protocol v2 campaigns persisted yet._

### Registry lifecycle

**On campaign initialization:**

- Create `docs/audit_results/<CAMPAIGN_DIR>/` and campaign `README.md`.
- Immediately add **one** row here with `status` = `IN_PROGRESS`, `campaign_end_sha` = `—`, `completed_at` = `—`, `overall_verdict` = `—`.
- Never append a duplicate registry row for the same `campaign_id`.

**On campaign completion:**

- Update the **same** row: `status` = `COMPLETE`, `completed_at` populated, `campaign_end_sha` populated, `overall_verdict` populated.

**On abort:**

- Update the **same** row to `ABORTED` with completion/abort timestamp; preserve the campaign directory and evidence.

Rows are **newest-first**. This registry is sufficient for discovering the latest campaign, the latest `COMPLETE` campaign, and any active (`IN_PROGRESS`) campaign.

### Global registry (this file)

This root `README.md` is:

- protocol entry point
- global campaign registry
- latest campaign discovery
- legacy pointer

### Legacy vs protocol v2

Results under [legacy/](legacy/README.md) were produced under superseded protocols (Legacy Audit Protocol v1, plan-satellite audit registers). They are useful for historical comparison only — **not** evidence of current platform maturity.

**Protocol v2** (this tree, from 2026-08-18) is model-driven and periodic. No 2026-08-18 v2 campaign is persisted until independently reviewed after this protocol migration.

---

## How to run an audit

1. Read this README.
2. Follow [AUDIT_PROTOCOL.md](AUDIT_PROTOCOL.md) end to end.
3. Persist accepted results under a new `docs/audit_results/<CAMPAIGN_DIR>/` campaign directory and add the registry row here.

## How to remediate findings

1. Read this README.
2. Follow [AUDIT_REMEDIATION_PROTOCOL.md](AUDIT_REMEDIATION_PROTOCOL.md).
3. Update campaign finding register and remediation status in the relevant campaign `README.md`; do not rewrite immutable per-layer audit text.
