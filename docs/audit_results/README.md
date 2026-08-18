# Intergrax Platform Audit

`docs/audit_results/` is the **single canonical source of truth** for Intergrax platform audit methodology, campaigns, historical results, finding status, remediation status, and verification traceability.

| Document | Role |
|----------|------|
| [AUDIT_PROTOCOL.md](AUDIT_PROTOCOL.md) | Canonical instruction for conducting an adversarial platform audit |
| [AUDIT_REMEDIATION_PROTOCOL.md](AUDIT_REMEDIATION_PROTOCOL.md) | Canonical instruction for implementing accepted audit findings |

**Not duplicated here:** target system design lives in `docs/project/architecture/`; implementation work lives in `docs/project/maintainers/plans/`. Architecture and plans may reference canonical finding IDs; audit history, campaign status, and remediation status belong under `docs/audit_results/` only.

**Campaign status tracks the audit lifecycle, not the remediation lifecycle.**

- `COMPLETE` = scoped audit completed and audit baseline frozen (`completed_at`, `campaign_end_sha`, `overall_verdict`, per-layer evidence).
- Finding statuses (`ACCEPTED` → `IMPLEMENTING` → `IMPLEMENTED` → `VERIFIED` → `CLOSED`) track subsequent remediation in the campaign `README.md` finding register and remediation rollup.
- The root registry row remains unchanged while remediation proceeds. Do **not** add a second campaign status system or remediation columns to this registry.

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
- **Campaign `README.md`** = master register: metadata, layer register, finding register (authoritative current lifecycle), **audit rollup** (frozen at audit `COMPLETE`), **remediation rollup** (mutable post-audit), remediation/verification trace. **No separate `CAMPAIGN_SUMMARY.md`.**
- **Per-layer files** = immutable audit snapshots bound to an exact `audited_sha`. Completed historical reports are **not rewritten** to advance remediation status or pretend a problem never existed.
- **Remediation / verification evidence** is tracked in the campaign finding register, remediation rollup, and via [AUDIT_REMEDIATION_PROTOCOL.md](AUDIT_REMEDIATION_PROTOCOL.md). Remediation normally begins against a `COMPLETE` campaign; remediation completion does **not** change campaign status or frozen audit baseline fields.
- **Later periodic audits** create new dated campaigns; they do not silently overwrite prior evidence. Closing all findings does **not** retroactively change a prior campaign's `overall_verdict` — a fresh audit determines current platform verdict independently.

---

## Campaign registry

| Campaign | started_at | completed_at | status | campaign_start_sha | campaign_end_sha | scope | overall_verdict |
|----------|------------|--------------|--------|--------------------|------------------|-------|-----------------|
| [2026-08-18](2026-08-18/README.md) | 2026-08-18 | — | IN_PROGRESS | `9658224495c775fcefd55ab52bbcc7a94c84fb50` | — | Platform audit — 2 layers complete (`STRATEGIC_HARNESS_MODEL`, `TIER_LAYER_BOUNDARIES`) | — |

### Registry lifecycle

**On campaign initialization:**

- Create `docs/audit_results/<CAMPAIGN_DIR>/` and campaign `README.md`.
- Immediately add **one** row here with `status` = `IN_PROGRESS`, `campaign_end_sha` = `—`, `completed_at` = `—`, `overall_verdict` = `—`.
- Never append a duplicate registry row for the same `campaign_id`.

**On audit campaign completion:**

- Update the **same** row: `status` = `COMPLETE`, `completed_at` populated, `campaign_end_sha` populated, `overall_verdict` populated. These fields are frozen at audit closeout and MUST NOT be rewritten during remediation.

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

**Protocol v2.1** (this tree, from 2026-08-18) is model-driven and periodic. Active campaign: [2026-08-18](2026-08-18/README.md) — two layers complete: `STRATEGIC_HARNESS_MODEL` **FAIL** (10 ACCEPTED findings) and `TIER_LAYER_BOUNDARIES` **FAIL** (5 ACCEPTED findings); campaign `IN_PROGRESS`.

### Audit scope shapes (Protocol v2.1)

Protocol v2.1 supports three audit shapes:

1. **DOMAIN / LAYER AUDIT**
2. **CONCEPTUAL / CROSS-DOMAIN AUDIT**
3. **PLATFORM CONSUMER AUDIT**

Platform consumer audits verify whether applications, agents, plugins, and integration adapters correctly reuse canonical Intergrax mechanisms, respect layer ownership, preserve platform guarantees, and avoid duplicate/bypass infrastructure. See [AUDIT_PROTOCOL.md](AUDIT_PROTOCOL.md) section D3 for consumer audit scope, conformance matrix, and falsification questions.

---

## How to run an audit

1. Read this README.
2. Follow [AUDIT_PROTOCOL.md](AUDIT_PROTOCOL.md) end to end.
3. Persist accepted results under a new `docs/audit_results/<CAMPAIGN_DIR>/` campaign directory and add the registry row here.

## How to remediate findings

1. Read this README.
2. Follow [AUDIT_REMEDIATION_PROTOCOL.md](AUDIT_REMEDIATION_PROTOCOL.md).
3. Update campaign finding register and remediation status in the relevant campaign `README.md`; do not rewrite immutable per-layer audit text.
