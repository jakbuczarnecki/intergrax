# INTEGRAX Internal Independent Agent Certification (IIAC) Protocol

**Policy version (document):** `IIAC-2026-11-05.1`  
**Activation:** [`IIAC_POLICY_ACTIVATION_RECORD.json`](./IIAC_POLICY_ACTIVATION_RECORD.json) — **`NOT_IN_FORCE`**. Session B re-audit PASS is preserved evidence. `IN_FORCE` requires the fail-closed gate (`--require-in-force`) and a separate operator activation record.

## Purpose

Replace mandatory **external human** Protocol 03 B6 authority for **pre-production platform stages** (`EVOL-X`, `ENT-AUDIT-X`) with **process-enforced internal independent agent certification**, without retroactive PASS rewrite, without weakening runtime/PROD-Q/legal gates, and without implementer self-certification.

Historical human Protocol 03 records (e.g. EBH-6, DUP-X, ROADMAP-REPLAY-X) remain **category 6 historical evidence**.

## Roles

| Role | Allowed | Forbidden |
|------|---------|-----------|
| **IMPLEMENTER** | Code, tests, candidate evidence, requal after findings | B6 PASS, FRZ promotion, self-CLOSED, fixes during auditor session |
| **INDEPENDENT AUDITOR** | Read-only audited scope, exact-SHA replay (ECAP Mode A/B/C), findings, scoped verdict | Product-path edits between `audited_implementation_sha` and `audit_execution_sha`; accepting own implementation commits |
| **CERTIFICATION / GOVERNANCE AGENT** | Manifest validation, role separation checks, eligibility for ladder/FRZ publication | Replacing audit with checklist-only; overriding fail-closed; certifying validators they authored without Session B review |

Role registry: [`governance/agent-control-plane/role_registry.yaml`](../../../../governance/agent-control-plane/role_registry.yaml).

## Closure vocabulary (canonical)

Platform stage formal closure title:

```text
{STAGE} — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED
```

Use `(SCOPED)` and limitations in manifest and pass-record body — **not** in a false global roadmap headline.

## B6 under IIAC (IIAC-B6)

- ECAP Protocol 03 **semantics preserved**: Mode A/B/C, exact SHA, adversarial negatives.
- **Authority** moves to: independent agent session + [`iiac_audit_session_manifest.v1`](./schemas/iiac_audit_session_manifest.v1.schema.json) + certifier publication gate.
- Pass record filename pattern: `*_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` (parallel to human templates; **no human signature block**).

## FRZ promotion

- No mass FRZ PASS; each row requires manifest-cited evidence.
- **FRZ-AUD-10 (OD-ENT-02):** dual disposition — **LIVE** (live branch protection proof) or **COMPENSATING** (`PASS — COMPENSATING (SCOPED)` + package + Session D acceptance). **OPEN-BP-01** may remain under compensating path.

## Session provenance

### Audit manifest (required fields)

Published JSON conforming to [`schemas/iiac_audit_session_manifest.v1.schema.json`](./schemas/iiac_audit_session_manifest.v1.schema.json):

| Field | Meaning |
|-------|---------|
| `stage_id` | e.g. `IIAC-GOV-1`, `EVOL-X`, `ENT-AUDIT-X` |
| `certification_profile` | `iiac-v1` |
| `policy_version` | Active IIAC document version |
| `implementation_session_id` / `implementation_agent_id` | Session A (or stage implementer) |
| `auditor_session_id` / `auditor_agent_id` | Independent auditor chat (must ≠ implementer) |
| `certifier_session_id` / `certifier_agent_id` | Certification agent (Session B/C/D as applicable) |
| `audited_implementation_sha` | SHA under audit |
| `test_execution_sha` | SHA at test replay |
| `evidence_publication_sha` | SHA when evidence bundle published |
| `audit_execution_sha` | SHA when auditor finished replay |
| `audit_report_digest` | Digest of audit report artifact |
| `audit_scope_manifest_digest` | Digest of scope manifest |
| `test_matrix_digest` | Digest of executed test matrix |
| `findings_digest` | Digest of findings list |
| `verdict` | `PASS`, `PASS_SCOPED`, `CONDITIONAL`, or `FAIL` |
| `limitations` | Array of scoped limitation strings |
| `certification_publication_sha` | SHA of certifier publication commit |

**Assurance level:** default `process_enforced` — handoff record + operator timestamp; no claim of cryptographic session proof without telemetry.

Example (non-PASS): [`examples/iiac_audit_session_manifest.example.json`](./examples/iiac_audit_session_manifest.example.json).

### Handoff record (mandatory per audit)

Template: [`IIAC_SESSION_HANDOFF_RECORD.template.md`](./IIAC_SESSION_HANDOFF_RECORD.template.md).

Operator confirms: new chat, roles from registry, implementer chat ID ≠ auditor chat ID, scoped read/write surfaces.

Ticket handoff may also live under [`governance/tickets/`](../../../../governance/tickets/) with session IDs.

## Implementation → certification handoff (G2)

1. Implementer completes scope @ `GOVERNANCE_IMPLEMENTATION_SHA` (or stage SHA).
2. Pin `audited_implementation_sha`; publish candidate evidence digests.
3. Independent auditor session replays Mode A/B/C @ pinned SHAs.
4. Findings → implementer-only correction loop → optional re-audit.
5. Certifier validates manifest + `verify_iiac_*` + role separation before any PASS publication.

## Policy activation

Status remains **`NOT_IN_FORCE`** until a certification session after the ACT-01 gate correction. Session B `PASS` and the historical `PARTIAL` record are evidence, not activation.

All required before `IN_FORCE`:

1. Governance batch and ACT-01 corrective batch merged at pinned SHAs.
2. Independent Session B verdict **PASS** on `stage_id=IIAC-GOV-1`, with artifact SHA-256 digests. A historical `PARTIAL` / `CONDITIONAL` manifest cannot be the active acceptance record.
3. Operator decision file `IIAC_OD_ACTIVATION_*.md` with decision id, policy version, scope `IIAC policy activation`, `Decision status: AUTHORIZED`, the audit digest, and the exact line `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>`. Session A implementation approval is not this decision.
4. Two-phase SHA publication. Phase 1 commits the `IN_FORCE` record with `effective_from_sha` empty. Phase 2 sets `effective_from_sha` and `policy_activation_sha` to that phase-1 commit. The gate rejects `effective_from_sha = HEAD` and any blob that is still inactive.
5. [`verify_iiac_policy_activation.py`](../../../../scripts/governance/verify_iiac_policy_activation.py) `--require-in-force` exit 0. Default invocation is the same fail-closed check. `--expect-not-in-force` exits 0 only while the policy is inactive and states no active claim.

Evidence used for activation must be published under [`iiac_evidence/`](./iiac_evidence/) (byte-preserving copies). Paths limited to `.tmp/session/` are not durable authority. The gate runs manifest validation and role separation; it does not promote EVOL-X, ENT-AUDIT-X, or ARCH-FREEZE to CLOSED.

## Supersession

- [`IIAC_REQUIREMENT_MIGRATION_MATRIX.md`](./IIAC_REQUIREMENT_MIGRATION_MATRIX.md) — per OD/B6/FRZ mapping.
- [`IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json`](./IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json) — G0 inventory.
- ECAP-P4-EVOL-SCOPE-AUTO-B6: **superseded by IIAC** (not dual activation).

## Validators

| Script | Role |
|--------|------|
| `verify_iiac_audit_manifest.py` | Schema + session inequality + SHA ancestry |
| `verify_iiac_role_separation.py` | Auditor product write-surface |
| `verify_iiac_policy_activation.py` | Fail-closed activation gate (`--require-in-force`; default rejects inactive policy) |
| `verify_od_ent_02_compensating_control.py` | Option B package completeness |

## Related

- [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](./ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md)
- [ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md](./ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md)
- [`INDEPENDENT-REVIEW-HANDOFF.md`](../../technical/devos/INDEPENDENT-REVIEW-HANDOFF.md)
