---
id: IJ-2026-06-17-004
date: 2026-06-17
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-EVOL-8-DOC
  - APP-EVOL-8.1
  - APP-EVOL-8.2
  - APP-EVOL-8.3
  - P1-ARCH-01
status: completed
commit: eb70ba38
adr: ADR-APP-003
---

# Tier-3 — Hierarchical profile bundles architecture (P1-ARCH-01)

## Operator request

Document the P1-ARCH-01 recommendation: reduce flat `ApplicationEnvironmentProfile` namespace growth via nested capability bundles; add implementation plan tasks and update cross-references.

## Summary

Accepted hierarchical bundle model under the existing `ApplicationEnvironmentProfile` root. Added architecture §22.6, plan register `APP-EVOL-8` (M1–M3), ADR-APP-003, and synced hub, author guides, audit prompt, ACP cross-refs, and debt register.

## Project impact

Tier-3 authors get a canonical grouping model (HostMeta, SecurityEnvelope, CapabilityBundle, CognitionBundle, GovernanceBundle, TopologyBundle, IsolationBundle) before code lands. Implementation can proceed in non-breaking M1 phases with clear acceptance criteria.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §22.6 |
| Plan | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — APP-EVOL-8 |
| ADR | `docs/project/technical/adr/entries/2026-06-17/ADR-APP-003.md` |
| Debt | `docs/project/technical/guides/ARCHITECTURE_DEBT_REGISTER.md` — P1-ARCH-01 |

## Changed artifacts

- `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — §22.6, §41, APP-INV-06
- `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — APP-EVOL-8 register, fidelity matrix, backlog
- `docs/project/technical/adr/entries/2026-06-17/ADR-APP-003.md` — new
- `docs/project/technical/adr/README.md` — index row
- `docs/project/architecture/intergrax_runtime_architecture.md` — hub summary
- `docs/project/technical/guides/APPLICATION_CREATION_GUIDE.md` — §1.1 bundles
- `docs/project/technical/guides/AGENT_CREATION_GUIDE.md` — Appendix H.2 map
- `docs/project/technical/guides/HARNESS_ENVIRONMENT.md` — control plane cross-ref
- `docs/audit_results/TIER3_APPLICATION_ENVIRONMENT.md` — audit vocabulary + gaps
- `docs/project/technical/guides/GOVERNANCE_CONSISTENCY_AUDIT.md` — registry vs profile note
- `docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layer 28 gap
- `docs/project/technical/guides/ARCHITECTURE_DEBT_REGISTER.md` — P1-ARCH-01
- `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — §22.6 cross-refs
- `docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md` — tier boundary note
- `intergrax/applications/USAGE.md` — orchestration profile note

## Verification

```bash
python scripts/maintenance/check_harness_adr.py
python scripts/docs/check_docs_domain_pairs.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass (after journal section fix).

## Risks and follow-ups

- M1 implementation (`APP-EVOL-8.1`–`8.3`) not started — flat wire remains authoritative until shims land.
- M3 (`spec_version` 2.0) is breaking — requires `ProfileMigration` extension and author migration guide.
- Snapshot/diff digest parity must be validated when nested JSON is introduced (`APP-EVOL-8.3`).
