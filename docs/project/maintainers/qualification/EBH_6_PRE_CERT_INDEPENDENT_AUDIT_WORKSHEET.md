# EBH-6 — Pre-cert Independent Audit Worksheet (E01)

| Field | Value |
| --- | --- |
| **Workstream** | E01 — PRE-CERT EVIDENCE ACCEPTANCE |
| **Baseline SHA** | `202239f726dd5e3fe23bf1a68cafa61c1eb9fca7` |
| **Mechanical SSOT** | [`_t2_evidence_bundles.py`](../../../tests/qualification/ebh_6/_t2_evidence_bundles.py) · [`_tier_audit.py`](../../../tests/qualification/ebh_6/_tier_audit.py) |
| **Implementer disposition** | T1=0 · T2=131 · T3=2 · 0× implementer INSUFFICIENT |
| **Independent pre-cert verdict** | **E01-PRE-01 — PASS (SCOPED) — 2026-10-29** — [`EBH_6_E01_PRE_01_INDEPENDENT_ACCEPTANCE_RECORD.md`](EBH_6_E01_PRE_01_INDEPENDENT_ACCEPTANCE_RECORD.md) @ `f27bfaf2f…` |

**Rule:** **131 T2 rows are audit candidates**, not automatic PASS. Each bundle requires replay of behavioral gates + scope + negatives.

## T3 controls (mandatory)

| Pair | Implementer outcome | Gate anchor | Independent disposition | Auditor notes |
| --- | --- | --- | --- | --- |
| `execution` × B | PASS_CANDIDATE | EE-A1 P0 inventory | **PENDING** | |
| `semantic_core` × F | FINDING | Q2-D2 register + class F gate | **PENDING** | Blocked on **OD-EBH6-Q2D2** — see [EBH_6_Q2D2_DECISION_DOSSIER.md](EBH_6_Q2D2_DECISION_DOSSIER.md) |

## Bundle replay (19 units)

| Task | Bundle ID | Invariant (summary) | Primary gates (replay @ SHA) | Independent disposition | FAIL → task |
| --- | --- | --- | --- | --- | --- |
| E01-BND-01 | `integration_factory` | FRZ integration factory / reachability | EBH-5 factory + reachability nodeids in SSOT | **PENDING** | E04-BND |
| E01-BND-02 | `sanctioned_composition` | Sanctioned composition roots | `test_ebh6_exit_sanctioned_composition_roots_non_empty` | **PENDING** | E04-BND |
| E01-BND-03 | `execution_bypass` | Execution bypass | EE-A1 | **PENDING** | E04-BND |
| E01-BND-04 | `nexus_reachability` | Nexus L3 reachability | Sanctioned + L3 primary gates | **PENDING** | E04-BND |
| E01-BND-05 | `governance_boundary` | Governance ≠ execution | NPSC42 H1 | **PENDING** | E04-BND |
| E01-BND-06 | `persistence_sqlite` | SQLite annex | `test_ebh6_exit_sqlite_annex_accountability` | **PENDING** | E04-BND |
| E01-BND-07 | `observability_authority` | Bounded observability sink | W5-B2 invoke gate | **PENDING** | E04-BND |
| E01-BND-08 | `recovery_plane` | Recovery admission | NPSC5e | **PENDING** | E04-BND |
| E01-BND-09 | `agent_lifecycle` | Lifecycle bypass | Lifecycle architecture gate | **PENDING** | E04-BND |
| E01-BND-10 | `rag_bootstrap` | RAG graph bootstrap | EBH-5 RAG reachability | **PENDING** | E04-BND |
| E01-BND-11 | `config_hosting` | CONFIG-X structural | CONFIG-X 54/54 + EBH-5 inventory | **PENDING** | E04-BND |
| E01-BND-12 | `contracts_typing` | FRZ-TYP contracts + app seams | CSP + `_shared` seam gate + APP-TYP registry | **PENDING** | E04-BND |
| E01-BND-13 | `platform_vendor_scan` | Vendor scan A/D | Prohibited/unclassified gates | **PENDING** | E04-BND |
| E01-BND-14 | `replaceability_j` | FRZ-RPL class J | RPL-G invoke + VBN matrix | **PENDING** | E04-BND |
| E01-BND-15 | `q2d2_finding` | Q2-D2 FINDING track | Class F semantic_core gate | **PENDING** | E03/E04 |
| E01-BND-16 | `platform_fail_closed` | Fail-closed | Class gates | **PENDING** | E04-BND |
| E01-BND-17 | `platform_hardcoding` | Hardcoding register | Register integrity | **PENDING** | E04-BND |
| E01-BND-18 | `platform_typing_scan` | Typing scan | Class F non-core gates | **PENDING** | E04-BND |
| E01-BND-19 | `l3_reachability` | L3 primary_gate | L3 reachability annex | **PENDING** | E04-BND |

## 133× pair sampling (E01-PAR-01)

Auditor confirms per pair: layer owner · L3 mechanism link · test scope · positive/negative coverage · **ACCEPTED / INSUFFICIENT / N/A / FINDING**.

| Metric | Implementer | Independent |
| --- | ---: | --- |
| Pairs reviewed | 133 (mechanical map complete) | **0 / 133** |
| INSUFFICIENT @ independent review | 0 (implementer) | **PENDING** |

## Acceptance (E01-PRE-01)

**DONE when:** all mandatory pairs independently dispositioned; zero undocumented INSUFFICIENT; signed pre-cert record in [EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md](EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md).

**Agent forbidden:** issuing pre-cert PASS or Protocol 03 PASS.
