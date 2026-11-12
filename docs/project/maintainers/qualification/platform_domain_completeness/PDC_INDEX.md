# PDC-X — campaign index (SSOT hub)

**Stage ID:** `PDC-X` (`PLATFORM-DOMAIN-COMPLETENESS-X`)
**Roadmap status:** `PLANNED / MANDATORY / CURRENT` (roadmap §3, §3.2, §5)
**Position:** `MCP-SERVER-X` → **`PDC-X`** → `ARCH-FREEZE`
**Baseline pin:** `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b`
**ROADMAP_AMENDMENT_SHA:** `bc9ef52bd690738e71638a599eec64c76ac356c9` (PR #461 merge; see [`PDC_SHA_LADDER.md`](./PDC_SHA_LADDER.md))
**Tickets:** P0 [`T-IGX-036-A.md`](../../../../governance/tickets/T-IGX-036-A.md) · P1 [`T-IGX-036-B.md`](../../../../governance/tickets/T-IGX-036-B.md) · P2 [`T-IGX-036-C.md`](../../../../governance/tickets/T-IGX-036-C.md) · P3 [`T-IGX-036-D.md`](../../../../governance/tickets/T-IGX-036-D.md)

PDC-X is **one** campaign with eight certification workstreams (`PDC-D1`..`PDC-D8`). They are **not** separate roadmap stages.

## Authority map (one SSOT per concern; this dossier links and never duplicates)

| Concern | SSOT artifact |
| --- | --- |
| Program sequence, stage status, closure | [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3 / §3.2 / §5 |
| Freeze criteria (225 / 26) | [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](../PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) |
| Assurance operating protocol | [`ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md`](../ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) |
| Independent certification | [`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`](../INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md) + [`IIAC_POLICY_ACTIVATION_RECORD.json`](../IIAC_POLICY_ACTIVATION_RECORD.json); applicability to PDC-X → [`OD-PDC-IIAC-SCOPE`](./operator_decisions/OD-PDC-IIAC-SCOPE.md) |
| ARCH-FREEZE WBS (successor; 54 tasks preserved) | [`ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md`](../ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md) |
| Predecessor campaign | [`mcp_server_x/MCP_SERVER_X_INDEX.md`](../mcp_server_x/MCP_SERVER_X_INDEX.md) |

## Dossier artifacts

| Artifact | Phase | Status |
| --- | --- | --- |
| `PDC_MASTER_PLAN` → [`PDC_MASTER_PLAN.md`](./PDC_MASTER_PLAN.md) | P0 | Published (pointer to the approved plan) |
| `PDC_BASELINE_AND_SCOPE` → [`PDC_BASELINE_AND_SCOPE.md`](./PDC_BASELINE_AND_SCOPE.md) | P0 | Published |
| `PDC_SHA_LADDER` → [`PDC_SHA_LADDER.md`](./PDC_SHA_LADDER.md) | P0 → P9 | Open |
| `PDC_OPERATOR_DECISIONS` → [`PDC_OPERATOR_DECISIONS.md`](./PDC_OPERATOR_DECISIONS.md) | P0 → P9 | Open |
| `PDC_CLOSED_WORLD_INVENTORY` → [`PDC_CLOSED_WORLD_INVENTORY.md`](./PDC_CLOSED_WORLD_INVENTORY.md) + [`.json`](./PDC_CLOSED_WORLD_INVENTORY.json) | P1 | Published (T-IGX-036-B) |
| `PDC_DOMAIN_REQUIREMENTS_MATRIX` → [`.md`](./PDC_DOMAIN_REQUIREMENTS_MATRIX.md) + [`.json`](./PDC_DOMAIN_REQUIREMENTS_MATRIX.json) | P2 | Published (T-IGX-036-C; 168 rows, generated) |
| `PDC_GAP_REGISTER` → [`PDC_GAP_REGISTER.md`](./PDC_GAP_REGISTER.md) | P2 | Published (generated) |
| `PDC_FRZ_COVERAGE_MATRIX` → [`PDC_FRZ_COVERAGE_MATRIX.md`](./PDC_FRZ_COVERAGE_MATRIX.md) | P2 | Published (generated; OD-PDC-FRZ-01 pending) |
| `PDC_OWNERSHIP_MATRIX` → [`PDC_OWNERSHIP_MATRIX.md`](./PDC_OWNERSHIP_MATRIX.md) | P3 | Draft (lock pending OD-PDC-ADR-01) |
| `PDC_ARCHITECTURE_DECISIONS` → [`PDC_ARCHITECTURE_DECISIONS.md`](./PDC_ARCHITECTURE_DECISIONS.md) | P3 | Draft (ADR-PDC-01..08 Proposed) |
| `PDC_IMPLEMENTATION_WBS` | P4 | Not started |
| `PDC_DEPENDENCY_DAG` | P4 | Not started (outline in the master plan §I) |
| `PDC_CROSS_DOMAIN_TEST_MATRIX` | P4 | Not started (outline in the master plan §H) |
| `PDC_DOMAIN_CERTIFICATION_MATRIX` | P7 | Not started |
| `PDC_INDEPENDENT_AUDIT` | P8 | Not started |
| `PDC_FINAL_ACCEPTANCE_REPORT` | P9 | Not started |
| `PDC_ARCH_FREEZE_HANDOFF` | P9 | Not started |

## Workstreams

| ID | Domain |
| --- | --- |
| `PDC-D1` | AGENT-LIFECYCLE |
| `PDC-D2` | CONTEXT-KNOWLEDGE |
| `PDC-D3` | MODEL-RUNTIME |
| `PDC-D4` | HOST-RUNTIME |
| `PDC-D5` | EVALUATION |
| `PDC-D6` | RUNTIME-SECURITY |
| `PDC-D7` | BACKGROUND-SCHED |
| `PDC-D8` | ARTIFACT-LIFECYCLE |
