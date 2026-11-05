# ASF/SEF — Semantic ownership & reuse matrix

**Status:** Published architecture satellite (P1-B1) · **SSOT tables:** Master Plan Deliverable 5  
**Binding plan:** [`governance/tickets/plans/ASF-SEF-MASTER-PLAN.md`](../../../../governance/tickets/plans/ASF-SEF-MASTER-PLAN.md)

| Concern | Canonical owner | Engineering plane may | Scenario plane may | Duplicate risk |
|---------|-----------------|----------------------|--------------------|----------------|
| Policy/authorization | PolicyEngine + Governance | read-only policy for eng sandbox tool allowlists | full runtime path | DevOS rules ≠ PolicyEngine |
| Execution root | HostTaskExecutionPort → Nexus | **must not** root Nexus for IDE/repo edits | **yes** admitted scenarios | Cursor shell as SSOT |
| Scenario lifecycle | Tier-3 manifest + graph spec | produce artifacts | run admitted | local bypass |
| Tools/side effects | ToolRuntime | sandbox-limited tools only | production tool profiles | direct subprocess |
| Agent lifecycle | Agent Distribution | package agents in repo | activate via distribution | AgentRegistry write |
| Qualification | ECAP + FRZ checklist | candidate evidence | scoped qual records | implementer B6 |
| Change classification | POST-FREEZE governance + EVOL | assess before merge | N/A post-build | ad-hoc labels |
| Roadmap SSOT | PLATFORM_ENTERPRISE_COMPLETION_ROADMAP | no new stage without OD | SCENARIO-GATE | SEF-X via OD-SEF-02 only |
| Code craft / ephemeral dev | CodeCraft domain (ADR-CODECRAFT-001) | eng sandbox patterns | not production authority | second execution root |
| DevOS / delivery | Ticket + plan approval gates | orchestrate engineering jobs | N/A | conflate with Nexus |
| Nexus / UAEP | Tier-1 runtime | invoke only via proof hosts for REF demos | primary scenario loop | engineering loop SSOT |

**Dependency direction:** Engineering → (artifacts, manifests, tests, evidence) → Tier-3 composition → Scenario runtime. **Never** Engineering → live ToolRuntime production profiles without PROD-Q.
