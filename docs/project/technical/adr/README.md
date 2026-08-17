# Intergrax Harness — Architecture Decision Records

**Domain:** Tier-0 platform + Tier-1 Nexus (`intergrax`, `intergrax/runtime`)

Canonical architecture: [`../../architecture/intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
Implementation tracker: [`../../architecture/intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)

---

## When to write an ADR

Create or update an ADR for **significant** Harness decisions, including:

- Nexus execution semantics, orchestration contracts, lifecycle, delegation
- Tool / skill / integration layer boundaries and catalog contracts
- LLM adapter envelopes, RAG retrieval policy, memory models
- Policy, HITL, observability, and cross-cutting platform behavior
- New universal Tier-0 mechanisms or changes that affect multiple agents

**Not required:** typo fixes, test-only changes, agent-specific business logic (use agent ADRs),
or product-host wiring that does not change platform contracts.

If no ADR is needed, record **"no ADR needed"** with rationale in the PR or plan row.

## Layout

```text
docs/project/technical/adr/
  README.md          ← index (this file)
  TEMPLATE.md        ← copy source
  entries/
    YYYY-MM-DD/
      ADR-{AREA}-{NNN}.md
```

Day folders group ADRs by **creation date** (same convention as the [implementation journal](../../maintainers/implementation-journal/README.md)).

## Naming

```text
ADR-{AREA}-{NNN}.md
```

Examples: `ADR-FLOW-001`, `ADR-LLM-001`, `ADR-ADAPT-001`.

## Process

1. Create `entries/YYYY-MM-DD` if needed (use today's date).
2. Copy [`TEMPLATE.md`](TEMPLATE.md) to `entries/YYYY-MM-DD/ADR-{AREA}-{NNN}.md`.
3. Fill **Context**, **Decision**, **Consequences**, and **Compliance**.
4. Add a row to the **Index** below.
5. Link from canon (`intergrax_runtime_architecture.md`) and/or the relevant `docs/project/maintainers/plans/<DOMAIN>.md`.
6. Set **Status** to `Accepted` when implemented; `Superseded` when replaced.
7. Run `python scripts/maintenance/check_harness_adr.py`.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-UCL-001](entries/2026-08-01/ADR-UCL-001.md) | Unified Context Lifecycle ownership, single-budget authority, versioned context projections | Proposed / Ready for Review |
| [ADR-ADAPT-001](entries/2026-06-05/ADR-ADAPT-001.md) | Adaptive Harness Intelligence over classical RL | Accepted |
| [ADR-ADAPT-002](entries/2026-06-22/ADR-ADAPT-002.md) | ADAS (Agent Design Search) inside AHI — not a separate layer | Accepted |
| [ADR-LLM-001](entries/2026-06-06/ADR-LLM-001.md) | Typed LLM adapter response envelope | Accepted |
| [ADR-LLM-002](entries/2026-06-14/ADR-LLM-002.md) | Central ModelCatalog and context window resolution | Accepted |
| [ADR-LLM-003](entries/2026-06-19/ADR-LLM-003.md) | LLM routing rules — Protocol contract and custom rule classes | Accepted |
| [ADR-FLOW-001](entries/2026-06-07/ADR-FLOW-001.md) | Declarative delegation (`DELEGATES_TO`) expansion | Accepted · implemented |
| [ADR-FLOW-002](entries/2026-06-07/ADR-FLOW-002.md) | Reserved lifecycle states | Accepted |
| [ADR-FLOW-003](entries/2026-06-07/ADR-FLOW-003.md) | `MODIFY_PLAN` decision semantics | Accepted |
| [ADR-CRITIC-001](entries/2026-06-07/ADR-CRITIC-001.md) | Critic & Verification Layer — tier-separated PEV verify stack | Accepted |
| [ADR-MEM-001](entries/2026-06-08/ADR-MEM-001.md) | Context Compiler — global budget allocator and degradation ladder | Accepted |
| [ADR-OBS-001](entries/2026-06-08/ADR-OBS-001.md) | Harness Observability Spine — unified bus for all tiers | Accepted |
| [ADR-SCALE-001](entries/2026-06-08/ADR-SCALE-001.md) | Harness Elastic Capacity Plane — complement K8s HPA | Accepted |
| [ADR-FLOW-004](entries/2026-06-09/ADR-FLOW-004.md) | Graph spec seed guard via `trigger_capabilities` | Accepted |
| [ADR-GR-001](entries/2026-06-09/ADR-GR-001.md) | LLM guardrail integration plane (M-P12) | Accepted |
| [ADR-SCALE-002](entries/2026-06-09/ADR-SCALE-002.md) | Ingress controller vs nginx integration slug (ECP-6.1) | Accepted |
| [ADR-CODECRAFT-001](entries/2026-06-10/ADR-CODECRAFT-001.md) | Ephemeral Code Craft as separate Harness domain | Accepted |
| [ADR-TOOL-001](entries/2026-06-10/ADR-TOOL-001.md) | Catalog tool dispatch and full-gateway routing (TOOL-ENG-1/2) | Accepted |
| [ADR-AGENT-001](entries/2026-06-11/ADR-AGENT-001.md) | Agent cognitive patterns as Tier-2 library — Nexus remains Agent OS | Accepted |
| [ADR-AGENT-002](entries/2026-06-11/ADR-AGENT-002.md) | Author-facing `Agent.run()` facade over UAEP; per-agent environment merge | Accepted |
| [ADR-AGENT-003](entries/2026-06-11/ADR-AGENT-003.md) | Agent step loop (`on_next_step`); dual observability planes | Accepted |
| [ADR-TOOL-002](entries/2026-06-11/ADR-TOOL-002.md) | Bounded multi-iteration tool loop; graph boundary (TOOL-ENG-6) | Accepted |
| [ADR-APP-002](entries/2026-06-12/ADR-APP-002.md) | EnvironmentSnapshot on STRICT task intake | Accepted |
| [ADR-APP-003](entries/2026-06-17/ADR-APP-003.md) | Hierarchical profile bundles on ApplicationEnvironmentProfile | Accepted |
| [ADR-CTX-001](entries/2026-06-12/ADR-CTX-001.md) | Context Engineering as first-class domain and plugin engine | Accepted |
| [ADR-MEM-002](entries/2026-06-14/ADR-MEM-002.md) | Three-domain vector memory catalog (knowledge, LTM, episodic) | Accepted |
| [ADR-FLOW-005](entries/2026-06-12/ADR-FLOW-005.md) | Retire Tier-1 AgentEngine pipeline stack (ACP-CLOSE-LEG-5) | Accepted |
| [ADR-TOOL-003](entries/2026-06-12/ADR-TOOL-003.md) | `ToolInvocationPattern` protocol and orchestration plugin model (TOOL-ENG-16) | Accepted |
| [ADR-TOOL-004](entries/2026-06-12/ADR-TOOL-004.md) | Semantic tool catalog index vs RAG; selection boundary (TOOL-ENG-13) | Accepted |
| [ADR-TOOL-005](entries/2026-06-12/ADR-TOOL-005.md) | Hierarchical selection v1; strategy plugin surfaces (TOOL-ENG-14,26,31) | Accepted |
| [ADR-OBS-002](entries/2026-06-13/ADR-OBS-002.md) | Unsigned Execution Boundary Export (EBE) — partner PoC side channel | Accepted |
| [ADR-OBS-004](entries/2026-06-19/ADR-OBS-004.md) | EBE-9 host-side boundary event signing (Ed25519 statement) | Accepted |
| [ADR-OBS-003](entries/2026-06-17/ADR-OBS-003.md) | Layered runtime event identity — spine + event_kind + EventCatalog | Accepted |
| [ADR-MOD-001](entries/2026-06-19/ADR-MOD-001.md) | Speech provider slug identity via Integration Library (no enum) | Accepted |
| [ADR-SEC-001](entries/2026-06-19/ADR-SEC-001.md) | Security & Trust Planes — S1/S2/S3 discipline and `intergrax.security_defenses` EP | Accepted |
| [ADR-HOST-001](entries/2026-07-13/ADR-HOST-001.md) | Application Hosting as a Dedicated Platform Domain | Accepted |
| [ADR-EXTWORK-001](entries/2026-07-20/ADR-EXTWORK-001.md) | Provider-neutral external work contracts (money + status) | Accepted |
| [ADR-EXTWORK-002](entries/2026-07-20/ADR-EXTWORK-002.md) | Provider-neutral ExternalWorkIntegration boundary | Accepted |
| [ADR-GOVERNED-CONTINUATION-001](entries/2026-07-20/ADR-GOVERNED-CONTINUATION-001.md) | Governed Continuation as Nexus composition | Accepted |
| [ADR-POLICY-SIDE-EFFECT-001](entries/2026-07-20/ADR-POLICY-SIDE-EFFECT-001.md) | Meaningful external side effects require policy before execution | Accepted |
| [ADR-GOVERNED-PROOF-001](entries/2026-07-20/ADR-GOVERNED-PROOF-001.md) | Governed proof profiles describe, but do not own, execution evidence | Accepted |
| [ADR-RUNTIME-POLICY-BUNDLE-001](entries/2026-07-20/ADR-RUNTIME-POLICY-BUNDLE-001.md) | Immutable attested policy bundle identity | Accepted |
| [ADR-EXECUTION-BOUNDARY-EVENT-001](entries/2026-07-20/ADR-EXECUTION-BOUNDARY-EVENT-001.md) | Governed execution boundary event (host-owned) | Accepted |
| [ADR-HOST-ATTESTATION-001](entries/2026-07-20/ADR-HOST-ATTESTATION-001.md) | Host attestor and portable ProofReceipt | Accepted |
| [ADR-MP-001](entries/2026-08-11/ADR-MP-001.md) | Collaborative Work Plane ownership | Accepted (architecture only) |
| [ADR-MP-002](entries/2026-08-11/ADR-MP-002.md) | Principal / Membership / Delegation semantics | Accepted (architecture only) |
| [ADR-AGENT-004](entries/2026-08-12/ADR-AGENT-004.md) | Agent distribution, installation and enablement architecture (AGENT-PLATFORM-1) | Accepted (architecture only) |
| [ADR-PLATFORM-PLUGIN-001](entries/2026-08-14/ADR-PLATFORM-PLUGIN-001.md) | Declarative policy REQUIRE_HITL → canonical Nexus HITL bridge (ENTERPRISE-4-ADR-1) | Accepted / Implemented |
| [ADR-GOVERNED-EXECUTION-001](entries/2026-08-16/ADR-GOVERNED-EXECUTION-001.md) | Governance Evaluation Points and Enforcement Ownership (Governed Execution G1A) | Accepted |
| [ADR-GOVERNED-EXECUTION-002](entries/2026-08-17/ADR-GOVERNED-EXECUTION-002.md) | Policy Catalog Identity, Versioning, and Runtime Ownership (Governed Execution G2A) | Accepted |

**Consolidation:** platform ownership, lifecycle, and invariants for GEC-0…GEC-6 — [`docs/project/technical/platform/governed_external_execution.md`](../platform/governed_external_execution.md).

---

*Scaffold baseline: 2026-06-12*
