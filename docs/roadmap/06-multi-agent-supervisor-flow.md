# [06] Multi-Agent Supervisor Flow for Business Workstreams

**Status:** Planned  
**Owner:** Artur Czarnecki  
**Start Date:** _(add date)_  
**Target Version:** v1.x  
**Last Updated:** _(add date)_

---

## Goal
Design and implement a **Supervisor-driven multi-agent flow** that decomposes a user’s business request into discrete steps, assigns each step to the most suitable agent (virtual employee), passes intermediate artifacts and context forward, and converges to a final deliverable (decision, report, offer, or email).

The system must support:
- Planning and decomposition of tasks into an ordered or branching workflow.
- Role assignment to specialized agents (e.g., Project Designer, Director, HR, CFO).
- Context handoff between steps (documents, decisions, structured data).
- Conditional routing and approvals.
- Full auditability and reproducibility.

---

## Description
This milestone introduces an orchestration layer (the **Supervisor**) capable of:
1. **Planning**: Transforming a natural-language request into a structured, multi-step plan.
2. **Assignment**: Mapping each step to a specific agent (tool-using LLM component).
3. **Execution**: Running steps in sequence or in parallel, with dependency and condition handling.
4. **Memory & Artifacts**: Persisting outputs as typed artifacts, available to downstream steps.
5. **Governance**: Logging decisions and enforcing permission checks.
6. **Delivery**: Producing a consolidated final output aligned with the user’s intent.

The Supervisor may be implemented as:
- A native intergrax component: `intergraxSupervisor` (graph executor + planner).
- An adapter over a popular framework: **LangGraph** (preferred), CrewAI, or an equivalent state-machine/graph framework.
- A hybrid: intergrax planning + LangGraph execution.

---

## Business Scenario (Example)
**User request**:  
“Based on the attached project documentation, assess whether our company can accept this contract from a technical standpoint (resources, fit with our philosophy/domain/capabilities). Estimate roles and headcount needed, investment required, potential profit. Prepare a cost estimate, select rates, and generate either a client offer or a well-justified refusal.”

**Proposed plan and roles**:
1. **Technical and substantive assessment** — Agent: *Project Designer*  
2. **Policy and strategic fit review** — Agent: *Director/Auditor*  
3. **Staffing and capability check** — Agent: *HR*  
4. **Delivery plan and milestones** — Agent: *Project Manager*  
5. **Costing and internal budget** — Agent: *Accounting*  
6. **Financial approval** — Agent: *CFO*  
7. **Condition: feasible** → Draft the client offer — Agent: *PM/Director*  
8. **Condition: not feasible** → Draft a refusal email — Agent: *PM/Director*

The Supervisor must:
- Parse the request and available documents (RAG integration).
- Build the plan with dependencies and conditions.
- Execute steps, store artifacts, and route based on outcomes.
- Produce a final deliverable (offer or refusal) with traceable rationale.

---

## Key Components

| Component | Role |
|-----------|------|
| **Planner** (`intergraxPlanner`) | Converts user intent + context into a step graph with roles, inputs, outputs, and conditions. |
| **Graph Executor** (`intergraxSupervisor` or LangGraph) | Executes the plan as a DAG/StateGraph with typed edges, retries, and timeouts. |
| **Agent Registry** (`intergraxToolsAgent` + registry) | Resolves agent capabilities; provides tool access (RAG, Web, DB, MCP tools). |
| **Artifact Store** (`intergraxArtifacts`) | Persists typed outputs (JSON, docs, summaries, decisions) for downstream steps. |
| **Memory Layer** (`intergraxConversationalMemory`) | Maintains session context, plan state, and step outcomes. |
| **Policy & Guardrails** (`intergraxGovernance`) | Enforces permissions, PII masking, and approval gates. |
| **Adapters** | Optional execution via LangGraph; MCP bridge for external/internal tools. |

**Artifact types** (examples): `TechAssessment.json`, `PolicyFitReport.json`, `StaffingPlan.json`, `ProjectPlan.md`, `CostEstimate.xlsx|json`, `FinancialDecision.json`, `ClientOffer.md`, `RefusalEmail.md`.

---

## Implementation Plan

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Define the **Plan Schema** (steps, dependencies, inputs/outputs, conditions, roles). | ☐ |
| 2 | Implement **Planner**: prompt-guided decomposition with role assignment and step typing. | ☐ |
| 3 | Implement **Graph Executor**: choose native intergrax executor or LangGraph; support retries, timeouts, and branching. | ☐ |
| 4 | Implement **Artifact Store** and typed payloads; ensure versioning and citations. | ☐ |
| 5 | Integrate **Agents** (PM, HR, Accounting, CFO, Director) with RAG/Web tools via MCP. | ☐ |
| 6 | Add **Governance**: approvals, permissions, redaction, and audit logs. | ☐ |
| 7 | Build **Evaluation Harness**: golden tasks, assertions, and regression tests. | ☐ |
| 8 | Provide **FastAPI endpoints** to submit tasks, stream step events, and fetch artifacts. | ☐ |

---

## Planning & Execution Details

### Plan Schema (draft)
```json
{
  "title": "Bid Readiness and Offer Generation",
  "steps": [
    {
      "id": "tech_assessment",
      "role": "ProjectDesigner",
      "inputs": ["docs.project", "docs.requirements"],
      "outputs": ["artifact.tech_assessment"],
      "success_criteria": ["risks_identified", "feasibility_scored"]
    },
    {
      "id": "policy_fit",
      "role": "Director",
      "inputs": ["artifact.tech_assessment", "docs.company_policy"],
      "outputs": ["artifact.policy_fit"]
    },
    {
      "id": "staffing",
      "role": "HR",
      "inputs": ["artifact.tech_assessment"],
      "outputs": ["artifact.staffing_plan"]
    },
    {
      "id": "delivery_plan",
      "role": "ProjectManager",
      "inputs": ["artifact.tech_assessment", "artifact.staffing_plan"],
      "outputs": ["artifact.delivery_plan"]
    },
    {
      "id": "costing",
      "role": "Accounting",
      "inputs": ["artifact.delivery_plan", "artifact.staffing_plan"],
      "outputs": ["artifact.cost_estimate"]
    },
    {
      "id": "finance_approval",
      "role": "CFO",
      "inputs": ["artifact.cost_estimate", "artifact.policy_fit"],
      "outputs": ["artifact.financial_decision"]
    },
    {
      "id": "offer_or_refusal",
      "role": "ProjectManager",
      "condition": "artifact.financial_decision.approved == true",
      "inputs": ["artifact.delivery_plan", "artifact.cost_estimate"],
      "outputs": ["artifact.client_offer"]
    },
    {
      "id": "refusal_email",
      "role": "ProjectManager",
      "condition": "artifact.financial_decision.approved == false",
      "inputs": ["artifact.policy_fit", "artifact.tech_assessment"],
      "outputs": ["artifact.refusal_email"]
    }
  ]
}
