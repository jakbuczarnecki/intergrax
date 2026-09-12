# Autonomous Enterprise Self-Healing Strategy (R1)

**Task:** `AUTONOMOUS-ENTERPRISE-SELF-HEALING-STRATEGY-R1`

**Status:** Architecture frozen

**Invariant:** `DECISION_IS_NOT_EXECUTION` — strategies evaluate; Safety + Governance authorize; External Operation spine executes.

---

## 1. Authority model

| Layer | Role |
| ----- | ---- |
| Central Diagnostic Engine | Evidence and investigation context — **no healing execution** |
| `SelfHealingStrategy` | Pure decision — **evaluate only** |
| `SelfHealingDecision` | Proposed actions + confidence — **not authorization** |
| `SelfHealingSafetyEvaluator` | Blast radius, rollback, approval hints |
| `SelfHealingAdmissionGate` | ALLOW / DENY / REQUIRES_APPROVAL |
| `ExternalOperationAdmission` | Same verdicts on translated intent |
| Execution Runtime | Single spine via `ExternalOperationExecutionGate` |

Platform defaults (`intergrax.runtime.self_healing.defaults`) are plugins — **default ≠ authority**.

---

## 2. Contracts (`intergrax/contracts/self_healing/`)

- `SelfHealingStrategy` — `evaluate(context) -> SelfHealingDecision | None`; no `execute()`
- `SelfHealingContext` — readonly diagnostic + predictive + operation catalog; no I/O
- `SelfHealingDecision` — `proposed_actions`, `evidence_refs`, `required_approval`
- `SelfHealingActionProvider` — translate decision action → `ExternalOperationIntent`
- `SelfHealingStrategyRegistry` — register / resolve / list_available port
- `SelfHealingStrategyQualityProfile` — outcome loop metrics
- `SelfHealingAuditRecord` — operator reconstructability

Read attachment: `intergrax/contracts/self_healing_investigation_read.py` → `DiagnosticInvestigationView.self_healing_history`.

---

## 3. Runtime (`intergrax/runtime/self_healing/`)

- `InMemorySelfHealingStrategyRegistry` — discovery, versioning, tenant scope
- `resolve_strategies_for_context` — tenant → capability → evidence → priority / specificity
- `SelfHealingDecisionEngine` — contained evaluation; `STRATEGY_FAILED` isolation
- `SelfHealingSafetyEvaluator` — pre-governance constraints
- `resolve_self_healing_governance_chain` — Decision → Safety → Governance → external admission
- `GovernedSelfHealingOrchestrator` — translate + audit; reuses external gate
- `SelfHealingOutcomeEngine` — strategy quality updates
- `project_self_healing_history` — diagnostic read projection

---

## 4. Plugin lifecycle

Registration via `registry.register(strategy)`. Application and vendor plugins implement the same SPI. Plugin failure yields `STRATEGY_FAILED` and resolution continues — diagnostics and execution spine are not blocked.

---

## 5. Execution reuse

Forbidden: per-domain healers (`crm_healer.execute()`), local retry engines, synthetic Problem stores, plugin diagnostic engines.

Required: `ExternalOperationIntent`, admission gates, `ExternalOperationExecutionGate`, platform execution identity, central diagnostic reconstruction on failure.

---

## 6. Operator read model

Trail: Risk → Strategy selected → Decision → Approval → Execution → Outcome on `self_healing_history`.

---

## 7. Architectural prohibitions

- `strategy.execute()` or decision self-execution
- Confidence as authorization
- Plugin bypass of governance
- Hardcoded business `if kafka_down` scenarios in platform tier
- Second execution or Problem store inside plugins
