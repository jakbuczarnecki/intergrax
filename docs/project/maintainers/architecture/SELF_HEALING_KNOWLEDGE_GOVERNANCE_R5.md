# Self-Healing Knowledge Governance and Audit Evolution (R5.6)

**Status:** Foundation (R5.6) — audit and control metadata for derived strategy knowledge; **no execution authority**.

## Why governance is needed

R5.4–R5.5 evolve **derived** strategy knowledge from experiences and context. Before R6 controlled autonomy, every transition `Knowledge Version N → N+1` must be **visible and replayable**. Governance answers:

> *Why does current strategy knowledge look like this?*

Without auto-acceptance, auto-correction, rollback of strategies, or workflow changes.

## Boundary vs R6

| In scope (R5.6) | Out of scope (R6+) |
|-----------------|-------------------|
| Change records, audit repository, integrity checks | Autonomous approval of knowledge |
| Plugin governance policies (descriptive controls) | Automatic knowledge correction |
| `StrategyKnowledgeUpdated` event shape | Execution or lifecycle decisions |
| Revision metadata (`evolution_mechanism_id`, `recorded_at`) | ML governance / AI judge |

```text
Knowledge Governance
        |
        v
Audit / Validation

NOT:

Knowledge Governance → Execution
```

## Audit model

| Component | Role |
|-----------|------|
| `StrategyKnowledgeRepository` | Current knowledge state (R5.4) |
| `StrategyKnowledgeAuditRepository` | Append-only history of **changes** and **update events** |
| `StrategyKnowledgeChangeRecord` | Single transition N → N+1 (what, why, when, mechanism, experience refs) |
| `StrategyKnowledgeRevision` | Extended audit bundle (mechanism + timestamp + version chain) |
| `StrategyKnowledgeGovernanceService` | Records change + policy assessment + optional integrity report |

Persistence chain (no vendors in domain):

```text
Audit domain contracts
        → StrategyKnowledgeAuditRepository (port)
        → configured vendor adapter (future)
        → storage provider
```

## Versioning

Each successful evolution produces:

1. `StrategyKnowledgeRevision` appended to `StrategyKnowledgeRepository`
2. Optional `StrategyKnowledgeGovernanceService.record_knowledge_evolution` (wired via `StrategyKnowledgeEvolutionService.knowledge_governance`)
3. `StrategyKnowledgeChangeRecord` + `StrategyKnowledgeUpdated` in audit store

Version chain integrity is validated by `StrategyKnowledgeIntegrityValidator` (default: `BasicStrategyKnowledgeIntegrityValidator`).

## Plugin points

```text
StrategyKnowledgeGovernanceService
        ├── StrategyKnowledgeAuditRepository
        ├── StrategyKnowledgeGovernancePolicy  → DefaultKnowledgeGovernancePolicy
        └── StrategyKnowledgeIntegrityValidator (optional) → BasicStrategyKnowledgeIntegrityValidator
```

Future policy examples (contracts only today): Strict, EnterpriseApproval, RiskBased — all via `StrategyKnowledgeGovernancePolicy` without domain vendor imports.

## Integration

`StrategyKnowledgeEvolutionService` accepts optional `knowledge_governance`. When set, after `append_revision` the service records audit metadata only — it does **not** invoke orchestrator, lifecycle, or selectors.

## Related docs

- [SELF_HEALING_STRATEGY_KNOWLEDGE_EVOLUTION_R5.md](./SELF_HEALING_STRATEGY_KNOWLEDGE_EVOLUTION_R5.md) (R5.4)
- [SELF_HEALING_CONTEXTUAL_KNOWLEDGE_OPTIMIZATION_R5.md](./SELF_HEALING_CONTEXTUAL_KNOWLEDGE_OPTIMIZATION_R5.md) (R5.5)
