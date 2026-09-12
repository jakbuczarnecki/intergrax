# Self-Healing Autonomy Contracts Foundation (R6.1)

**Status:** Implemented — contracts and plugin orchestration only. No execution, lifecycle, or spine integration.

## Cel

R6.1 dostarcza kontraktowy fundament kontrolowanej autonomii: modele, porty SPI i minimalne implementacje domyślne, które klasyfikują postawę autonomii dla rekomendacji R5.3 bez wykonywania strategii.

## Przepływ (zgodny z blueprint R6)

```text
StrategyRecommendation (R5.3)
        |
        v
AutonomyControlRequest + AutonomyDecisionContext
        |
        v
AutonomyControlEngine (plugin orchestration)
        ├── AutonomyPolicy
        ├── AutonomyRiskEvaluator
        └── HumanApprovalRequirementResolver
        |
        v
AutonomyControlDecision
        |
        v
Decision Authority (poza R6.1)
        |
        v
AutonomyExecutionGuard (kontrakt — niewpięty)
        |
        v
Execution Spine (bez zmian)
```

## Modele

| Artefakt | Lokalizacja | Rola |
|----------|-------------|------|
| `AutonomyLevel` | `contracts/.../autonomy/level.py` | Poziomy postawy; `FULL_AUTONOMY` tylko kontrakt — aktywacja runtime odrzucana |
| `AutonomyDecisionContext` | `context.py` | Id rekomendacji, wymagany poziom, refs kontekstu, audyt — bez executor/command |
| `AutonomyControlRequest` | `request.py` | Wiązanie `StrategyRecommendation` z kontekstem |
| `AutonomyControlDecision` | `decision.py` | Wynik klasyfikacji — nie jest poleceniem wykonania |
| `AutonomyAuditBundle` | `audit.py` | Korelacja tenant/investigation/problem |

## Kontrakty (plugin points)

| Port | Odpowiedzialność |
|------|------------------|
| `AutonomyControlEngine` | Orkiestracja oceny — bez wykonania i wyboru strategii |
| `AutonomyPolicy` | Zasady postawy (np. przyszłe `RiskBasedAutonomyPolicy`) |
| `AutonomyRiskEvaluator` | Ocena ryzyka — bez algorytmu scoringu w R6.1 |
| `HumanApprovalRequirementResolver` | Czy wymagany człowiek — bez UI/workflow |
| `AutonomyExecutionGuard` | Przyszła brama przed executorami — tylko `check()` |
| `AutonomyRepository` | Persystencja decyzji — port domenowy |

## Implementacje domyślne (runtime)

| Klasa | Zachowanie |
|-------|------------|
| `DefaultAutonomyPolicy` | `RECOMMEND_ONLY` — bezpieczny domyślny tryb |
| `DefaultAutonomyRiskEvaluator` | Neutralny `UNKNOWN` band |
| `DefaultHumanApprovalRequirementResolver` | Approval przy `APPROVAL_REQUIRED` lub ryzyku HIGH/CRITICAL |
| `PluginAutonomyControlEngine` | DI pluginów, konserwatywne scalanie poziomów |
| `AutonomyControlService` | Opcjonalny `AutonomyRepository` |
| `InMemoryAutonomyRepository` | Adapter testowy / bootstrap |

## Granice authority

- **R5.3** — rekomenduje (doradca).
- **R5.6** — audyt zmian wiedzy (równoległy concern, refs opcjonalne w kontekście).
- **R6.1 engine** — klasyfikuje postawę autonomii.
- **Decision authority** — decyduje o działaniu (poza ten pakiet).
- **Guard + spine** — wykonanie (bez zmian w R6.1).

## Czego jeszcze nie implementujemy

- Wykonywanie strategii i integracja z executorami
- Zmiany workflow, lifecycle, orchestratora
- Risk scoring produkcyjny, approval workflow, UI
- Aktywacja `FULL_AUTONOMY`
- Wpięcie `AutonomyExecutionGuard` do execution spine
- Vendor persistence poza portem `AutonomyRepository`

## Powiązane dokumenty

- [SELF_HEALING_ENTERPRISE_AUTONOMY_CONTROLS_ARCHITECTURE_R6.md](./SELF_HEALING_ENTERPRISE_AUTONOMY_CONTROLS_ARCHITECTURE_R6.md)
- [SELF_HEALING_STRATEGY_RECOMMENDATION_R5.md](./SELF_HEALING_STRATEGY_RECOMMENDATION_R5.md)
- [SELF_HEALING_KNOWLEDGE_GOVERNANCE_R5.md](./SELF_HEALING_KNOWLEDGE_GOVERNANCE_R5.md)

## Testy

`tests/unit/runtime/self_healing/test_autonomy_contracts_r6_1.py` — poziomy, domyślna polityka, wymienność pluginów, brak coupling z execution.
