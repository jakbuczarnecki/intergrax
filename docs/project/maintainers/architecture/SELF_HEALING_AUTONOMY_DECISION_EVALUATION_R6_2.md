# Self-Healing Autonomy Decision Evaluation (R6.2)

**Status:** Implemented — explainable evaluation only. No execution, lifecycle, or guard wiring.

## Cel

R6.2 odpowiada na pytanie: **„Jak oceniliśmy możliwość działania?”** — nie **„Wykonaj działanie.”** Warstwa scala wyniki polityki, ryzyka i wymogu akceptacji człowieka w audytowalny, wymienny wynik oceny z uzasadnieniem.

## Przepływ

```text
AutonomyControlRequest (R6.1)
        |
        v
AutonomyDecisionEvaluationService
        |
        v
AutonomyDecisionEvaluator (plugin)
        ├── AutonomyPolicyEvaluator  → AutonomyPolicyEvaluationResult
        ├── AutonomyRiskEvaluator    → AutonomyRiskEvaluationResult (R6.1 assessment)
        └── HumanApprovalEvaluator   → HumanApprovalEvaluationResult
        |
        v
AutonomyEvaluationResult + AutonomyDecisionExplanation
        |
        v
Decision authority / R6.3+ (poza R6.2)
```

## Modele

| Artefakt | Lokalizacja | Rola |
|----------|-------------|------|
| `AutonomyEvaluationResult` | `evaluation_result.py` | Id oceny, werdykt, poziom, confidence, powody, źródła, pod-wyniki |
| `AutonomyDecisionExplanation` | `explanation.py` | because / reguły / aktywne ograniczenia |
| `AutonomyPolicyEvaluationResult` | `policy_evaluation.py` | PASS/FAIL polityki oddzielone od werdyktu końcowego |
| `AutonomyRiskEvaluationResult` | `risk.py` | Band, confidence, explanation — na bazie R6.1 assessment |
| `HumanApprovalEvaluationResult` | `approval_evaluation.py` | Czy człowiek, dlaczego — bez workflow |

## Plugin points

| Port | Implementacja domyślna |
|------|-------------------------|
| `AutonomyDecisionEvaluator` | `PluginAutonomyDecisionEvaluator` |
| `AutonomyPolicyEvaluator` | `AutonomyPolicyPluginEvaluator` (wrap `AutonomyPolicy`) |
| `AutonomyRiskEvaluator` | `DefaultAutonomyRiskEvaluator` (R6.1) |
| `HumanApprovalEvaluator` | `HumanApprovalPluginEvaluator` (wrap resolver R6.1) |
| `AutonomyDecisionRepository` | `InMemoryAutonomyDecisionRepository` |
| `AutonomyEvaluationAuditRecorder` | kontrakt — storage w host |

Przyszłe pluginy (poza scope): `PolicyBasedDecisionEvaluator`, `RiskAwareDecisionEvaluator`, `EnterpriseDecisionEvaluator`.

## Granice względem execution

- Brak executorów, commandów i akcji workflow w modelach oceny.
- `AutonomyDecisionEvaluationService` nie wywołuje `AutonomyExecutionGuard` ani execution spine.
- `AutonomyControlEngine` (R6.1) pozostaje osobną ścieżką klasyfikacji postawy; R6.2 nie zastępuje authority wykonania.

## Testy

`tests/unit/runtime/self_healing/test_autonomy_decision_evaluation_r6_2.py`

## Powiązane

- [SELF_HEALING_AUTONOMY_CONTRACTS_FOUNDATION_R6_1.md](./SELF_HEALING_AUTONOMY_CONTRACTS_FOUNDATION_R6_1.md)
- [SELF_HEALING_ENTERPRISE_AUTONOMY_CONTROLS_ARCHITECTURE_R6.md](./SELF_HEALING_ENTERPRISE_AUTONOMY_CONTROLS_ARCHITECTURE_R6.md)
