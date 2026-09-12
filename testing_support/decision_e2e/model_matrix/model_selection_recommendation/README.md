# Model selection recommendation (DS-E2E-15J-L4)

## Responsibility

Suggest the most suitable model for a task using existing `ModelCapabilityProfile` data. **No model execution**, routing, or Execution Engine integration.

## Data flow

```text
TaskRequirements + CapabilitySelectionConstraints + ModelCapabilityProfile[]
        → ModelSelectionRequest
        → ModelSelectionEngine (injected SelectionStrategy plugins)
        → ModelSelectionRecommendation (evidence + audit metadata)
```

The engine stops at the recommendation. A human or upstream policy may accept it.

## Extension

Implement `SelectionStrategy` with a new `strategy_id` and pass it in `ModelSelectionEngine(strategies=(..., YourStrategy()))`. Do not edit the engine.

Default strategies live in `strategies.py` (`default_selection_strategies()`).
