# Cross-model behavioral analysis (DS-E2E-15J-L2)

## Responsibility

Compare standardized `ModelQualificationOutcome` rows from multi-model qualification without executing qualification or changing runtime. Produces an auditable `BehavioralComparisonResult`.

## Input

- `CrossModelBehavioralAnalysisRequest`: `scenario_id`, expected `matrix_version`, tuple of `ModelQualificationOutcome`.

## Output

- `BehavioralComparisonResult`: compared model identities, source outcome references, per-area findings, analysis metadata (`analysis_task_id`, `analyzed_at`, `status`).

## Extension

Implement `BehaviorAnalyzer` (see `protocol.py`) and pass instances to `run_cross_model_behavioral_analysis(..., analyzers=(...,))`. Built-in analyzers live in `analyzers.py`; add new analyzer classes there or in a sibling module—do not branch inside `engine.py`.
