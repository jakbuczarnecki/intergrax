# Model capability baseline (DS-E2E-15J-L3)

## Responsibility

Build durable, factual **model capability profiles** from existing `ModelQualificationOutcome` rows and optional `BehavioralComparisonResult`. No routing, selection, or runtime changes.

## Data flow

```text
ModelQualificationOutcome (+ optional BehavioralComparisonResult)
        → CapabilityProfileBuildRequest
        → CapabilityProfileEngine (build_model_capability_profiles)
        → CapabilityExtractor plugins (per dimension)
        → ModelCapabilityProfile (per model, evidence refs only)
```

Profiles store **observations** (strong / weak / neutral levels with factual descriptors), not business rankings.

## Extension

Implement `CapabilityExtractor` with a new `dimension_id` and register it in the tuple passed to `build_model_capability_profiles`. Do not edit the engine.

Default extractors live in `extractors.py`.
