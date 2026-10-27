# DIST-SCALE-X — Dimension L Evidence (P5)

| Field | Value |
| --- | --- |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P5-v1` |

## Executed

| Scenario | Test | PASS semantics |
| --- | --- | --- |
| Admission REJECT when saturated | `test_p5_l_admission_reject_when_saturated` | `DependencyConcurrencyExceededError` on second acquire |

## Reuse

WL-6 queue pressure / WL-4 admission chain — cite P3 @ `e6020efc3` for storm boundedness (dim C/E overlap).
