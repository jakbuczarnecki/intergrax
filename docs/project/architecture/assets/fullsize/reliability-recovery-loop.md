# Reliability Recovery Loop

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="../reliability-recovery-loop-dark.svg"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="../reliability-recovery-loop-light.svg"
  >
  <img
    src="../reliability-recovery-loop-light.svg"
    alt="Conceptual recovery diagram: FAILURE flows to classify failure, then ResiliencePolicy, then retry, degrade, or HITL. Retry maps to R0–R3; degrade to partial result; HITL to interrupt, human decision, resume or stop. Recovery evidence flows to RuntimeEvent and HOS."
  >
</picture>

[Open light image](../reliability-recovery-loop-light.svg) ·
[Open dark image](../reliability-recovery-loop-dark.svg)
