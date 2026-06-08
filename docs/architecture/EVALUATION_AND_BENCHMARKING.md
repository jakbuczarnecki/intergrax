# Evaluation and Benchmarking

**Status:** Canonical architecture (decomposed from platform canon)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Target reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../IDEAL_HARNESS_AI_ARCHITECTURE.md)

---

# 34. Evaluation Model

Since Intergrax is an experimentation laboratory, every agent should be evaluated.

Evaluation criteria may include:

- task success
- output quality
- factuality
- completeness
- cost
- latency
- usefulness
- repeatability
- user satisfaction
- failure frequency
- business value

Agents should not be considered successful only because they produced text.

**Closed-loop adaptation (L4):** offline/online/shadow/human evaluation feeds the **Adaptive Control Plane** — observe → propose → gate → apply → verify — with bounded envelopes and human-governed policy learning. Evaluation is an input to harness improvement, not only a post-hoc score. Full specification: [`ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md`](ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md) · canon summary §54.

---

