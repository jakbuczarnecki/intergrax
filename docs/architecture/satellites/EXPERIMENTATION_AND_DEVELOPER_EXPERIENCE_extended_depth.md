# EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE — extended depth

**Parent hub:** [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)

# 42. Evaluation and Benchmarking Operations

Evaluation is a **first-class runtime subsystem**, not a post-hoc script.

## 42.1 Modes

| Mode | Purpose |
|------|---------|
| Offline | Golden datasets, regression before merge |
| Online | Production sampling, score trends |
| Shadow | Compare candidate path without user impact |
| Human | HITL rubric scoring |

## 42.2 Components

| Module | Role |
|--------|------|
| `runtime/architecture/evaluation_modes.py` | Mode contracts |
| `evaluation_automation.py` | Runner automation |
| `evaluation_registry_trends.py` | Score history / trends |
| `online_evaluation_registry.py` | Live eval registry |
| `evaluation_assets.py` | Golden asset catalog |
| `runtime/eval/` | NexusEvalRunner integration |

Evaluators: rule-based, schema, LLM-judge (see [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md)).

**Plan:** [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) Phase EVAL · [`plan/CRITIC_VERIFICATION.md`](../plan/CRITIC_VERIFICATION.md) CRIT-V.

---
