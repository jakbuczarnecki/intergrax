# DECISION_VERIFICATION — implementation pipeline

**Parent hub:** [`DECISION_VERIFICATION.md`](../DECISION_VERIFICATION.md)

## Phase DS-VER-PIPE — Pipeline foundation (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-VER-PIPE-01 | P0 | Verification Pipeline orchestrator contract | **Planned** |
| DS-VER-PIPE-02 | P0 | Stage plugin interface + registration | **Planned** |
| DS-VER-PIPE-03 | P0 | Verification Result + Challenge typed contracts | **Planned** |
| DS-VER-PIPE-04 | P0 | Deterministic-before-probabilistic ordering | **Planned** |
| DS-VER-PIPE-05 | P1 | Challenge → Lifecycle handoff (no in-place mutation) | **Planned** |
| DS-VER-PIPE-06 | P1 | Fail-closed unavailable required stage | **Planned** |
| DS-VER-PIPE-07 | P2 | Stage telemetry → Observability | **Planned** |

## Phase DS-VER-STAGES — Stage migration from CVL (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-VER-STAGE-L0 | P0 | Structural/deterministic stage from `L0Gateway` | **Planned** |
| DS-VER-STAGE-SEM | P1 | Semantic stage from `L1Gateway` / `eval.judge` | **Planned** |
| DS-VER-STAGE-TRAJ | P1 | Trajectory stage from `eval.trajectory` | **Planned** |
| DS-VER-STAGE-EVID | P1 | Evidence verification stage | **Planned** |
| DS-VER-STAGE-GR | P2 | Guardrail merge from `guardrail_l0` | **Planned** |
| DS-VER-STAGE-DOM | P2 | Independent/domain verifier stage | **Planned** |

## Open requirements (migrated from Critic audit)

| ID | Priority | Status |
|----|----------|--------|
| DS-VER-RUBRIC-PROVENANCE-INTEGRITY | P0/P1 | ACCEPTED / PLANNED |
| DS-VER-PRODUCER-INDEPENDENCE | P0/P1 | ACCEPTED / PLANNED |
| DS-VER-ADVERSARIAL-SEMANTIC | P1 | ACCEPTED / PLANNED |
| DS-VER-RESULT-COHERENCE | P1/P2 | ACCEPTED / PLANNED |
