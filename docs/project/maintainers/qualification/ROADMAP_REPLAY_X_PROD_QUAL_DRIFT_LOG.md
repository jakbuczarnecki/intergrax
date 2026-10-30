# ROADMAP-REPLAY-X — Production / qualification drift log (W3)

| Field | Value |
| --- | --- |
| **AUDITED_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **Prod baseline** | DUP-X acceptance `a7d83925b375b5f08eb0f175ced2c17fa026bba9` |

## Production delta (`intergrax/` · `agents/` · `applications/`)

| Scope | `git diff a7d83925..HEAD` | Reopen? |
| --- | --- | --- |
| **Scoped (REPLAY-X L2 disposition)** | `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py` — CTRL-X pyright callback typing + ruff format; admission/shutdown semantics unchanged · regression `tests/qualification/control_planes/ctrl_x/test_ctrl_x_typing_gates.py` · **L2 PASS** @ `TEST_EXECUTION_SHA` `22fbc0e` (`l2-ctrl_x.log`) | **no** (scoped attestation) |
| **Unscoped** (mandatory §5 parent anchors) | **empty** | **no** |

**Distinction:** production delta **exists (scoped)** for one CTRL-X-adjacent path; **historical accepted evidence is NOT invalidated** for unrelated paths — `historical_evidence_invalidated_by_later_changes` remains **0** (@ `_closure_counters.py` + [`_prod_delta_disposition.py`](../../../../tests/qualification/roadmap_replay_x/_prod_delta_disposition.py)).

Mechanical gate: `test_replay_x_prod_delta_empty_since_dup_x` asserts **unscoped** emptiness.

## Qualification delta (`tests/qualification/`)

| Path | Delta vs DUP-X pin | L2 obligation |
| --- | --- | --- |
| `dup_x/` | new modules @ HEAD | **mandatory rerun** @ TEST_EXECUTION_SHA |
| `ebh_6/` | handoff / export paths | inspect + selective rerun |
| `roadmap_replay_x/` | REPLAY-X SSOT (this wave) | mechanical integrity gates |

## Counter impact @ implementation wave

| Counter | Value | Notes |
| --- | --- | --- |
| reopened_historical_stages | 0 | no **unscoped** prod drift |
| historical_evidence_invalidated_by_later_changes | 0 | scoped path excluded; accepted §5 SHAs intact elsewhere |
| unresolved_replay_blockers | 0 | `_replay_blocker_register.py` (empty register @ handoff) |
