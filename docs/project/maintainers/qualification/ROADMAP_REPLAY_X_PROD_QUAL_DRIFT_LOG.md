# ROADMAP-REPLAY-X — Production / qualification drift log (W3)

| Field | Value |
| --- | --- |
| **AUDITED_HEAD** | `5f54ba76d68e9065256ac4597bc8e8d9e012b6d2` |
| **Prod baseline** | DUP-X acceptance `a7d83925b375b5f08eb0f175ced2c17fa026bba9` |

## Production delta (`intergrax/` · `agents/` · `applications/`)

| Scope | `git diff a7d83925..HEAD` | Reopen? |
| --- | --- | --- |
| **Scoped (REPLAY-X L2 + PRE-ENT L3)** | `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py` — CTRL-X pyright + ruff; regression `tests/qualification/control_planes/ctrl_x/test_ctrl_x_typing_gates.py` @ L2 head `91649e5` · plus PRE-ENT predecessor + 034-B paths @ `01b3e196` — [`PRE_ENT_REPLAY_L3_PROD_DELTA.md`](pre_ent_enterprise_certification/PRE_ENT_REPLAY_L3_PROD_DELTA.md) | **no** (scoped attestation) |
| **Unscoped** (mandatory §5 parent anchors) | **empty** @ `01b3e196` | **no** |

**Distinction:** production delta since DUP-X is **fully scoped** @ PRE-ENT Wave 1 replay head; L2 CTRL-X historical evidence is **not** invalidated — `historical_evidence_invalidated_by_later_changes` remains **0** (@ `_closure_counters.py` + [`_prod_delta_disposition.py`](../../../../tests/qualification/roadmap_replay_x/_prod_delta_disposition.py)).

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

## Wave 06 addendum (T-IGX-066-A)

| Scope | git diff a7d83925..HEAD | Reopen? |
| --- | --- | --- |
| **Scoped (PRE-ENT L3 + W06 EQO)** | 88 historical + **418** W06 paths — [W06_REPLAY_PROD_DELTA.md](pre_ent_enterprise_certification/W06_REPLAY_PROD_DELTA.md) | **no** |
| **Unscoped** | **empty** @ mechanical replay | **no** |

| Counter | Value @ W06 mechanical replay |
| --- | --- |
| historical_evidence_invalidated_by_later_changes | 0 |
| vendor_neutrality_replay_failures | 0 |
| B6 R2 ecap_counter_vs_mechanical | **empty** |

## Factory train-pin addendum (ec-v6-ec02-closure S07D)

| Scope | git diff a7d83925..HEAD @ integrate worktree | Reopen? |
| --- | --- | --- |
| **Scoped (W06 EQO + train-pin delta)** | +44 paths (factory S01–S07 integrates) — [`W06_REPLAY_PROD_DELTA.md`](pre_ent_enterprise_certification/W06_REPLAY_PROD_DELTA.md) | **no** |
| **Unscoped** | **empty** @ mechanical replay | **no** |

Mechanical replay: `test_roadmap_replay_x_mechanical_exit_gates.py` **16/16 PASS**, `test_roadmap_replay_x_scoped_path_net_zero.py` **PASS** (S07D locked pytest).
