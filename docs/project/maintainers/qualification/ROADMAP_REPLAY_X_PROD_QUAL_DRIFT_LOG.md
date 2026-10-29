# ROADMAP-REPLAY-X — Production / qualification drift log (W3)

| Field | Value |
| --- | --- |
| **AUDITED_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **Prod baseline** | DUP-X acceptance `a7d83925b375b5f08eb0f175ced2c17fa026bba9` |

## Production delta (`intergrax/` · `agents/` · `applications/`)

| Parent anchor | `git diff a7d83925..HEAD` | Reopen? |
| --- | --- | --- |
| All mandatory parents (§5.1 matrix) | **empty** | **no** |

Mechanical gate: `test_replay_x_prod_delta_empty_since_dup_x`.

## Qualification delta (`tests/qualification/`)

| Path | Delta vs DUP-X pin | L2 obligation |
| --- | --- | --- |
| `dup_x/` | new modules @ HEAD | **mandatory rerun** @ TEST_EXECUTION_SHA |
| `ebh_6/` | handoff / export paths | inspect + selective rerun |
| `roadmap_replay_x/` | REPLAY-X SSOT (this wave) | mechanical integrity gates |

## Counter impact @ implementation wave

| Counter | Value | Notes |
| --- | --- | --- |
| reopened_historical_stages | 0 | no prod drift |
| historical_evidence_invalidated | 0 | accepted §5 SHAs intact |
| unresolved_replay_blockers | 0 | no BLOCKER register rows |
