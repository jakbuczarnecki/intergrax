# AF_CLOSE_HANDOFF — EC-W21-W3 pre-G2 batch

Evidence-only handoff from diagnostic replay. No AF-CLOSE implementation performed.

## Mandatory dependency groups
- **ACT-N11 / RB-2**: IIAC activation gate failures (WB01, governance-fast EXP-RED).
- **EVOL-X B4**: ten HIST preflight nodes; fixture classification clarified, semantics unresolved.
- **ENT-AUDIT**: historical five ENT-A gate failures unchanged.
- **B-1 product/governance failures**: see B1_REPLAY_MATRIX.json failed_test_node_ids.
- **Production qualification / external provider**: not in this batch scope.
- **Post-G2 RV-01**: not executed.

## Newly observed failures
See `runs/pre_g2_batch_786088a9/B1_REPLAY_MATRIX.json` and per-workstream `pre_g2_replay_786088a9/pytest.log`.

Execution SHA: `786088a9798ec18a26c22778f8217608e5f15bff`.
