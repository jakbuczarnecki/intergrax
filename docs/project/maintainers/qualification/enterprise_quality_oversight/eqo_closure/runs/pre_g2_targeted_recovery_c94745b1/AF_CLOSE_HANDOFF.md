# AF-CLOSE handoff — EC-W3 targeted recovery

Execution SHA: `c94745b11b35a3be55e4e9cb6502befc60262e26`

**PRE_G2_TARGETED_DIAGNOSTIC_EXECUTION — NOT FINAL QUALIFICATION**

## Outstanding failure families

- ACT-N11 / RB-2: unresolved owner route (not replayed here).
- EVOL-X B4: not in batch scope.
- BND05 / T-IGX-043-H: `test_nt_bnd_05_2_provider_imports_stay_on_the_closed_list` — owner ticket; excluded.
- MCP P5: child `test_mcp_trust_a_principal_binding.py` exit 0 (`logs/mcp_trust_a.log`); wrapper still fails on `test_mcp_server_p4_lifecycle.py::test_lab_application_omits_mcp_mount_when_disabled` (`logs/mcp_p5.log`).
- Harness-01: see `logs/harness_01.log`.
- PROD-Q PB-S13: see `logs/prod_q_pb_s13.log`.
- ROADMAP-REPLAY-X: see `logs/roadmap_replay_x.log`.
- XD-12: see `logs/xd12.log`.
- Remaining provider qualification: not closed.
- Remaining post-G2 RV-01: not closed.
