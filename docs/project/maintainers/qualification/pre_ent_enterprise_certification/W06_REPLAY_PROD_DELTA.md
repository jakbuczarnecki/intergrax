# Wave 06 — ROADMAP-REPLAY-X prod-Δ (OD-REPLAYX-09 / T-IGX-066-A)

| Field | Value |
| --- | --- |
| **DUP-X pin** | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` |
| **W06 qualification head** | `da5ca5b53fbe510e655b12bc91ab4e6af1cef6e4` (read pin) |
| **SSOT modules** | `_w06_eqo_prod_delta.py`, `_prod_delta_disposition.py` |
| **Batches** | W06-B01..B11 (11 classification bands) |
| **New scoped paths** | 418 @ evidence pin `da5ca5b53`; **+44** train-pin reconcile @ `factory/ec-v6-ec02-closure` (S07D / ec-v6-closure integrate) |

Mechanical contract: `test_roadmap_replay_x_mechanical_exit_gates.py` — **16/16 PASS** after disposition merge; scoped_path_net_zero **PASS** @ S07D pytest lock.

Evidence emit helper (session only): `.tmp/session/w06/emit_w06_eqo_prod_delta.py`.