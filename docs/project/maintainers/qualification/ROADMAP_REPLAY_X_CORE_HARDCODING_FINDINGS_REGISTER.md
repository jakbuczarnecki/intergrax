# ROADMAP-REPLAY-X — `CORE_HARDCODING_FINDINGS_REGISTER` (VBN §2.2)

| Field | Requirement |
| --- | --- |
| **Artifact** | `CORE_HARDCODING_FINDINGS_REGISTER` |
| **Historical SSOT** | [`EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md`](EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md) |
| **Python SSOT** | `tests/qualification/ebh_6/_findings_register.py` |
| **AUDITED_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |

## Replay revalidation

| Metric | @ HEAD | Source |
| --- | --- | --- |
| PROHIBITED unresolved (applicable scope) | **0** (register + pytest) | ebh_6 mechanical gates |
| New PROHIBITED from prod delta | **0** | empty prod diff since DUP-X |

**Disposition:** **REUSE** EBH-6 register; replay counter `vendor_neutrality_replay_failures` reads OPEN PROHIBITED rows mechanically.
