# ROADMAP-REPLAY-X — `VENDOR_SELECTION_REACHABILITY_EVIDENCE` (VBN §2.5)

| Field | Requirement |
| --- | --- |
| **Artifact** | `VENDOR_SELECTION_REACHABILITY_EVIDENCE` |
| **Historical SSOT** | [`EBH_5_VENDOR_SELECTION_REACHABILITY.md`](EBH_5_VENDOR_SELECTION_REACHABILITY.md) |
| **EBH-6 L3** | `tests/qualification/ebh_6/_l3_reachability_evidence.py` (**49** mechanisms) |
| **AUDITED_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |

## Replay revalidation

Prod-effective reachability **unchanged** @ HEAD (prod Δ empty). Graph bootstrap module absence remains **scoped** EBH-5 B6 non-claim.

**Disposition:** **REUSE** reachability dossier + ebh_6 L3 annex; selective rerun only if qual delta touches `ebh_6/` export manifest.
