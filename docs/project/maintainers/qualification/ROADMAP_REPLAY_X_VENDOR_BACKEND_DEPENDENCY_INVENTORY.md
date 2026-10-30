# ROADMAP-REPLAY-X — `VENDOR_BACKEND_DEPENDENCY_INVENTORY` (VBN §2.1)

| Field | Requirement |
| --- | --- |
| **Artifact** | `VENDOR_BACKEND_DEPENDENCY_INVENTORY` |
| **Producing stage** | EBH-5; replay **ROADMAP-REPLAY-X** |
| **Historical SSOT** | [`EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md`](EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md) @ **`33ecf19cc020b112d9e586c87b49e819202f5d6d`** |
| **AUDITED_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **Prod delta** since DUP-X | **empty** |
| **L2 attestation** | `l2-ebh_5.log` @ `.tmp/proof/roadmap-replay-x-immutable-6fbc4a8/` (**EVIDENCE_EXPORT_REQUIRED**) |
## Replay revalidation (@ AUDITED_HEAD)

| Check | Result | Evidence |
| --- | --- | --- |
| Closed-world row count unchanged vs EBH-5 B6 | **PRESERVE** | `tests/qualification/ebh_5/` closed-world gates |
| New production-effective vendor deps | **0** | `git diff a7d83925..HEAD -- intergrax agents applications` |
| Reachability graph module | still absent (scoped) | EBH-5 B6 record |

**Disposition:** **REUSE EBH-5 inventory** + delta attestation; **not** a new global FRZ PASS.

Mechanical owner: `tests/qualification/ebh_5/_closed_world_discovery.py`.
