# RED_FIRST rest result

Integration head: `b1107cb215a4ebd1c07c5434d04224dbf6ee71b2` (`af-v4/p1b-red-rest`). Worktree: `D:\Projekty\_wt\intergrax\af-v4-p1b`.
Measured **20** residual `EVIDENCE_MISSING` rows.

## Totals

| measurement | count |
| --- | ---: |
| RED_AT_PREFIX | 5 |
| RED_FIRST_UNREPRODUCIBLE | 15 |
| EVIDENCE_MISSING | 0 |
| INTEGRATED_FAIL | 0 |

Strict mode: detached checkout at parent of earliest negative-test add commit; overlay negatives from integration head when absent at prefix; `RED_AT_PREFIX` only when pytest exits **1** with **FAILED**.

| frz_id | prefix_sha | strict exit | measurement | integrated ok | log |
| --- | --- | ---: | --- | --- | --- |
| `FRZ-BND-06` | `307a871c059f…` | 1 | **RED_AT_PREFIX** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-BND-06.log` |
| `FRZ-CFG-02` | `754ee000675d…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-CFG-02.log` |
| `FRZ-CFG-08` | `754ee000675d…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-CFG-08.log` |
| `FRZ-DEBT-04` | `8bdb0038d7a8…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-DEBT-04.log` |
| `FRZ-EXE-02` | `b7855b83e98e…` | 1 | **RED_AT_PREFIX** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-EXE-02.log` |
| `FRZ-FRZ-10` | `332c5217c390…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-FRZ-10.log` |
| `FRZ-OWN-02` | `754ee000675d…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-OWN-02.log` |
| `FRZ-OWN-03` | `754ee000675d…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-OWN-03.log` |
| `FRZ-PLG-05` | `459dfebc712f…` | 1 | **RED_AT_PREFIX** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-PLG-05.log` |
| `FRZ-PLG-06` | `2b12af1dc502…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-PLG-06.log` |
| `FRZ-PRD-03` | `754ee000675d…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-PRD-03.log` |
| `FRZ-PRD-05` | `754ee000675d…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-PRD-05.log` |
| `FRZ-PRD-08` | `754ee000675d…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-PRD-08.log` |
| `FRZ-REG-09` | `754ee000675d…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-REG-09.log` |
| `FRZ-REL-07` | `66664e2bffa1…` | 3 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-REL-07.log` |
| `FRZ-REL-09` | `ec7f1e27c4a0…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-REL-09.log` |
| `FRZ-REL-10` | `a01b7e415797…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-REL-10.log` |
| `FRZ-RPL-04` | `754ee000675d…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-RPL-04.log` |
| `FRZ-SEC-04` | `ccc7287c4067…` | 1 | **RED_AT_PREFIX** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-SEC-04.log` |
| `FRZ-SEC-06` | `459dfebc712f…` | 1 | **RED_AT_PREFIX** | True | `.tmp/session/arch-freeze/red-first-rest/logs/FRZ-SEC-06.log` |
