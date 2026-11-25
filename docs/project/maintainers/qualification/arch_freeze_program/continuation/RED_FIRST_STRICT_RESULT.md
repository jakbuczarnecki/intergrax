# RED_FIRST strict result

Integration head: `a678671fe6dfc95bc74b8095387146dbcc5d361a` (`af-fc/red-strict`). Worktree: `D:\Projekty\_wt\intergrax\af-fc-reds`.
Strict pass on **17** `RED_FIRST_DEFECT` rows from replay.

## Totals

| measurement | count |
| --- | ---: |
| RED_AT_PREFIX | 9 |
| RED_FIRST_UNREPRODUCIBLE | 8 |
| NOT_RUN | 0 |
| INTEGRATED_FAIL | 0 |

Strict mode: detached checkout at `red_first_prefix_sha` with negative tests overlaid from integration head when absent at prefix; `RED_AT_PREFIX` only when pytest exits **1** and output contains **FAILED**.

| frz_id | prefix_sha | strict exit | measurement | integrated ok | log |
| --- | --- | ---: | --- | --- | --- |
| `FRZ-BND-04` | `0cee41a88274…` | 1 | **RED_AT_PREFIX** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-BND-04.log` |
| `FRZ-BND-05` | `0cee41a88274…` | 1 | **RED_AT_PREFIX** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-BND-05.log` |
| `FRZ-CFG-06` | `0cee41a88274…` | 1 | **RED_AT_PREFIX** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-CFG-06.log` |
| `FRZ-CTR-01` | `0cee41a88274…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-CTR-01.log` |
| `FRZ-CTR-02` | `0cee41a88274…` | 1 | **RED_AT_PREFIX** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-CTR-02.log` |
| `FRZ-CTR-06` | `0cee41a88274…` | 1 | **RED_AT_PREFIX** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-CTR-06.log` |
| `FRZ-OWN-05` | `0cee41a88274…` | 1 | **RED_AT_PREFIX** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-OWN-05.log` |
| `FRZ-PLG-01` | `22993254b13a…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-PLG-01.log` |
| `FRZ-PLG-02` | `22993254b13a…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-PLG-02.log` |
| `FRZ-PLG-07` | `22993254b13a…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-PLG-07.log` |
| `FRZ-TEN-01` | `0cee41a88274…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-TEN-01.log` |
| `FRZ-TEN-02` | `0cee41a88274…` | 0 | **RED_FIRST_UNREPRODUCIBLE** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-TEN-02.log` |
| `FRZ-TEN-07` | `0cee41a88274…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-TEN-07.log` |
| `FRZ-TEN-10` | `0cee41a88274…` | 2 | **RED_FIRST_UNREPRODUCIBLE** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-TEN-10.log` |
| `FRZ-TYP-01` | `0cee41a88274…` | 1 | **RED_AT_PREFIX** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-TYP-01.log` |
| `FRZ-TYP-03` | `0cee41a88274…` | 1 | **RED_AT_PREFIX** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-TYP-03.log` |
| `FRZ-TYP-05` | `0cee41a88274…` | 1 | **RED_AT_PREFIX** | True | `D:/Projekty/_wt/intergrax/_scratch/af-fc-reds/logs/FRZ-TYP-05.log` |
