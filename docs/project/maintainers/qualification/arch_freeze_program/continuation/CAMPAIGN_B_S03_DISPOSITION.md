# Campaign B — S03 subprocess disposition (af-fc/b-alpha-s03)

**Train base:** `8c0158ac8152a7ff97734b34faf37c25682a77be` (`arch-freeze/fc-train`)  
**Branch / worktree:** `af-fc/b-alpha-s03` @ `D:/Projekty/_wt/intergrax/af-fc-s03`  
**Evidence (alpha replay):** `D:/Projekty/_wt/intergrax/_scratch/af-fc-b1/S03_harness_01_stdout.txt`

## Quoted setup errors

```
ERROR at setup of test_harness_01_public_authoring_surfaces_do_not_reference_nexus_as_import_api
E   subprocess.CalledProcessError: Command '['...\\.venv\\Scripts\\python.exe', '-m', 'pip', 'install', '...\\examples\\platform_plugins\\intergrax_reference_enterprise_plugin', '-q']' returned non-zero exit status 1.
Captured stderr setup: ... No module named pip

ERROR at setup of test_harness_01_public_authoring_negative_synthetic_nexus_import_fails
(same pip install / No module named pip)
```

## Classification

| Node | Class |
| --- | --- |
| `test_harness_01_public_authoring_surfaces_do_not_reference_nexus_as_import_api` | **missing_extra** — campaign worktree venv lacks `pip`; plugin fixture install cannot run |
| `test_harness_01_public_authoring_negative_synthetic_nexus_import_fails` | **missing_extra** — same setup dependency |

No product change authorized for these nodes on this branch (orchestrator: environment/setup class, not MEP-AF2 §G product assertion).

## Product node (separate)

`FAILED test_harness_01_shared_host_nexus_importers_are_w6_debt_not_final_legal` — **product assertion** (HOST_COMPOSITION Nexus importers = 0 post convergence); remediated on this branch via EE-owned `session_host_composition` import surface and inventory row removal.