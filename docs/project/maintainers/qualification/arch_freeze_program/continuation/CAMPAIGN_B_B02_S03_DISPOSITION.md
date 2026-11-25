# Campaign B b2 rerun — S03 disposition (af-fc/b02-rerun)

**Execution SHA:** `87b1a7bec29c603515b058c8337402b1e813cc56`  
**Runner result commit:** `22288975a0dd8a120ff658a85d2d6a4b18e41a00`  
**Worktree:** `D:/Projekty/_wt/intergrax/af-fc-b02` (branch `af-fc/b02-rerun`)

## Node id and command

- **S03** (owner **PT-B02**, contract AB-10, ctx `HEAD_DIAGNOSTIC`)
- **pytest:** `pytest tests/qualification/harness_01 -q --maxfail=3 --tb=line`
- **Runner:** exit `1`, result **FAIL** (3 setup **ERROR**s, not assertion failures; 48 passed)

## Quoted failure

```
ERROR at setup of test_harness_01_public_authoring_surfaces_do_not_reference_nexus_as_import_api
E   subprocess.CalledProcessError: Command '[...\\.venv\\Scripts\\python.exe', '-m', 'pip', 'install', '...\\examples\\platform_plugins\\intergrax_reference_enterprise_plugin', '-q']' returned non-zero exit status 1.
Captured stderr setup: ... No module named pip

ERROR at setup of test_harness_01_public_authoring_negative_synthetic_nexus_import_fails
ERROR at setup of test_harness_01_public_authoring_positive_public_abi_import_allowed
(same pip install / No module named pip)
```

Evidence: `.tmp/session/w2-recovery-03/campaign-b/b2/HEAD_DIAGNOSTIC/87b1a7bec29c/default/S03/stdout.txt`

## Classification

**MISSING_EXTRA** — campaign worktree venv created by `uv` has no `pip` module; plugin fixture `pip install` in test setup cannot run. No product-behavior assertion failed on this rerun (distinct from prior Nexus importer assertion remediated @ `4b7acac4`).

## Product change

**None** on this commit.