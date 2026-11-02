# EVOL-X B6 — Mode B replay provenance (valid v2)

**Independent replay SHA (HEAD):** `3c4a641b3e48a13f437e830a0562e2a80b03aabb`  
**Evidence ref (--evidence-ref):** `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1`

## Commands (session v2)

```powershell
# @ wt-tes worktree on TES
uv run python scripts/qualification/evol_x_vl3_manifest.py --check
uv run pytest tests/qualification/evol_x -q `
  --basetemp=d:\Projekty\intergrax\build\pytest-b6-tes-replay `
  --junitxml=.tmp/session/evol-x-b6/junit-v2.xml
uv run python scripts/qualification/evol_x_p03_evidence.py verify-test-execution `
  --evidence-ref bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1 `
  --certification --replay-junit .tmp/session/evol-x-b6/junit-v2.xml
```

## Results

| Step | Log | Outcome |
|------|-----|---------|
| VL3 | `a2-vl3-v2.log` | 290 node ids |
| Pytest | `a2-pytest-v2.log` | 290 passed (~457s) |
| Mode B | `a2-mode-b-v2.log` | `EVIDENCE_INTEGRITY_OK`, mode B |

## Replay JUnit fingerprint (local v2)

| File (session) | SHA256 |
|----------------|--------|
| `wt-tes/.tmp/session/evol-x-b6/junit-v2.xml` | `2a0c675dce2976ff510a6d870b5afb827a6289f13a8353fc6b295be4195e6c8e` |

**Not equal** to CI JUnit SHA256 — expected (platform/runner/xml formatting); outcome oracle is canonical id map @ FINAL.

## Disposition

**G-06 technical PASS** (v2) — see F-001 for superseded invalid attempt.
