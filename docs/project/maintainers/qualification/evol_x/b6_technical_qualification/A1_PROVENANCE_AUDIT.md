# A1 — SHA, CI and provenance audit

**Agent:** A1 (read-only) · **Baseline HEAD:** `f5b23ffacdd5a9ecf3653d7988c338b5b705c2d5`

## Per-gate summary

| Check | Status | Evidence | Blocker |
|-------|--------|----------|---------|
| TES ancestry @ orchestrator HEAD | **PASS** | `git merge-base --is-ancestor 3c4a641b… HEAD` | — |
| FINAL ancestry @ orchestrator HEAD | **PASS** | `git merge-base --is-ancestor bbf3e33b… HEAD` | — |
| Remote alignment | **PASS** | `origin/development` = `f5b23ffac…` | — |
| VL3 population @ HEAD | **PASS** | `evol_x_vl3_manifest.py --check` → 290 node ids | — |
| Qualified test tree frozen since TES | **PASS** | `git diff 3c4a641b…..HEAD -- tests/qualification/evol_x` empty | — |
| Post-TES script drift | **NOTE** | `scripts/qualification/evol_x_finalize_*` changed on `development` after TES; **does not alter** qualified VL3 tree | Document only |
| CI run `41337077334` | **PASS** | `gh run view` → success; log: checkout + `rev-parse HEAD` = TES | — |
| CI JUnit SHA256 (canonical) | **PASS** | [EVOL_X_CI_QUALIFICATION_RECORD.md](../EVOL_X_CI_QUALIFICATION_RECORD.md): `69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4` | Local artifact copy **absent** (see limitation) |
| Local replay JUnit byte identity vs CI | **N/A (by design)** | Mode B matches **outcome population** @ FINAL, not byte-identical XML | — |
| Replay JUnit SHA256 (session v2) | **RECORDED** | `2a0c675dce2976ff510a6d870b5afb827a6289f13a8353fc6b295be4195e6c8e` @ `junit-v2.xml` | — |
| MANIFEST @ FINAL | **PASS** | `certification_profile=AUTOMATION_FIRST_SCOPED`, `implementation_sha`=TES | — |
| ARTIFACT_DIGESTS @ committed FINAL bundle | **PASS** | [evol_x_p03_independent_replay/ARTIFACT_DIGESTS.json](../evol_x_p03_independent_replay/ARTIFACT_DIGESTS.json) @ `bbf3e33b…` tree | — |
| Historical TECH vs active FINAL | **PASS** | TECH `1e175ff8…` @ superseded TES `2b165516…`; active FINAL `bbf3e33b…` @ TES `3c4a641b…` | — |
| Stale SHA in active ladder | **PASS** | [EVOL_X_SHA_LADDER.md](../EVOL_X_SHA_LADDER.md) matches CI record | — |
| B6_ACCEPTANCE_PUBLICATION_SHA | **OPEN** | Ladder field empty (required) | G-07 formal |

## SHA chain (active)

| Role | SHA |
|------|-----|
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| EVIDENCE_TECH (historical) | `1e175ff8f678d605aa841ae52362486a56ed006d` |
| CI qualified run | `41337077334` @ TES |

## Evidence limitation

Gitignored path `.tmp/session/evol-x-w6-04-requal-3c4a641/junit.xml` referenced in CI record was **not present** on orchestrator workstation at consolidation time. CI qualification relies on GitHub run logs + published `junit_sha256` in [EVOL_X_CI_QUALIFICATION_RECORD.md](../EVOL_X_CI_QUALIFICATION_RECORD.md) and workflow artifact name `evol-x-p3-l3-3c4a641b…`.

## Disposition

**A1 — TECHNICAL PASS** (provenance reconciled; no invalidation of TES population detected).
