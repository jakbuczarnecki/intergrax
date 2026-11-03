# ENT-AUDIT-X — one-pass agent handoff (Protocol 03 / B6)

**Agent disposition (wave 2):** `ENT-AUDIT-X — IMPLEMENTATION PARTIAL / EXACT BLOCKERS`  
**Do not treat as:** B6 PASS, OD approval, FRZ CLOSED, ARCH-FREEZE, platform QUALIFIED @ pinned TES.

## Wave 2 integration (security + R1/R2/R3 harness)

- **VBN provenance (C008/C010):** `validate_domain_report` — run_id, SHA40, manifest digest pin, exact D01–D31, unauthorized NA rejection; runner pins `expected_manifest_digest`.
- **Evidence (C006/C026):** `assert_certifying_publication_eligible` — closed-world 85 outcomes, exit 0, no skips, FULL label, run_status completed; bundle fields include `audit_execution_sha` + `check_manifest_digest`.
- **Baseline/drift (C022/C023/C028):** check schema version in REGRESSED; `check_manifest_digest` on baseline/candidate; FREEZE certifying digest guard.
- **Sandbox (C014):** per-run `ENT_AUDIT_SANDBOX_RUN_ID` isolation env.
- **CHANGED-SINCE (C024/C029):** `scripts/enterprise_audit/build_impact_map.py`.
- **Semantic acceptance (C033–C036):** 85 wrappers call `_gate_oracle` (skip unless `ENT_AUDIT_X_SEMANTIC=1`).
- **CI (C007):** nightly no auto FREEZE bootstrap / evidence bundle.

## Qual evidence @ HEAD (pre-TES)

```powershell
uv run pytest tests/qualification/ent_audit_x/unit tests/qualification/ent_audit_x/integration tests/qualification/ent_audit_x/l3 -q
# 108 passed (semantic 85 matrix opt-in)
$env:ENT_AUDIT_X_SEMANTIC='1'
uv run pytest tests/qualification/ent_audit_x/acceptance -q
```

## Exact blockers before §7 / B6

| Blocker | Detail |
| --- | --- |
| **TEST_EXECUTION_SHA** | Not pinned post wave-2 commit |
| **Real certifying FULL 85/85 exit 0** | Not demonstrated (semantic matrix + runner) |
| **ADV31 @ TES** | Not replayed @ pinned SHA |
| **EVIDENCE_PUBLICATION_SHA / B6** | Unset |
| **Matrix C doc reconciliation** | Appendix dual-axis needs operator sync post-qual |
| **Independent Protocol 03 B6** | Required |

## Publication

Append **IMPLEMENTATION_SHA (wave 2)** on [ENT_AUDIT_X_SHA_LADDER.md](./ENT_AUDIT_X_SHA_LADDER.md) after commit.
