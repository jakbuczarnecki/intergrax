# EVOL-X — Final B6 audit dossier (Path A — ready for independent signature)

> **Disposition:** **TECHNICAL EVIDENCE COMPLETE / FORMAL PROTOCOL 03 UNSIGNED**  
> **Not:** `B6_ACCEPTANCE_PUBLICATION_SHA` · not EVOL-X CLOSED · not FRZ promotion

**Audience:** Named **OD-EVOL-11** independent auditor only.  
**Appointment:** [EVOL_X_OD_EVOL_11_INDEPENDENT_AUDITOR_APPOINTMENT_RECORD.md](./EVOL_X_OD_EVOL_11_INDEPENDENT_AUDITOR_APPOINTMENT_RECORD.md)  
**W7-01 package:** [EVOL_X_W7_01_FORMAL_INDEPENDENT_B6_ACCEPTANCE_PACKAGE.md](./EVOL_X_W7_01_FORMAL_INDEPENDENT_B6_ACCEPTANCE_PACKAGE.md)  
**Commands SSOT:** [EVOL_X_B6_INDEPENDENT_AUDIT_PACKAGE.md](./EVOL_X_B6_INDEPENDENT_AUDIT_PACKAGE.md)

---

## 1. Scoped acceptance wording (on PASS only)

**Disposition:** `INDEPENDENTLY ACCEPTED — AUTOMATION-FIRST SCOPED`

**Exclusions preserved:** ARCH-FREEZE, global FRZ PASS, PROD-Q live slice, ENT-AUDIT-X CLOSED, production rollback port (OD-EVOL-05b Option 1), **M-09H NOT_MEASURED** on active FINAL.

---

## 2. Exact SHA references

| Role | SHA |
|------|-----|
| TES | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| FINAL | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| CI run | `41337077334` (290/290 VL3 @ TES) |
| CI JUnit SHA256 (canonical) | `69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4` |

---

## 3. Reproducible audit command sequence

Run from repository root. Auditor MUST use clean checkout; implementer session logs are reference only.

### Mode A @ FINAL

```powershell
cd <repo>
git checkout bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1
uv sync --extra dev-ci --frozen
uv run python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish
```

**Expected:** exit 0 · `EVIDENCE_INTEGRITY_OK` · `certification_profile=AUTOMATION_FIRST_SCOPED` · **EVOL-M-09H NOT_MEASURED**.

### Mode B @ TES (independent replay)

```powershell
git checkout 3c4a641b3e48a13f437e830a0562e2a80b03aabb
uv sync --extra dev-ci --frozen
New-Item -ItemType Directory -Force -Path .tmp/session/evol-x-b6-auditor | Out-Null
uv run pytest tests/qualification/evol_x -q `
  --basetemp=<repo>/build/pytest-b6-auditor-replay `
  --junitxml=.tmp/session/evol-x-b6-auditor/junit.xml
uv run python scripts/qualification/evol_x_p03_evidence.py verify-test-execution `
  --evidence-ref bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1 `
  --certification --replay-junit .tmp/session/evol-x-b6-auditor/junit.xml
```

**Expected:** **290** manifest nodes · exit 0 · `--evidence-ref` MUST be **FINAL**, not TES.

### Optional orchestrator (technical bundle; not organizational sign-off)

```powershell
git checkout <auditor-working-branch-or-detached-FINAL-parent>
uv run python scripts/qualification/evol_x_b6_independent_verification.py `
  --session-root .tmp/session/evol-x-b6-auditor-orchestrator `
  --tes 3c4a641b3e48a13f437e830a0562e2a80b03aabb `
  --evidence-final bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1
```

---

## 4. Findings and residual risks (summary)

| ID | Status | Formal impact |
|----|--------|---------------|
| F-001 | CLOSED | Procedural G-06 invalid ref preserved |
| F-003 | **OPEN** | **Blocks B6** — G-02 auditor identity |
| F-004 | **OPEN** | **Blocks B6** — G-07 unsigned Protocol 03 |
| F-005 | **OPEN** | W7-03 FRZ-EVOL-07 Outcome A/B |
| F-006 / F-007 | **OPEN** | OD-EVOL-12 / OD-EVOLX-08 |
| F-008 | N/A | Scoped M-09H policy correct |

Full register: [b6_technical_qualification/EVOL_X_B6_FINDINGS_REGISTER.md](./b6_technical_qualification/EVOL_X_B6_FINDINGS_REGISTER.md)

**102-ID matrix:** [b6_technical_qualification/EVOL_X_B6_102_ID_COVERAGE_MATRIX.md](./b6_technical_qualification/EVOL_X_B6_102_ID_COVERAGE_MATRIX.md)

---

## 5. FRZ-EVOL-07 (auditor selects one)

Under OD-EVOL-05b Option 1, auditor records **Outcome A** or **Outcome B** in signed Protocol 03 before W7-03. Dossier: [b6_technical_qualification/EVOL_X_B6_FRZ_EVOL_SCOPED_READINESS.md](./b6_technical_qualification/EVOL_X_B6_FRZ_EVOL_SCOPED_READINESS.md)

---

## 6. Signature procedure (auditor)

1. Complete Mode A + Mode B; record commands, exit codes, HEAD SHAs in PASS record.
2. Publish `docs/project/maintainers/qualification/evol_x/EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md` with auditor signature block (name, date, independence attestation).
3. Commit on `development`; set **`B6_ACCEPTANCE_PUBLICATION_SHA`** = that commit on [EVOL_X_SHA_LADDER.md](./EVOL_X_SHA_LADDER.md).

**Forbidden:** agent-signed PASS; renaming draft without signature.

---

## 7. Verification procedure (operator / maintainer)

1. Confirm signer matches register **OD-EVOL-11** `auditor_id`.
2. Confirm PASS record cites exact TES/FINAL/CI SHAs above.
3. Re-run Mode A @ FINAL and spot-check Mode B or accept auditor-attested replay with JUnit digest.
4. Run `uv run python scripts/qualification/evol_x_closure_publication_preflight.py` — must pass before W7-05.

---

## 8. Unsigned draft (do not treat as PASS)

[b6_technical_qualification/EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_RECORD_DRAFT.md](./b6_technical_qualification/EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_RECORD_DRAFT.md)
