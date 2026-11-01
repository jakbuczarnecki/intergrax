# EVOL-X — Independent B6 audit package (E3 PREP)

**Disposition:** preparation only — **no B6 PASS**, **no FRZ promotion**, **no agent signature**.

**Authority:** [EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md](../EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md) Part V–VII; [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](../ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) B6; [EVOL_X_AUTOMATION_FIRST_SCOPED_ACCEPTANCE_POLICY.md](./EVOL_X_AUTOMATION_FIRST_SCOPED_ACCEPTANCE_POLICY.md) (OD-EVOL-13).

## Active SHA ladder (closure chain)

| Role | SHA |
|------|-----|
| TEST_EXECUTION_SHA (TES) | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_TECH_PUBLICATION_SHA (historical) | `1e175ff8f678d605aa841ae52362486a56ed006d` @ TES `2b165516…` — **not** active FINAL |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` — **AUTOMATION_FIRST_SCOPED** (OD-EVOL-13) |
| B6_ACCEPTANCE_PUBLICATION_SHA | *pending independent auditor* |
| DOC / prep tip @ session | `22bf04b85a9fd4775ac2dd765f297bbf54df4e90` (B6 reconciliation baseline) |

**Superseded (forbid in active records):** TES `2b165516…`, `dcd7ae6e…`, `6fec9ec…`; TECH `9f875fc…`, `43a44b…`; human-only FINAL expectations without profile selection.

**Prerequisite:** Named **OD-EVOL-11** auditor (≠ implementer, ≠ Cursor agent) in [EVOL_X_OPERATOR_DECISION_REGISTER_W1.md](./EVOL_X_OPERATOR_DECISION_REGISTER_W1.md) before B6 sign-off.

---

## Certification profiles (mandatory distinction)

P03 **does not** expose a `--certification-profile` CLI flag. Profile is selected by **bundle contents** + **`MANIFEST.certification_profile`** + **Mode A disposition** (`--technical` vs default final).

| Profile | Operator basis | Mode A @ evidence HEAD | M-09H | M-09A / drill artifact |
|---------|----------------|------------------------|-------|-------------------------|
| **HUMAN_FULL** | OD-EVOL-08-M1 | `verify-evidence-publish` (no `--technical`) on bundle with **`certification_profile` absent or `HUMAN_FULL`** | **3/3 MEASURED PASS** | `evol_m09_human_drill.v2.json` + worksheet |
| **AUTOMATION_FIRST_SCOPED** | OD-EVOL-13 | `verify-evidence-publish` (no `--technical`) on bundle with **`certification_profile=AUTOMATION_FIRST_SCOPED`** | **NOT_MEASURED** (required) | `evol_m09_automated_diagnostic_qualification.v1.json` + `m09_automated_diagnostic/C*.json`; **EVOL-M-09A 3/3 MEASURED PASS** |

**Active EVOL-X scoped closure path:** **AUTOMATION_FIRST_SCOPED** @ FINAL `bbf3e33bc…`. **HUMAN_FULL** remains valid for auditors reviewing a future human drill bundle; it is **not** the active published FINAL.

---

## Certification matrix reference

Embedded **102×ID** matrix: EVOL-X Final Closure plan §5 (5-A … 5-J).

- **M-09T** vs **M-09H** vs **M-09A** — M-09T MEASURED @ TECH ≠ human attribution; **M-09H** applies only to **HUMAN_FULL**; **M-09A** applies only to **AUTOMATION_FIRST_SCOPED** (machine diagnostic, OD-EVOL-13).
- **M-03R / M-03C / M-03E** — separate supplements; do not merge with **M-03** without `m03_causal_audit.v1.json`.

Deliverable B6: reconcile Part VII rows → PASS (SCOPED) or **EXACT BLOCKERS** with evidence SHA citations.

---

## Mode A — evidence integrity

**Supported CLI:** `verify-evidence-publish` with optional **`--technical`** only.

### AUTOMATION_FIRST_SCOPED (active FINAL)

**When:** @ **EVIDENCE_FINAL_PUBLICATION_SHA** `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` (or descendant doc-only commit whose **committed** P03 bundle bytes are unchanged — prefer exact FINAL SHA for audit).

```powershell
cd <repo>
git checkout bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1
uv sync --extra dev-ci --frozen
uv run python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish
```

**Expected (auditor G-05):**

- TES `3c4a641b3…` ancestor of evidence HEAD (not equal — publication commit ≠ TES).
- `MANIFEST.test_execution_sha` = TES; `MANIFEST.certification_profile` = `AUTOMATION_FIRST_SCOPED`.
- `evol_m09_automated_diagnostic_qualification.v1.json` (`qualification_policy_id=OD-EVOL-13`); **no** `evol_m09_human_drill.v2.json` in `artifact_paths`.
- `EVOL-M-09A` 3/3 MEASURED PASS; **`EVOL-M-09H` NOT_MEASURED**.
- Tool SLA ≤120 s/case via automated qualification cases (not human `attribution_seconds`).

**STOP:** claiming **M-09H HUMAN PASS** or human median ≤900 s from this bundle.

### HUMAN_FULL (historical / alternate path — not active FINAL)

```powershell
git checkout <EVIDENCE_FINAL_SHA_WITH_HUMAN_FULL_MANIFEST>
uv run python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish
```

Requires genuine `evol_m09_human_drill.v2.json`, worksheet digest, M-09H 3/3 MEASURED (OD-EVOL-08-M1).

### Technical disposition (TECH lineage only — not scoped FINAL closure)

```powershell
git checkout 1e175ff8f678d605aa841ae52362486a56ed006d
uv run python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish --technical
```

**STOP:** `--technical` @ **AUTOMATION_FIRST_SCOPED FINAL** HEAD for closure (M-09A MEASURED fails technical rule).

---

## Mode B — independent replay @ TES (G-06)

**When:** `HEAD` = **TES**; auditor-only replay (not implementer substitute for semantic review).

**Supported CLI:** `verify-test-execution --evidence-ref <git-ref> [--replay-junit path] [--certification]`

### AUTOMATION_FIRST_SCOPED (active)

```powershell
git checkout 3c4a641b3e48a13f437e830a0562e2a80b03aabb
uv sync --extra dev-ci --frozen
New-Item -ItemType Directory -Force -Path .tmp/session/evol-x-b6 | Out-Null
uv run pytest tests/qualification/evol_x -q --junitxml=.tmp/session/evol-x-b6/junit.xml
uv run python scripts/qualification/evol_x_p03_evidence.py verify-test-execution `
  --evidence-ref bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1 `
  --certification --replay-junit .tmp/session/evol-x-b6/junit.xml
```

**Critical:** `--evidence-ref` = **FINAL publication SHA** (bundle + manifest @ that commit), **not** TES and **not** historical TECH `1e175ff8…`.

**Expected:** **290** manifest node ids; archived outcomes @ FINAL match replay; exit 0.

### HUMAN_FULL (when auditing human FINAL bundle)

Same TES checkout; `--evidence-ref` = that bundle's **EVIDENCE_FINAL_PUBLICATION_SHA** (human drill artifacts present).

---

## FRZ-EVOL-07 — W7-03 checklist (Part X, G-08)

Unchanged: five-step audit; **Outcome A/B** required; **OD-EVOL-05b Option 1** — no production W4-04 rollback port claims.

**FRZ-EVOL-07 → PASS** only in **E4** after Outcome A + operator authorization — never E3 alone.

---

## CI evidence slot (W6-04 / G-01)

Parallel gate — not substitute for Mode B.

| Slot | Source |
|------|--------|
| WORKFLOW_DEF_SHA | `main` — see [EVOL_X_CI_QUALIFICATION_RECORD.md](./EVOL_X_CI_QUALIFICATION_RECORD.md) |
| Active dispatch TES | **`3c4a641b3e48a13f437e830a0562e2a80b03aabb`** |
| Run proof | [EVOL_X_CI_QUALIFICATION_RECORD.md](./EVOL_X_CI_QUALIFICATION_RECORD.md) — **active run** (post-reconciliation) |
| Historical | run `41327408027` @ TES `2b165516…` (288 nodes) — **superseded for active TES** |
| Procedure | [EVOL_X_CI_QUALIFICATION_PROCEDURE.md](./EVOL_X_CI_QUALIFICATION_PROCEDURE.md) |

**Operator authorization:** OD-EVOL-10 (+ OD-EVOL-13 for scoped FINAL semantics).

---

## B6 deliverable template

**Path (to create @ B6 SHA):**  
`docs/project/maintainers/qualification/evol_x/EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`

**Minimum sections:**

1. **Publication SHA** — `B6_ACCEPTANCE_PUBLICATION_SHA`
2. **Profile** — `AUTOMATION_FIRST_SCOPED` **or** `HUMAN_FULL`
3. **Evidence refs** — TES, EVIDENCE_FINAL, `independent_replay_sha` = TES
4. **Mode A** — command, exit status, evidence HEAD, profile observed in MANIFEST
5. **Mode B** — `--evidence-ref` FINAL SHA; JUnit replay match (**290** nodes @ current TES)
6. **Semantic review (G-14)** — M-09A cause identity / adversarial controls (scoped); **M-09H NOT_MEASURED** attestation
7. **W7-03 FRZ-07** — Outcome A or B
8. **CI** — C-CI-01 row @ TES `3c4a641b3…` or EXACT BLOCKER
9. **Disposition** — `INDEPENDENTLY ACCEPTED — AUTOMATION-FIRST SCOPED` **or** `EXACT BLOCKERS`
10. **Signatures** — named independent auditor only

---

## Findings register (template)

| Finding ID | Severity | Norm / Gate | Evidence SHA | Disposition | Operator action |
|------------|----------|-------------|--------------|-------------|-----------------|
| F-001 | | | | OPEN / SCOPED PASS / N/A | |

**Agent row forbidden:** «B6 PASS by agent» or FRZ promotion.

---

## G-xx blockers @ reconciliation time

| ID | Blocks B6 / closure |
|----|---------------------|
| G-01 | ~~W6-04 CI OPEN~~ **CLOSED** — run **`41337077334`** @ TES **`3c4a641b3…`** (290 nodes) |
| G-02 | OD-EVOL-11 name missing |
| G-03 | **HUMAN_FULL only:** M-09H drill — **N/A for AUTOMATION_FIRST_SCOPED** (M-09H must stay NOT_MEASURED) |
| G-04 | ~~EVIDENCE_FINAL empty~~ **CLOSED** @ `bbf3e33bc…` (scoped) |
| G-05 | Mode A final @ FINAL with correct profile |
| G-06 | Mode B independent @ TES + `--evidence-ref` FINAL |
| G-07 | No B6 record |
| G-08 | W7-03 not decided |
| G-09–G-10 | OD-EVOL-12 / OD-EVOLX-08 pending |
| G-13 | SHA chain — active TES/FINAL/CI recorded |

---

## E4 sequence (auditor read-only until B6 DONE)

1. B6 published (`INDEPENDENTLY ACCEPTED — AUTOMATION-FIRST SCOPED` or blockers)
2. OD-EVOL-12
3. FRZ-EVOL-01..10 (FRZ-07 only if W7-03 Outcome A)
4. OD-EVOLX-08
5. Roadmap §5 + FORMAL_CLOSURE_SHA
6. ENT-AUDIT-X handoff

**Agent:** E4 **BLOCKED** until B6.

---

*Prepared for independent audit — implementing agent does not sign B6.*
