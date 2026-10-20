# PROD-Q-R1-R7-Q-OD-R2-03 — SINGLE-NODE PRODUCTION PROFILE V1

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** |
| **Task** | `PROD-Q-R1-R7-Q-OD-R2-03-SINGLE-NODE-PRODUCTION-PROFILE-V1` |
| **Parent** | **PROD-Q-R1-R7-Q** — **IN PROGRESS** / **NOT QUALIFIED** |
| **START_HEAD** | `a7122b6e019db9123a3dccad2816389f735fa855` |
| **FINAL_COMMIT / AUDITED_HEAD** | `8b7edb0e7d0217d24e964577a7ae40758f7258e4` |
| **EXECUTION_SHA** | `8b7edb0e7d0217d24e964577a7ae40758f7258e4` — **8/8** passed |
| **Does not close** | OD-R2-03 · IMPL-01-R2 · R7-Q · PROD-Q · any global **FRZ-*** PASS |

Operator authorized **single-node Linux VM profile v1** (exactly-one writer, ext4 SSD, SQLite GCA/PIS WAL+FULL, systemd, least-privilege identity). This dossier is repository + lab evidence only.

---

## 1. Closed-world inventory

| Concern | Semantic owner | Composition owner | Contract / surface |
|---------|----------------|-------------------|-------------------|
| Strict production storage paths | Applications shared | `open_strict_production_serving_session` | `StrictProductionProcessStorage` |
| Profile v1 typed paths | Applications shared | Operator env → profile loader | `single_node_production_profile_v1.py` |
| GCA host SQLite | GCA application host | `open_sqlite_governed_external_work_host_store_bundle_at` | `SQLiteGovernedExternalWorkHostStoreBackend` |
| PIS SQLite | Runtime enterprise reliability | `open_provider_invocation_store_at` | `SQLiteProviderInvocationStore` |
| Single active lifecycle / writer (in-process) | Applications shared | `open_production_process_lifecycle` | `_active_production_process_lifecycle` slot |
| Strict GCA ASGI | GCA host | D4 launcher | `open_gca_strict_production_serving_session` |
| Linux systemd (infra) | Operator / deployment | GCA `deploy/linux/single-node-production-profile-v1/` | unit + env example |

No new semantic or composition owners. No second deployment framework.

---

## 2. Profile v1 artifacts

| Artifact | Path |
|----------|------|
| Typed configuration | `intergrax/applications/_shared/single_node_production_profile_v1.py` |
| systemd unit | `applications/governed_contractor_application/deploy/linux/single-node-production-profile-v1/intergrax-gca-strict-production.service` |
| Env example | `.../intergrax-gca-strict-production.env.example` |
| Target probe (read-only) | `.../target-storage-probe.sh` |
| REPO-TESTED gates | `tests/qualification/prod_q/test_prod_q_r1_r7_q_od_r2_03_single_node_production_profile_v1_gate.py` |

Environment prefix: `INTERGRAX_SINGLE_NODE_PRODUCTION_V1_*` (see module docstring).

---

## 3. PROFILE-01…10 matrix

| Gate | Requirement | CONFIGURED | REPO-TESTED | TARGET-VERIFIED | OPERATOR-ACCEPTED |
|------|-------------|------------|-------------|-----------------|-------------------|
| **PROFILE-01** | Authorized Linux VM identity | Env `SERVICE_RUNTIME_USER` + unit `User=` | `test_od_r2_03_profile_01_*`, `test_od_r2_03_profile_05_*` | VM hostname / asset record | **OPEN** |
| **PROFILE-02** | Durable ext4 + stable paths | `DATA_DIR` + path co-location | Ephemeral rejection; probe off-linux | `target-storage-probe.sh` on VM | **OPEN** |
| **PROFILE-03** | Single-writer enforcement | `ENFORCE_SINGLE_ACTIVE_WRITER` | `test_od_r2_03_profile_03_*` (in-process slot) | One systemd unit + no second instance | **OPEN** |
| **PROFILE-04** | SQLite WAL / synchronous FULL | Pragma helpers | `test_od_r2_03_profile_04_*` | Probe script pragma section | **OPEN** |
| **PROFILE-05** | Service lifecycle / cleanup | Unit `TimeoutStopSec`, `Restart` | `test_od_r2_03_profile_05_*` | `systemctl` stop/start journal | **OPEN** |
| **PROFILE-06** | GCA/PIS restart persistence | Paths bound to `StrictProductionProcessStorage` | Mapped tests below | Fresh-process on VM | **OPEN** |
| **PROFILE-07** | No unsafe redispatch | Unchanged F02 / Q2 semantics | Q2 + OD-R2-03 restart module | Operator replay on VM | **OPEN** |
| **PROFILE-08** | Cross-tenant fail-closed | Unchanged port gates | `test_prod_q_r1_r7_q_impl_01_r2_od_r2_03_restart_safety_operator_admission.py` | — | **OPEN** |
| **PROFILE-09** | Backup + verified restore | Documented procedure §5 | Lab SQLite file copy tests (DUR/Q2) | Operator backup/restore drill | **OPEN** |
| **PROFILE-10** | Exact-SHA evidence | This dossier + git ancestry gate | pytest @ `FINAL_COMMIT` | Operator sign-off | **OPEN** |

---

## 4. GCA/PIS restart proof mapping

| Scenario | REPO-TESTED module |
|----------|-------------------|
| Initial CREATE persistence | `test_prod_q_r1_r7_q_gca_sqlite_durable_recovery.py` |
| Dual SQLite Q2 ordering / redispatch | `test_prod_q_r1_r7_q_impl_01_dual_sqlite_restart_qualification_q2.py` |
| Fresh process + abrupt termination | `test_prod_q_r1_r7_q_impl_01_r2_od_r2_03_restart_safety_operator_admission.py` |
| Cross-tenant denial | same OD-R2-03 restart module |
| Package D GCA owned-store cleanup | Package D qualification @ historical SHAs |

---

## 5. Recovery qualification (target procedure)

Operator executes on **authorized VM** with **synthetic data** and **fake provider** only:

1. Deploy profile v1 env + entrypoint (D4 strict session, no `main.run()`).
2. CREATE via strict external-work port; record invocation ids.
3. Graceful `systemctl restart`; verify no duplicate CREATE.
4. `kill -9` bounded drill; verify fail-closed / recovery per Q2.
5. Corrupt/missing F02 — expect fail-closed (Q2).
6. Cross-tenant read — expect denial (OD-R2-03 restart tests).
7. Stop service; filesystem backup of `DATA_DIR`; restore to disposable path; verify open + tenant isolation.
8. Archive probe output + pytest log + `git rev-parse HEAD`.

---

## 6. Remaining VM / operator evidence

- Provisioned Linux VM with ext4 data volume **not yet assumed**.
- `intergrax-gca-strict-production-entrypoint` script (operator-owned) must open `open_gca_strict_production_serving_session` then uvicorn factory.
- Package B operator mutation policy + live credentials (**OP-01…04** still OPEN).
- TARGET-VERIFIED ext4 mount identity and backup/restore drill.

---

## 7. Execution record

```text
uv run pytest -p no:xdist \
  tests/qualification/prod_q/test_prod_q_r1_r7_q_od_r2_03_single_node_production_profile_v1_gate.py \
  -q
```

Log: `.tmp/session/od-r2-03-single-node-profile-v1/pytest.log`

---

## 8. ECAP summary

| Item | Result |
|------|--------|
| A1–A8 | No new authority; profile is configuration + infra templates only |
| B1–B5 | Layer boundaries preserved; typed settings at applications shared |
| B6 | **Deferred** — operator acceptance OPEN |

---

## 9. Applicable FRZ (no promotion)

| ID | Applicable | Note |
|----|------------|------|
| FRZ-REL-03 | partial | Lab restart / SQLite only |
| FRZ-PRD-03/04/05/08/12 | partial | Strict paths + D4 wiring; VM not qualified |
| FRZ-SEC-04 | partial | Service hardening template only |
| FRZ-TEN-* | partial | Cross-tenant tests cited |
| FRZ-OWN / FRZ-CTR / FRZ-BND / FRZ-TYP / FRZ-REG | partial | Inventory §1 |

---

## 10. Unresolved findings

| Finding | Class |
|---------|-------|
| No TARGET-VERIFIED VM ext4 / systemd drill | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** |
| Package B operator production policy | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** |
| Integrated PROD-Q / R7-Q closure | **IN-SCOPE BLOCKER** (parent) |
| F02 full production E2E | **TRACKED FREEZE DEBT** |

**Roadmap disposition:** Continue **PROD-Q-R1-R7-Q** — profile v1 artifacts **READY FOR AUDIT**; parent remains **NOT QUALIFIED**.

---

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
