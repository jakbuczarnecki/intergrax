# PROD-Q-R1-R7-Q-OD-R2-03 — SINGLE-NODE PRODUCTION PROFILE V1

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** |
| **Task** | `PROD-Q-R1-R7-Q-OD-R2-03-PROFILE-V1-LIFECYCLE-AND-ENTRYPOINT-SAFETY` |
| **Parent** | **PROD-Q-R1-R7-Q** — **IN PROGRESS** / **NOT QUALIFIED** |
| **START_HEAD** | `da0fbb965ed3241cfbf052bec30fe9f27afed13c` |
| **FINAL_COMMIT / AUDITED_HEAD** | *(set at push — independent audit)* |
| **EXECUTION_SHA** | *(pytest @ FINAL_COMMIT)* |
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
| Profile v1 → strict startup seam | GCA host | `single_node_profile_v1_strict_startup.py` | `open_gca_strict_production_serving_session_from_single_node_production_profile_v1` |
| CONFIGURED admission gate | GCA host | `single_node_profile_v1_admission_gate.py` | fail-closed env validation (not TARGET probe) |
| Cross-process writer (claim 3) | Applications shared + OS lock | `NativeFileLock` under `DATA_DIR` | `acquire_cross_process_writer_admission_for_profile_v1` |
| Linux systemd (infra) | Operator / deployment | GCA `deploy/linux/single-node-production-profile-v1/` | unit + env + entrypoint template |

No new semantic or composition owners. No second deployment framework.

### 1.1 Lifecycle / lock ownership (post lifecycle correction)

| Phase | Cross-process writer lock | Strict serving session |
|-------|---------------------------|-------------------------|
| CONFIGURED admission (gate) | Not acquired (`acquire_cross_process_writer=False`) | Not opened |
| Profile v1 strict startup | Acquired in `load_validate_and_admit_*` | Open via `open_gca_strict_production_serving_session` |
| GCA open failure | Released in startup seam `except` | Not returned |
| Successful open | Bound to session via `bind_process_scoped_release_on_close(writer_hold.release)` | Returned; D4 ASGI uses existing factory |
| ASGI shutdown / `session.close()` | Released in session `close()` finally | Existing D4 drain + lifecycle close |
| systemd stop | Operator process exit → session close path | Unit `TimeoutStopSec` |

**Invariant:** `writer lock held ⇔ active profile-v1 strict session owns the hold until `StrictProductionServingSession.close()`.

---

## 2. Profile v1 artifacts

| Artifact | Path |
|----------|------|
| Typed configuration | `intergrax/applications/_shared/single_node_production_profile_v1.py` |
| systemd unit | `applications/governed_contractor_application/deploy/linux/single-node-production-profile-v1/intergrax-gca-strict-production.service` |
| Env example | `.../intergrax-gca-strict-production.env.example` |
| Target probe (read-only, **informational**) | `.../target-storage-probe.sh` |
| Entrypoint template | `.../intergrax-gca-strict-production-entrypoint.sh` |
| REPO-TESTED gates | `tests/qualification/prod_q/test_prod_q_r1_r7_q_od_r2_03_single_node_production_profile_v1_gate.py` |

Environment prefix: `INTERGRAX_SINGLE_NODE_PRODUCTION_V1_*` (see module docstring).

---

## 3. PV1-H01…10 admission-hardening matrix

| Gate | Claim | REPO-TESTED |
|------|-------|-------------|
| **PV1-H01** | Profile reaches sanctioned strict startup | `test_pv1_h01_profile_reaches_sanctioned_strict_startup` |
| **PV1-H02** | Missing configuration fails closed | `test_pv1_h02_*`, `test_od_r2_03_profile_missing_data_dir_fail_closed` |
| **PV1-H03** | No alternate composition owner | `test_pv1_h03_no_alternate_startup_composition_authority` |
| **PV1-H04** | In-process slot vs cross-process writer lock separated | `test_pv1_h04_in_process_*`, `test_pv1_h04_cross_process_writer_lock_*` |
| **PV1-H05** | WAL/FULL on **owned** backend connections | `test_pv1_h05_owned_connection_sqlite_wal_full_*` |
| **PV1-H06** | Ephemeral / invalid storage rejected | `test_pv1_h06_*`, `test_od_r2_03_profile_rejects_missing_data_directory` |
| **PV1-H07** | systemd + entrypoint contract | `test_pv1_h07_systemd_and_entrypoint_agree` |
| **PV1-H08** | Probe informational; admission gate fail-closed | `test_pv1_h08_probe_informational_admission_gate_fail_closed` |
| **PV1-H09** | Failed startup does not widen provider surface | `test_pv1_h09_failed_startup_does_not_expand_provider_surface` |
| **PV1-H10** | TARGET / B6 explicitly unproven | `test_pv1_h10_target_production_qualification_explicitly_unproven` |

**F01–F06 disposition (consolidated):** startup seam (**F01**), shipped entrypoint + required env (**F02**), cross-process lock distinct from in-process slot (**F03**), owned-connection pragma evidence (**F04**), probe labeled informational + admission gate (**F05**), required `EnvironmentFile` + gate (**F06**). **F07** remains operator TARGET.

### 3.1 LIFE-01…08 matrix (lifecycle & entrypoint safety)

| Gate | Requirement | REPO-TESTED |
|------|-------------|-------------|
| **LIFE-01** | Startup failure after writer lock releases hold | `test_life_01_startup_failure_after_writer_lock_releases_hold` |
| **LIFE-02** | Two processes cannot hold same profile lock | `test_life_02_cross_process_writer_lock_excludes_second_holder` |
| **LIFE-03** | Lock held while strict session lifecycle active | `test_life_03_writer_lock_held_for_active_strict_session_lifecycle` |
| **LIFE-04** | Normal shutdown releases lock | `test_life_04_session_close_releases_writer_lock_for_subsequent_startup` |
| **LIFE-05** | Failed startup does not admit traffic / widen surface | `test_life_05_failed_startup_does_not_return_serving_session` |
| **LIFE-06** | Sanctioned D4 GCA ASGI path preserved | `test_life_06_sanctioned_d4_gca_asgi_factory_preserved` |
| **LIFE-07** | Missing `INTERGRAX_GCA_STRICT_SERVE_EXEC` fails closed | `test_life_07_missing_operator_serve_configuration_fails_closed` |
| **LIFE-08** | TARGET-VERIFIED / operator acceptance OPEN | `test_life_08_target_verified_and_operator_acceptance_remain_open` |

**Pre-audit F01–F03 disposition:** F01 startup failure lock cleanup — **addressed** in startup seam + idempotent `release()`; F02 successful session lock lifetime — **addressed** via session attach + `close()`; F03 operator serve command — **deployment prerequisite** (`INTERGRAX_GCA_STRICT_SERVE_EXEC` required; entrypoint does not substitute D4 wiring).

---

## 4. PROFILE-01…10 matrix

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

## 5. GCA/PIS restart proof mapping

| Scenario | REPO-TESTED module |
|----------|-------------------|
| Initial CREATE persistence | `test_prod_q_r1_r7_q_gca_sqlite_durable_recovery.py` |
| Dual SQLite Q2 ordering / redispatch | `test_prod_q_r1_r7_q_impl_01_dual_sqlite_restart_qualification_q2.py` |
| Fresh process + abrupt termination | `test_prod_q_r1_r7_q_impl_01_r2_od_r2_03_restart_safety_operator_admission.py` |
| Cross-tenant denial | same OD-R2-03 restart module |
| Package D GCA owned-store cleanup | Package D qualification @ historical SHAs |

---

## 6. Recovery qualification (target procedure)

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

## 7. Remaining VM / operator evidence

- Provisioned Linux VM with ext4 data volume **not yet assumed**.
- Install entrypoint template to `/opt/intergrax/bin/`; set `INTERGRAX_GCA_STRICT_SERVE_EXEC` to operator Package B serve wiring calling `open_gca_strict_production_serving_session_from_single_node_production_profile_v1` + D4 uvicorn factory.
- Package B operator mutation policy + live credentials (**OP-01…04** still OPEN).
- TARGET-VERIFIED ext4 mount identity and backup/restore drill.

---

## 8. Execution record

```text
uv run pytest -p no:xdist \
  tests/qualification/prod_q/test_prod_q_r1_r7_q_od_r2_03_single_node_production_profile_v1_gate.py \
  -q
```

Log: `.tmp/session/od-r2-03-single-node-profile-v1/pytest.log`

---

## 9. ECAP summary

| Item | Result |
|------|--------|
| A1–A8 | No new authority; writer lock lifetime bound via `bind_process_scoped_release_on_close` on `StrictProductionServingSession.close()` |
| B1–B5 | Layer boundaries preserved; typed settings at applications shared |
| B6 | **Deferred** — operator acceptance OPEN |

---

## 10. Applicable FRZ (no promotion)

| ID | Applicable | Note |
|----|------------|------|
| FRZ-REL-03 | partial | Lab restart / SQLite only |
| FRZ-PRD-03/04/05/08/12 | partial | Strict paths + D4 wiring; VM not qualified |
| FRZ-SEC-04 | partial | Service hardening template only |
| FRZ-TEN-* | partial | Cross-tenant tests cited |
| FRZ-OWN / FRZ-CTR / FRZ-BND / FRZ-TYP / FRZ-REG | partial | Inventory §1 |

---

## 11. Unresolved findings

| Finding | Class |
|---------|-------|
| No TARGET-VERIFIED VM ext4 / systemd drill | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** |
| Package B operator production policy | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** |
| Integrated PROD-Q / R7-Q closure | **IN-SCOPE BLOCKER** (parent) |
| `INTERGRAX_GCA_STRICT_SERVE_EXEC` operator serve wiring on VM | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** |
| F02 full production E2E on TARGET VM | **TRACKED FREEZE DEBT** |

**Roadmap disposition:** Continue **PROD-Q-R1-R7-Q** — profile v1 artifacts **READY FOR AUDIT**; parent remains **NOT QUALIFIED**.

---

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
