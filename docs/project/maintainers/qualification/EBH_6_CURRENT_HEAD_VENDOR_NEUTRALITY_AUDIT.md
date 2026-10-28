# EBH-6 — Current-HEAD Vendor Neutrality Audit (rollup skeleton)

| Field | Value |
| --- | --- |
| **Purpose** | Single current-HEAD rollup per VBN spec §2.7 |
| **HEAD_SHA (git evidence)** | `50d3b7785c7e13dbc6708143e7484a2eeea5eaca` |
| **IMPLEMENTATION_SHA** | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` |
| **CONFIG-X historical PASS** | `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` |
| **Disposition** | **READY FOR INDEPENDENT REVIEW** — not ARCH-FREEZE entry |

## Rollup summary

| Component | Status @ HEAD |
| --- | --- |
| EBH-5 inventory replay | **unclassified=0** (mechanical) |
| EBH-5 replaceability matrix | **51 NYQ** — explicit limitation |
| CORE_HARDCODING_FINDINGS_REGISTER | **57 findings**, prohibited unresolved **0** |
| SQLite annex diff | **Closed-world match** (38 paths) |
| RPL-G @ HEAD | Replay via EBH-5 gates (subset) |

## CONFIG-X reconciliation

Historical FRZ-CFG PASS preserved @ fac9a43f…; EBH-6 records current-HEAD structural obligations via EBH-5 VBN inventory + CORE_HARDCODING_FINDINGS_REGISTER — not PASS upgrade.

| Metric | Value |
| --- | --- |
| CONFIG-X concerns | **54** |
| Mapped to VBN lineage @ HEAD | **54/54** |
| Drift mechanism | [`tests/qualification/ebh_6/_config_x_reconciliation.py`](../../../tests/qualification/ebh_6/_config_x_reconciliation.py) |
| EBH-5 reconciliation reuse | [`tests/qualification/ebh_5/_config_x_reconciliation.py`](../../../tests/qualification/ebh_5/_config_x_reconciliation.py) |

### CONFIG-X vs current HEAD (explicit)

- **Does not** revoke FRZ-CFG PASS @ `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb`.
- **Does** require current-HEAD structural/reachability evidence via EBH-6 findings + EBH-5 inventory integrity.
- Active CONFIG-X blocker paths @ HEAD: expected **empty** (RPL-G parity).

## Acceptance criteria (§2.7) — candidate status

| Criterion | EBH-6 batch |
| --- | --- |
| `prohibited_unresolved` in frozen scope | **0** (mechanical) |
| `promised_unqualified` global closure | **Not claimed** — 51 NYQ deferred |
| CONFIG-X baseline preserved | **Yes** |
| Independent audit | **PENDING (L9)** |

## Scope exclusions

- **51** EBH-5 matrix rows **NOT YET QUALIFIED**
- **PROD-Q** live provider slice
- **DUP-X** duplicate/shadow certification
- Global **FRZ-*** PASS promotion

## Verification

```powershell
uv run pytest tests/qualification/ebh_6/ -q
```

Log: `.tmp/proof/ebh-6/pytest.log`
