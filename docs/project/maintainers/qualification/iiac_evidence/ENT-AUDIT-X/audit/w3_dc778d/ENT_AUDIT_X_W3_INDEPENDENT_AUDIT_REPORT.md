# ENT-AUDIT-X — W3 niezależny audyt jakości (IIAC Session D, publication Batch A)

**Rola:** `independent-quality-auditor`  
**Polityka:** `IIAC-2026-11-05.1` (IN_FORCE) · profil `iiac-v1`  
**Ticket:** `governance/tickets/T-IGX-024-B.md`  
**Poziom assurance:** `process_enforced`  
**Data audytu (UTC):** 2026-11-11  
**Worktree audytu:** `D:\Projekty\_wt\intergrax\w3-audit-dc778d` @ `dc778d125fd7e7205edd30fbbf144ca12320b0ee`  
**Publikacja W3 Batch A:** `D:\Projekty\_wt\intergrax\w4-closure-dc778d` @ `ent/w3-audit-pub`

**Adopcja dowodu:** werdykt i ustalenia operacyjne pochodzą z dossier sesji `c8e4a91f-2b6d-4f7e-a3c1-9d5e7f2b4a68` (`.tmp/session/ent-w2-audit/ENT_AUDIT_X_W3_INDEPENDENT_AUDIT_DOSSIER.md`); niniejsza publikacja w `iiac_evidence/` jest wykonana w osobnej sesji audytora (manifest `auditor_session_id`) bez zmian ścieżek produktowych.

## 0. Werdykt

**PASS_SCOPED**

Przy `audit_execution_sha=dc778d125fd7e7205edd30fbbf144ca12320b0ee` i `TEST_EXECUTION_SHA=32a9633440007eb4e8f00a825bbf2e399192b346` udowodniono genealogię git, integralność B5 FULL @ EVID `481b2cc2…`, certyfikujące FULL/FREEZE 85/85/0 bez bootstrap-candidate, zgodność digest baseline `480550d80…`, IIAC IN_FORCE oraz opublikowane drzewa dowodowe W2. Ograniczenia F-W3-001..006 są jawne i nie unieważniają opublikowanych bajtów @ TES′.

**Nie:** PASS bez SCOPED (FRZ-AUD-10 compensating, FREEZE B5 naming, brak pełnego Mode C replay).

## 1. SHA lock

| Rola | SHA |
| --- | --- |
| `audit_execution_sha` | `dc778d125fd7e7205edd30fbbf144ca12320b0ee` |
| `audited_implementation_sha` / `test_execution_sha` | `32a9633440007eb4e8f00a825bbf2e399192b346` |
| `evidence_publication_sha` (B5 FULL) | `481b2cc2c10a407fd2b93cf9c69714e8894aeccf` |
| W2 normal FREEZE publication | `bc4b5f61666a8fea638064f50809fff2089b1b8b` |
| `ACCEPTED_BASELINE_SHA` | `480550d80b2515fc8cbd2a0ba6f79d1ee1c44541` |

Delta produktowa TES → HEAD audytu: **pusta** w `intergrax/`, `agents/`, `applications/`.

## 2. Mode A — B5

| Drzewo | `verify_b5_publication_tree` |
| --- | --- |
| `qualification_at_32a963344…` | **0 błędów** |
| `freeze_normal_at_32a963344…` | **1 błąd:** prefix nazwy katalogu (bajty spójne; F-W3-001) |

## 3. L3 certyfikujące (opublikowane)

| Tryb | checks | baseline |
| --- | --- | --- |
| FULL @ TES | 85/85/0 | `480550d80…` |
| FREEZE normal @ TES | 85/85/0, bez bootstrap | `480550d80…` |

## 4. FRZ-AUD-01..10 (audytor @ TES′; bez promocji)

Wiersze ocenione PASS / PASS COMPENSATING (FRZ-AUD-10) w dossier; **promocja wyłącznie W4 certifier**.

## 5. Macierz C i Population E

Matrix C: 69 QUALIFIED + 6 operator-trust QUALIFIED (auditor replay); GAP overlay D-2 vs dowód techniczny — F-W3-002.  
Population E: **PASS** — 85 semantycznych oracle’ów w `enterprise_audit_outcomes.json`.

## 6. Ustalenia

Patrz `findings.json` (F-W3-001..006).

## 7. Ograniczenia zakresu

1. Brak pełnego Mode C (ADV-01..31 + pełna seria negatywów) w sesji dossier.  
2. Brak osobistego ponownego CLI FULL/FREEZE @ TES w sesji dossier (F-W3-006).  
3. Niniejsza publikacja **nie** ustawia `B6_ACCEPTANCE_PUBLICATION_SHA`, `frz_aud_status=CLOSED`, `OD-ENT-08`.  
4. Werdykt **PASS_SCOPED**, nie kryptograficzny.

## 8. Jawne non-actions

Brak CLOSED headline · brak promocji FRZ-AUD · brak certyfikacji W4 w tej publikacji.
