# Pulpit sesji EQO (F1 view)

**Branch:** `eqo/w1-f1-view`  
**Stan terminalny:** EQO **nie** zamknięte; brak `EQO_CLOSED` i brak certyfikacji `IIAC PASS`.

## Piny SHA

| Ref | SHA |
|-----|-----|
| `origin/development` | `a0598ba3e4f4f903eb4da3db8ea012d5cbf71225` |
| `origin/arch-freeze/rw-train` | `8e146e21cd8c63a47d621edc05d5f0036ba0c9ee` |

## Liczniki ścieżek (sekcja 2, first-match)

`OWNER_DECISION_REQUIRED` / `UNRESOLVED_EXACT_BLOCKER` / `DOCUMENTATION_CORRECTION_REQUIRED` / `FORMAL_REQUALIFICATION_REQUIRED` / `EXISTING_IMPLEMENTATION_REQUIRES_REPLAY` / `NEW_IMPLEMENTATION_REQUIRED` / `EXTERNAL_ARCH_FREEZE_DEPENDENCY`:

**313 / 310 / 167 / 165 / 40 / 33 / 1**

## `FORMAL_REQUALIFICATION_REQUIRED` — waluta

| Waluta | Liczba |
|--------|-------:|
| `STALE` | 165 |
| `CURRENT` | 0 |

Artefakt: `EQO_REQUAL_CURRENCY.json` · commit `a0e44dd7d`

## `DOCUMENTATION_CORRECTION_REQUIRED` — indeks

| Pole | Wartość |
|------|--------:|
| Wiersze | 167 |
| Właściciele (`owner_ref`) | 36 |

Artefakt: `EQO_DOC_PACKAGE_INDEX.json` · commit `531b4e323`

## Wyniki pakietów (zweryfikowane lokalnie, bez push)

| Pakiet | Commit (prefiks) | Wynik |
|--------|------------------|-------|
| `EQO-PKG-A4-006` | `960205aaa` | 3 passed |
| `EQO-PKG-A4-027` | `06604af53` | 1 passed · `T-IGX-055-B` |
| `EQO-PKG-A4-026` (train replay) | — | 12 passed |
| `EQO-PKG-A4-017` (train replay) | — | 11 passed |
| FC (train replay) | — | 59 passed |
| `EQO-PKG-A4-034` | `34a0af435` | `BLOCKED` |
| `EQO-PKG-A6-006` | `c5b9648b4` | `BLOCKED` |
| `EQO-PKG-A6-001` | — | open evidence gap |

## Poza tą sesją

Nie wykonywane w tej sesji EQO: `git push`, `FCS`, `PT-Q`, `M3`, `RB-2`, certyfikacja `IIAC`.

## Uwaga push (atrybucja)

Przed `git push` usuń `Co-authored-by: Cursor` z commitów, które go zawierają.