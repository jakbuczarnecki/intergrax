# EQO master closure matrix (C01–C25)

| ID | Status | Note |
|----|--------|------|
| C01 | PARTIALLY_MET | 240 IDs enumerated; disposition incomplete |
| C02 | PARTIALLY_MET | WB01–12 AUD packets @ PRE_AUD/OPEN (wave-3); session E verdicts pending |
| C03 | PARTIALLY_MET | WB05 EXE @ 28c27cb (14 passed); DIS/AUD open |
| C04 | PARTIALLY_MET | C-04 replay BLOCKED @ TRAIN_PIN (wave-3 log retained) |
| C05 | PARTIALLY_MET | E-02 report @ DOC_PIN; register edit blocked |
| C06 | PARTIALLY_MET | 225 FRZ counted; 109 OPEN |
| C07 | NOT_MET | 2 open BLOCKER severity |
| C08 | PARTIALLY_MET | WB02 tenant pytest @ TRAIN_PIN (66 pass); WB03–12 open |
| C09 | NOT_MET | FCS/M3 not reached |
| C10 | NOT_MET | Train delta not fully replayed |
| C11 | PARTIALLY_MET | Dossier A pinned; train evidence partial |
| C12 | PARTIALLY_MET | PASS_SCOPED scope documented |
| C13 | NOT_VERIFIED |  |
| C14 | PARTIALLY_MET | KR-01..28 classified; none RESOLVED at M3 |
| C15 | NOT_VERIFIED |  |
| C16 | NOT_VERIFIED | #680 open |
| C17 | NOT_VERIFIED |  |
| C18 | PARTIALLY_MET | owners proposed not confirmed |
| C19 | PARTIALLY_MET | A audit yes; IIAC IN_FORCE no |
| C20 | PARTIALLY_MET | A reproducible; B/C/D not |
| C21 | PARTIALLY_MET | W2 verify @ 28c27cb: #642–#679 merged; #680 not on train |
| C22 | NOT_VERIFIED |  |
| C23 | PARTIALLY_MET | WBS maps to train tickets |
| C24 | NOT_MET |  |
| C25 | NOT_MET |  |

## W3 AUD pre-session (wave-3)

**Land:** `8631b1e8459e096b74d1c80247d3e432b44fff30` (PR #690). **Packets:** `runs/wbNN/AUD_PACKET.md` — status PRE_AUD or OPEN only; verdict reserved for session E.

| WB | EXE exit | AUD packet | Session E |
|----|---------:|------------|-----------|
| WB01 | 1 | `OPEN` | queued (OPEN — EXE/DIS caveats) |
| WB02 | 0 | `PRE_AUD` | queued |
| WB03 | 0 | `PRE_AUD` | queued |
| WB04 | 1 | `OPEN` | queued (OPEN — EXE/DIS caveats) |
| WB05 | 0 | `PRE_AUD` | queued |
| WB06 | 4 | `OPEN` | queued (OPEN — EXE/DIS caveats) |
| WB07 | 1 | `OPEN` | queued (OPEN — EXE/DIS caveats) |
| WB08 | — | `OPEN` | queued (OPEN — EXE/DIS caveats) |
| WB09 | 1 | `OPEN` | queued (OPEN — EXE/DIS caveats) |
| WB10 | 1 | `OPEN` | queued (OPEN — EXE/DIS caveats) |
| WB11 | 1 | `OPEN` | queued (OPEN — EXE/DIS caveats) |
| WB12 | 0 | `PRE_AUD` | queued |

