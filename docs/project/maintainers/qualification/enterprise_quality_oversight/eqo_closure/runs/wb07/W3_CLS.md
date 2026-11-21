# EQO-W3-WB07 — W3 CLS disposition (wave 5)

**Task ID:** `EQO-W3-WB07-CLS`
**Disposition:** PARTIAL — not EQO_CLOSED.
**AUD session E (aggregate):** `FAIL` (AUD-2, `runs/wb07/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb07/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb07/AUD_SESSION_E.md` | Independent session E verdict |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close note (CLS_PRE)

Governance / IIAC-adjacent slice @ pin. Terminal EQO-W3-WB07-CLS deferred.

## Aggregate rationale (session E)

`FAIL` because three findings (A2-049, A4-034, A4-045) are confirmed by source at the pin and the slice gives them no coverage. A2-043 needs a delta replay. The EXE failure (HARNESS-01 inventory) is real but outside the four hypotheses.

## Required follow-up routes

1. A2-049: re-run the TRACE-X P6 bundle with `TRACE_X_P6_PASS1=1` and replace the directory-wide prefix rows with per-file classification, or record the accepted limitation.
2. A4-034: owner decision on the five retry-budget mechanisms.
3. A4-045: owner mapping of the artifact-reference vocabularies.
4. A2-043: replay the EBH-2E/2G/2H surfaces against their certified SHAs, or record the requalification.
5. HARNESS-01: classify the new higher-layer Nexus importers (separate defect, not decided here).

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
