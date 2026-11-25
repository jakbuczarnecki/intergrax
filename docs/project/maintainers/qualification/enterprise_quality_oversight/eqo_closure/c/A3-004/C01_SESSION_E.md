# EQO-C-01 — Session E evidence (HIST-85 / A3-004)

**Task:** `EQO-C-01` · **Finding:** EQOA-A3-004  
**Source SHA:** `8bdb0038d7a8fe506b5490a0ec2acad227e6b938`  
**Dossier anchor:** `A_BASELINE` `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`  
**Reviewer:** implementer session E (independent AUD pending)

## Sample binding (30/33)

Mechanical sample: `campaign_a/A3/hist85_sample.md` + `campaign_a/shards/A3/hist85.jsonl` (33 rows).

| Check | Result |
|-------|--------|
| Row count in `hist85.jsonl` for sample IDs | 33 |
| Documented agree / disagree | 30 agree / 3 disagree |
| Narrative promotion to 33/33 | **refused** |

The **30/33 rate binds** to the dossier at `A_BASELINE`. Live `development` HEAD does not re-score the sample; it validates that evidence pointers for disputed rows still resolve.

## Disputed rows @ source SHA

| FRZ ID | Dossier disagree | Paths exist @ `8bdb0038` | Session E disposition |
|--------|------------------|---------------------------|------------------------|
| FRZ-BND-05 | yes | `tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py` | **OPEN** — gate scope dispute unchanged (import direction vs composition) |
| FRZ-PRD-07 | yes | `intergrax/skills/execution_pinning_persistence.py` | **OPEN** — strict-path replay not demonstrated @ HEAD |
| FRZ-BND-01 | yes | `tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py` | **DEFERRED** — independent gate replay required (not dossier re-read) |

## Verdict

`ACCEPTED_SCOPED_LIMITATION` — 30/33 binding stands; three rows remain blockers for freeze promotion.

**Ref:** `governance/tickets/T-IGX-046-F.md`