# T-IGX-051-G — T-AF2-RW-TYP (frozen contract typ ratchet, OD-AF-11)

## Scope

Train node **T-AF2-RW-TYP**, ticket **T-IGX-051-G** per `MEP-AF2_v2.1.md` W3 table (`SCOPE`, `PT-WA`, `T-AF2-TYP-MEASURE`, `OD-AF-11`; PROD).

Implement **OD-AF-11** (DECIDED): **ratchet on frozen contract surfaces** `intergrax/contracts/**/*.py` (446 modules @ pin per `typ_measure.md`), not a mass deletion campaign on `Any` or `object` annotations.

**Operator decision (quote, `ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md` row OD-AF-11):** *Ratchet on the frozen contract surfaces.* TYP-01/03 remediated per DP-36. TYP-02/04/05 get a ratchet gate that fails on any new `Any`/`object` in a frozen contract surface, plus a reduction target for TYP-05. Owner: **T-AF2-RW-TYP (T-IGX-051-G)**.

**MEP acceptance (row T-AF2-RW-TYP):** *ratchet gate fails on a new `Any` in a frozen contract* (plan extends to `object` per OD-AF-11).

**Measurement @ pin** (`typ_measure.md`, cited): Any **164** / object **352** (mechanical scan); OD-AF-11 packet baseline **138** / **357**; reflection getattr **11**, setattr **0**, hasattr **3**; TYP-05 not run; tracked `typ_ratchet_baseline.v1.json` **0** paths. The operator chooses the authoritative Any/object baseline before implementation (plan §4).

Canonical plan: `governance/tickets/plans/PLAN_T-IGX-051-G.md`.

## Plan digest

Plan digest: d4e9005e793e3c429839580bc305400fc2d8dd299c46d2e9336b0ef37c7b7e3d

## Approved

Approved: operator-delegate @ 2026-11-24T18:15:00+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/continuation/OPERATOR_DECISIONS_AUTONOMY.md@0889a64bf93003548b73e3e282b09e3bc0be20b8
Delegate session: local_c387c27a-2faa-432c-baca-9bc2a29aa7d8

## Test contract

After plan approval:

- **Ratchet negative:** a deliberate new `Any` or `object` annotation in a file under `intergrax/contracts/**/*.py` causes the frozen-contract ratchet gate / pytest to **FAIL** (`NT-RW-TYP-RATCHET-ANY`, `NT-RW-TYP-RATCHET-OBJECT`).
- **Steady state @ baseline:** with no new annotations, ratchet passes at the operator-approved baseline counts (see plan §4 — not reconciled in this prep pack).
- **TYP-05:** when `measure_typ05.py` is promoted to a tracked path, gate enforces no new string-key lookup tables beyond baseline (5 @ train per `T-IGX-043-H` citation in `typ_measure.md`) unless plan amendment lists an allowed delta.
- **Reflection metrics:** no increase in getattr/setattr/hasattr counts on frozen surface (OD-AF-11 TYP-04).
- Serialize after the last concurrent editor of `intergrax/contracts/**` (RW-TYP merge rule in sibling plans).

Commands and paths are only in the promoted plan §7; this ticket does not run pytest until implementation starts.

## Out of scope

- Repo-wide `intergrax/` typing cleanup (3348 `Any` outside freeze — OD-AF-11 explicitly out of freeze scope).
- Mass removal of existing `Any`/`object` to chase mechanical vs packet count drift without operator baseline ruling.
- RW-GATES promotion / NOT_COMPOSED work (`T-IGX-051-F`).
- Dependency upgrades or `uv.lock` changes (`T-IGX-051-H` disposition: no upgrade this wave).
- FRZ promotion, G6 PASS, merge authority.

## Lifecycle state

in_progress

## Implementer session

Implementer session: local_e54e0892-7d8a-4304-98c7-45f1bc8ebb1c
