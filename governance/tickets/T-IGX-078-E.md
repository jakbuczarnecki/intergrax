# T-IGX-078-E - FRZ-PLG-08 fact check under DP-30 and DP-37

## Owner

AF-CLOSE, lane K1-D. Opened by GA.2R (T-IGX-078-B) as an ownership ticket.

## Scope

FRZ-PLG-08 is OPEN in the FRZ acceptance matrix and had no owning ticket with a current verdict path. The matrix records that the dossier label "N/A-with-evidence (Path B)" is invalid for this row (DP-37).

The ticket owns:

- the fact check of the FRZ-PLG-08 verdict against the decisions DP-30 and DP-37;
- the planned negative test NT-PLG-08-1 (`tests/unit/architecture/test_rw_plg08_dynamic_registration_scope.py`, not authored yet).

Relation to T-IGX-051-F: that ticket owns the NOT_COMPOSED reachability gate and its node NT-RW-GATES-NC-RAISE; this ticket reuses the gate and starts after it. The DP-30 exception texts stay with the nodes PT-D09 and PT-D10.

The candidate write set is listed under this ticket in `docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json`. It is indicative. This ticket grants no write authority: a plan with an explicit write set and its approval are required first.

## Plan

No plan exists yet.

## Out of scope

Every product or test edit until a plan is approved. No FRZ status is changed by this ticket file.

## Lifecycle state

plan_pending
