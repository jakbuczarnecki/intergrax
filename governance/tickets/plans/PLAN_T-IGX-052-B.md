# PLAN T-IGX-052-B — W5 freeze manifests final (scaffold)

Status: **DRAFT**. Extends **PLAN_T-IGX-050-G** for G2-final generation.

## 1. Purpose

Produce non-draft manifest artefacts from **G2** and wire drift detection before EBH-7 pin prep.

## 2. Prerequisites

- FCS; **T-IGX-050-G** drafts reviewed.
- DP-40 disposition stable in FRZ matrix.

## 3. Write set

| Item | Path |
|------|------|
| Manifests | `docs/.../arch_freeze_program/manifests/*` (final labels per README) |
| Drift gate | `scripts/maintenance/check_arch_freeze_manifest_drift.py` (when authorized) |
| Ticket | `governance/tickets/T-IGX-052-B.md` |

## 4. Test contract

Push-fast `igx_gov`; manifest gate pytest paths filled at `Approved:`.

## 5. Out of scope

W2 draft generator edits; composition fixes in product tree; independent acceptance.