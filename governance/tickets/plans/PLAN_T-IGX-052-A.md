# PLAN T-IGX-052-A — W5 ENT baseline pre-freeze (scaffold)

Status: **DRAFT**. Source: MEP-AF2 v2.1 section I; WP-12 **PT-S06** edge.

## 1. Purpose

Record baseline promotion prerequisites at **G2** so IIAC activation does not fail on manifest digest drift during later gates.

## 2. Prerequisites

- FCS rung; `git diff --name-only G2 <tip> -- intergrax/ agents/ applications/` empty before start (MEP).
- PT-S06 prep complete; T-AF2-RED-CLASSES merged per W5 checklist.

## 3. Write set

| Layer | Paths |
|-------|-------|
| Governance | IIAC activation notes under `governance/` (activation record only when Operator commissions) |
| Docs | Checkpoint lines in `continuation/W5_PREP_DOSSIER.md` |

## 4. Test contract

`igx_gov ticket` push-fast; optional scoped IIAC unit verifiers listed at approval.

## 5. Out of scope

Baseline promotion execution without certifier; product edits; merge to `arch-freeze/rw-train`.