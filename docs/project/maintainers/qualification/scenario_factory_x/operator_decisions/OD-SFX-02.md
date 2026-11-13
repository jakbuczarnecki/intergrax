# OD-SFX-02 — Scenario Studio frontend stack

| Field | Value |
| --- | --- |
| **Decision ID** | OD-SFX-02 |
| **Stage** | SCENARIO-FACTORY-X |
| **Status** | **APPROVED** |
| **Provenance** | Transcribed by the SFX orchestrator session as a labelled English rendering of the operator's Polish chat. Not operator-signed. |
| **Ticket** | `governance/tickets/T-IGX-037-A.md` |

## Question

Which stack does the Scenario Studio application use?

## Decision (English rendering of the operator's chat)

React + TypeScript SPA kept in the repository (option chosen: "React+TS SPA in the repo"): `applications/scenario_studio_application/` with a FastAPI backend built on `intergrax/fastapi_core` and a React + TypeScript + Vite SPA in the `web/` subfolder.

## Operator line

**Provenance:** signed by an operator-designated signing session under `docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-OD-SIGNING.md@7dab0e14c71c0a6f42a0c7bed5f4cb9d52355ad1` (IIAC Amendment 01, commission given in chat 2026-11-12; operator answer "Sesja podpisująca"). The line below is in the plain form because it is written by that session; it is not claimed to be operator-written. **Transcription check:** compared with the SFX planning chat transcript (session "Scenario Factory-X campaign planning", 2026-11-12), read by this session through the local session transcript: the operator selected the option "React+TS SPA w repo (Recommended)" (2026-11-12T11:53Z). The decision text renders that option's stated description (`applications/scenario_studio_application/`, FastAPI on `intergrax/fastapi_core`, React + TypeScript + Vite SPA in `web/`); judged faithful. It chooses a stack only and authorizes no product code before ARCH-FREEZE is CLOSED (OD-SFX-04).

```text
Approved: OD-SFX-02 @ 2026-11-12T14:10:14+01:00
```

**Explicit non-actions:** does not authorize SFX-P5 or later, does not accept any ADR, does not change FRZ, does not satisfy GATE-SFX-ENTRY, and does not bypass SCENARIO-GATE or PROD-Q.
