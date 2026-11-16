# AUTH-HIST-27PATH-2026-11-08 - Operator authorization for one broad diagnostic run (TRANSCRIBED by MASTER)

Provenance: NOT signed by the Operator. Transcription by the MASTER orchestrator of the Operator's chat reply of 2026-11-08 (processed after the review of the plan of T-IGX-031-A). The Operator wrote in Polish; the original is in the chat transcript and is not reproduced (English-only rule). Assurance: process_enforced.

## Question put to the Operator (MASTER, summarised)
The C10/C15 freeze (agent-env/rules/intergrax-env-vendor-qualification.md) requires a separate Operator authorization for broad diagnostic pytest. The independent plan review (rev-plan, finding 2) makes a 27-path run, about 8 minutes, a REQUIRED step of the independent audit of T-IGX-031-A (the only way to show that F-016 is resolved, i.e. no "already registered" failures in the worker's 27-path selection). Does the Operator authorize it?

## Operator's words (English rendering by MASTER)
"I consent to the 27-path run."

## Authorized (MASTER's reading - the Operator may narrow or correct it)
- ONE broad diagnostic pytest run of the worker's 27-path selection (the selection used by HIST-03: `hist03/run1.sh`, 27 paths), about 8 minutes.
- Purpose: show whether the tests-only batch T-IGX-031-A resolves HIST03-F-016 (registry pollution); executed by the independent AUDITOR of that batch, at the exact audited implementation SHA, after the implementer has finished. Not by the implementer, not by the plan approver.
- Conditions that stay in force: serial, one pytest process, heavy lock (`locks\heavy.lock`), `uv run --extra dev-ci`, exit code captured, no `-p no:cacheprovider`, no `--basetemp` outside the repo, no Docker, no new vendor integrations, no production configuration (C10/C15).
- Not authorized by this record: any other broad run; a rerun after failure beyond one retry of an identical command for an infrastructure error (to be reported); the QUAL-X Dimension A replay (that one is covered by DR-10 and is a separate run on the final integrated SHA); any run for the P2-C tests; removal of the `no_ci` marker.
- Expires: none stated by the Operator.
- Expected outcome is open: the run may show failures that are genuine (e.g. the 6 P2-C tests, finding F-015). Those are reported, not hidden.
