# EVOL-X OD-2 — decision record: threat-model boundary of the closure preflight

**Status:** `DECIDED — T1_T3_IN_SCOPE_T4_OUT_OF_SCOPE` · **Date:** 2026-11-06 · **Ticket:** `governance/tickets/T-IGX-019-A.md`
**Provenance:** the operator instructed the implementer agent session `local_4b071fe5-82d0-48bc-b524-146a3daf5c45` in chat to complete the steps of this task without further authorisation questions ("Wykonaj wszystkie niezbędne kroki, nie pytając mnie już o autoryzację. Udzielam ci autoryzacji na wszystkie decyzje."). The agent proposed OD-2 in its round-3/4/5 reports (report sections P and Q) and records the option below on the operator's behalf, exactly as for OD-1. This is a delegated, process-enforced entry (assurance `process_enforced`), not a handwritten or cryptographic signature; the operator can supersede this file at any time. It is **not** read by the closure preflight and takes none of OD-EVOL-12, OD-EVOLX-08 or P3.

## Question

Five audit rounds (W401-W405) each found another way for someone with write access to the machine that runs the preflight to bend its verdict. Where does the preflight's responsibility end?

## Capability tiers (as used by the independent auditor)

| Tier | Capability |
|---|---|
| T1 | write access to the working tree only: new or modified files that `git status` shows |
| T2 | write access to ignored or hidden files (git-ignored paths, `__pycache__`, `.git/info/exclude` entries, repository-root files) |
| T3 | write access to the git index, git config or refs |
| T4 | control of the Python environment or the machine: `sitecustomize`, `usercustomize`, `.pth` files, a replaced interpreter, the project virtual environment |

## Decision

* **In scope (the preflight must fail closed):** T1, T2 and T3, including the documented command forms (`python scripts/qualification/evol_x_closure_publication_preflight.py ...` and `python -m scripts.qualification...`). Findings in these tiers keep blocking acceptance.
* **Out of scope for the script (cannot be closed inside it):** T4. The mitigation is procedural and mandatory for the certification run: the certifier runs the preflight **on a fresh clone of the pinned SHA, in a clean environment** (new virtual environment created outside the clone, no `PYTHONPATH`, no `sitecustomize`/`usercustomize`, interpreter from a trusted location), and records the commands and the environment in the certification evidence.
* **Accepted LOW residuals within the in-scope tiers:** W405-01 (the IIAC companion subprocesses must keep the repository root on `PYTHONPATH`; a stray ignored root module is flagged by `A-21` but can execute inside them first; their exit codes cannot produce a PASS because the checked inputs are sealed), W405-02 (any untracked root-level Python file blocks the gates; a usability cost), W404-04, W401-RA-05, W401-RA-07. Each is re-checked by the next independent audit.

## Review trigger

The decision is re-opened if an in-scope tier (T1-T3) finding of severity MEDIUM or higher is reproduced against the pinned SHA, and before the P3 B6 acceptance is authorised.

## Limits

No material governance path, activation record, gate or protocol changed by this record. It authorises no B6 step, promotes no FRZ status and does not close EVOL-X.
