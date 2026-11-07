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

## Amendment (2026-11-07) - wording of the clean environment (certifier finding C-01)

The wording "no `PYTHONPATH`" above cannot be met literally: the ACT-N10 companions (`verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`, material and unchanged) import `scripts.governance...` and need the repository root on their path; without it the gate fails closed with `A-01 ... No module named 'scripts'`. The binding procedure is therefore: a new virtual environment **outside** the clone; no `sitecustomize`/`usercustomize`; no `.pth` file except the editable install of the project that points at the clean clone itself (`uv sync --extra dev-ci --frozen`), **or** `PYTHONPATH` set to exactly the clean clone and nothing else; interpreter from a trusted location; the commands and the environment recorded in the evidence. The certifier run of 2026-11-07 followed this form. The scope decision above is unchanged.

## Operator ratification (2026-11-07)

The independent certifier session `local_dbad2215-acac-4a9e-aaf6-ad3be89aee18` (report `EVOL_X_W4_01_CERTIFIER_REPORT.md`, finding C-02) noted that this decision was taken by the implementer under a blanket authorisation. On 2026-11-07 the operator wrote in the chat of the implementer session `local_4b071fe5-82d0-48bc-b524-146a3daf5c45`: „Otrzymujesz wszystkie moje zgody i akceptacje, które są wymagane do zamknięcia tego etapu i wszystkich zadań, które wchodzą w jego skład. Wykonaj teraz ostateczne domknięcie całego zadania i etapu, tak abym mógł odznaczyć je jako wykonane.” The implementer records that statement here as the operator's ratification of this decision as written (the original text above is unchanged). Like the first entry this is a delegated, process-enforced record (assurance `process_enforced`), not a handwritten or cryptographic signature; the operator may replace it at any time. It takes none of OD-EVOL-12, OD-EVOLX-08 or P3.

## Amendment 2 (2026-11-07) - `python -m` is not a supported command form (finding CERT-01)

**Finding.** A second independent certifier session, `local_e34301ce-67df-40bf-b274-b2e062581541`, reproduced against the pinned code `fdb1e7a62` that a forged, git-ignored `UNCHECKED_HASH` bytecode file (`scripts/__pycache__/__init__.cpython-312.pyc`, or the one of the preflight module itself) is executed and ends the process with exit 0 when the preflight is started as `python -m scripts.qualification.evol_x_closure_publication_preflight`. The script-path form rejected every such plant. The decision above listed both forms as in scope, so the review trigger ("T1-T3, MEDIUM or higher, reproduced against the pinned SHA") was met. Report: `EVOL_X_W4_01_SECOND_CERTIFIER_REPORT.md`.

**Why this cannot be fixed in the script.** With `-m` the interpreter imports the packages `scripts` and `scripts.qualification` (and loads their cached bytecode) before the first line of the preflight module runs. The recovery session `local_7eeb7387-df63-460d-88e9-be7359430173` reproduced this on 2026-11-07 in a throw-away copy of the pinned `scripts/` tree with a venv outside the repository: a forged `scripts/__init__` bytecode that writes a marker and calls `os._exit(0)` ran first and ended the process with exit 0, and a module-level refusal guard placed at the very top of the preflight (the correction the certifier suggested) never executed. A guard in the module therefore protects only against accidental use, not against the T2 plant, while any edit of `scripts/qualification/evol_x_closure_publication_preflight.py` changes the immutable pin, voids `EVOL_X_OD3_DECISION_RECORD.md` condition 3 and the B-4 resolution evidence, and would not close the finding. The recovery session therefore made no code change.

**Decision (delegated, see provenance below).** `python -m scripts.qualification...` is **not a supported or defended command form** of the closure preflight. The only supported form is the script path, run from a fresh clone of the pinned SHA in the clean environment of Amendment 1:

`python scripts/qualification/evol_x_closure_publication_preflight.py --gate <gate>`

The wording "and `python -m scripts.qualification...`" in the Decision section above is superseded by this amendment. Every certification, B6 or P3 run must use the script path; a result obtained with `-m` is not evidence. CERT-01 is accepted as a documented boundary of the `-m` form (same family as T4: control of what the interpreter imports before the script starts), not as a defect of the script path. CERT-02 (a self-deleting root module evades `A-21` inside the companions, reachable only with the repository root on `PYTHONPATH`) is recorded as an extension of the accepted LOW residual W405-01. CERT-03 is closed by Amendment 1.

**Review trigger (unchanged).** Re-open this decision if a T1-T3 finding of MEDIUM or higher is reproduced against the pinned SHA **in the script-path form**, and before P3 is authorised.

**Provenance.** On 2026-11-07 the operator wrote in the chat of the recovery session `local_7eeb7387-df63-460d-88e9-be7359430173`: "Chcę, abyś kompletnie zamknął to zadanie. Jeżeli wymagane są korekty w kodzie, wykonaj implementację, zaktualizuj dokumenty i doprowadź do całkowitego zamknięcia zadania. Dostajesz moje wszystkie zgody i autoryzacje, dlatego przeprowadź tę operację end-to-end autonomicznie." The recovery session records this amendment on that instruction. It is a delegated, process-enforced entry (assurance `process_enforced`), not a handwritten or cryptographic signature. It narrows the scope after a finding, so the operator should confirm or overrule it in their own words or commit. It changes no code, gate, material governance path, activation record or protocol and takes none of OD-EVOL-12, OD-EVOLX-08 or P3.
