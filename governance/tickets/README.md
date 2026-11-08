# InterGRAX delivery tickets (SSOT)

Ticket path: `governance/tickets/T-IGX-<NN>-<L>.md` where `<NN>` is zero-padded batch number and `<L>` is scope letter (A, B, …).

## DCP closure program (batch 004+)

Remaining Delivery DevOS closure work is tracked in **[`governance/DCP-CLOSURE-BACKLOG.md`](../DCP-CLOSURE-BACKLOG.md)** (waves W1–W8). Use ticket batches **`T-IGX-004-A`** through **`T-IGX-008-A`** (and `-B` splits only when file trees do not conflict). Batch **001–003** cover phase 0–7 delivery and proportionate doc follow-ups.

## Required sections

Every ticket MUST include these headings (order may vary):

1. **Scope** — what changes, linked domains/plans.
2. **Plan digest** — `Plan digest: <sha256>` from `scripts/governance/plan_digest.py` when a plan file is attached.
3. **Approved** — after plan review (procedural gateway), one of two permitted forms (policy; not enforced by tooling):
   - `Approved: <operator> @ <ISO8601>` — written by the operator; or
   - `Approved: operator-delegate @ <ISO8601>` followed on the next line by `Delegation: <reference>` — written by a session the operator commissioned (IIAC Amendment 01, rules R-1..R-6; a commission given in chat is sufficient; the reference points to its transcription, e.g. `<path>@<40-hex sha>`; tooling that follows Amendment 01 section 4.1 B-1 accepts only that form, so a local session record should first be published and referenced by commit). The commissioned session must be a new session dedicated to that task, not a continuation and not an implementation session (R-2), and must not be the implementer of the ticket (R-3).
   An implementer never writes either form for its own ticket; assurance of the delegate form is `process_enforced`. `verify_plan_approval.py` / `verify_ticket_lifecycle.py` currently test only for a line starting `Approved:`; they do not distinguish the forms.
4. **Test contract** — pytest/CI commands that must pass before batch close.
5. **Out of scope** — explicit exclusions (FRZ, Docker, vendor, etc.).
6. **Lifecycle state** — one of: `draft` | `plan_pending` | `plan_approved` | `in_progress` | `quality_gate` | `ready_for_audit` | `closed`.

Optional:

- **Lease holder** — advisory session/worktree id when parallel agents might collide (`ticket_lock.py`).
- **Checkpoint** — paste of `igx_gov ticket` stdout before push or audit handoff.

## Commit trailer

Each commit that touches product paths (`intergrax/`, `agents/`, `applications/`, `scripts/`) MUST include a trailer line:

```text
Ref: governance/tickets/T-IGX-012-A.md
```

Enforced by `verify_commit_ticket_refs.py` on pre-push (and `igx_gov ticket`).

## Lifecycle (summary)

| State | Meaning |
|-------|---------|
| `draft` | Ticket scaffold; no product commits yet. |
| `plan_pending` | Plan digest recorded; awaiting the approval line (operator or commissioned session). |
| `plan_approved` | `Approved:` present (operator or `operator-delegate` with `Delegation:`); implementation may start. |
| `in_progress` | Active worktree / agent batch. |
| `quality_gate` | Implementer ran `igx_gov ticket`; ready for PR/review. |
| `ready_for_audit` | Independent auditor session may start (G6). |
| `closed` | Merged or abandoned with operator note. |

Implementer MUST NOT set `ready_for_audit` to PASS or claim G6 certification — see `INDEPENDENT-REVIEW-HANDOFF.md`.

## Related docs

- [`governance/DCP-CLOSURE-BACKLOG.md`](../DCP-CLOSURE-BACKLOG.md)
- [`docs/project/technical/devos/DEVELOPER-QUALITY-PROCESS.md`](../../docs/project/technical/devos/DEVELOPER-QUALITY-PROCESS.md)
- [`governance/agent-control-plane/`](../../governance/agent-control-plane/)
