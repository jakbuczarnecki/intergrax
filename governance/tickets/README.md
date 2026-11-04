# InterGRAX delivery tickets (SSOT)

Ticket path: `governance/tickets/T-IGX-<NN>-<L>.md` where `<NN>` is zero-padded batch number and `<L>` is scope letter (A, B, …).

## DCP closure program (batch 004+)

Remaining Delivery DevOS closure work is tracked in **[`governance/DCP-CLOSURE-BACKLOG.md`](../DCP-CLOSURE-BACKLOG.md)** (waves W1–W8). Use ticket batches **`T-IGX-004-A`** through **`T-IGX-008-A`** (and `-B` splits only when file trees do not conflict). Batch **001–003** cover phase 0–7 delivery and proportionate doc follow-ups.

## Required sections

Every ticket MUST include these headings (order may vary):

1. **Scope** — what changes, linked domains/plans.
2. **Plan digest** — `Plan digest: <sha256>` from `scripts/governance/plan_digest.py` when a plan file is attached.
3. **Approved** — `Approved: <operator> @ <ISO8601>` after plan review (procedural gateway).
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
| `plan_pending` | Plan digest recorded; awaiting operator approval line. |
| `plan_approved` | Operator `Approved:` present; implementation may start. |
| `in_progress` | Active worktree / agent batch. |
| `quality_gate` | Implementer ran `igx_gov ticket`; ready for PR/review. |
| `ready_for_audit` | Independent auditor session may start (G6). |
| `closed` | Merged or abandoned with operator note. |

Implementer MUST NOT set `ready_for_audit` to PASS or claim G6 certification — see `INDEPENDENT-REVIEW-HANDOFF.md`.

## Related docs

- [`governance/DCP-CLOSURE-BACKLOG.md`](../DCP-CLOSURE-BACKLOG.md)
- [`docs/project/technical/devos/DEVELOPER-QUALITY-PROCESS.md`](../../docs/project/technical/devos/DEVELOPER-QUALITY-PROCESS.md)
- [`governance/agent-control-plane/`](../../governance/agent-control-plane/)
