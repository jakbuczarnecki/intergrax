---
name: igx-devos-ticket-driven-work
description: Ticket-first delivery for InterGRAX product batches. Use when implementing changes under governance/tickets/.
---

# Ticket-driven work

1. Open the active ticket under `governance/tickets/T-IGX-*.md`.
2. Stay within **Scope**; honor **Out of scope**.
3. Every product-path commit includes `Ref: governance/tickets/T-IGX-NNN-L.md`.
4. Before push: `uv run python -m scripts.governance.igx_gov ticket`.
5. Update ticket **Lifecycle state** and **Checkpoint** when closing a batch.
