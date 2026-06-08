# Tier-3 Application Environment, Sandbox, and Shadow Workspace

**Status:** Canonical architecture (decomposed from platform canon)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Target reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../IDEAL_HARNESS_AI_ARCHITECTURE.md)

---

# 19. UI / UX Testing Requirement

Even though Intergrax is not frontend-heavy, agents must be testable and observable.

The system should support minimal UI/UX surfaces for:

- viewing task list
- viewing task status
- viewing execution trace
- viewing agent outputs
- viewing tool calls
- viewing errors
- viewing artifacts
- approving or rejecting steps
- re-running tasks
- comparing outputs

This may be implemented as:

- lightweight dashboard
- CLI
- chat interface
- Slack/Teams messages
- internal debug panel

The UI is for observability and experimentation, not product polish.

---


---

# 20. Shadow Workspace Model

A Shadow Workspace is an isolated temporary workspace used to perform work without directly modifying the main environment.

Inspired by Cursor-like execution environments.

Shadow Workspaces may be used for:

- code experiments
- document analysis
- temporary data transformations
- simulated business workflows
- vendor research sessions
- legal document review sessions
- onboarding simulations

A Shadow Workspace should provide:

- isolation
- temporary storage
- reproducibility
- rollback safety
- inspectable artifacts
- cleanup

---


---

# 21. Sandbox Model

A sandbox is a controlled execution environment.

Use sandboxes for:

- code execution
- browser automation
- file manipulation
- risky tool use
- external data extraction
- generated script execution

Sandbox execution should be:

- isolated
- observable
- permission-controlled
- interruptible
- disposable
- reproducible when possible

---

