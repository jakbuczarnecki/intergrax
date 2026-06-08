# Integration and Adapter Architecture

**Status:** Canonical architecture (decomposed from platform canon)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Target reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../IDEAL_HARNESS_AI_ARCHITECTURE.md)

---

# 17. Adapter Architecture

Adapters are reusable integrations with external systems.

Examples:

- SlackAdapter
- TeamsAdapter
- EmailAdapter
- PostgreSqlAdapter
- RedisAdapter
- BrowserAdapter
- WebSearchAdapter
- FileSystemAdapter
- VectorStoreAdapter
- LlmProviderAdapter
- SandboxAdapter

Adapters MUST be treated like infrastructure components.

Adapters MUST NOT contain business workflow logic.

Adapters MUST NOT decide which agent to run.

Adapters expose operations.

Nexus or agents call those operations through explicit permissions.

---


---

# 18. Slack / Teams / Communication Integration Philosophy

Intergrax should support Slack and Teams as interaction surfaces.

This follows the Viktor-like idea where an AI worker can live inside organizational communication tools.

Slack and Teams should be implemented as adapters.

They may provide:

- task intake
- notifications
- approval requests
- progress updates
- final responses
- interactive buttons
- user context
- channel context

They should NOT own the runtime.

Correct model:

```text
Slack message
    -> SlackAdapter
    -> normalized Task
    -> Nexus Runtime
    -> Agent execution
    -> Nexus final result
    -> SlackAdapter sends response
```

Incorrect model:

```text
Slack bot contains orchestration logic
Slack bot directly manages agents
Slack bot stores global task state
```

---


---

# 46. Checklist For New Adapter Implementation

Before implementing a new adapter, answer:

```text
1. What external system does it connect to?
2. What operations does it expose?
3. What permissions are required?
4. Is it read-only or write-capable?
5. What are risk levels?
6. What errors can happen?
7. What timeout/retry policy is needed?
8. What data should be logged?
9. What data must be protected?
10. Which agents or runtime components may use it?
```

Adapters should be generic and reusable.

---

