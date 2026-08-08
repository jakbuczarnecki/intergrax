# Security Policy

## Supported versions

Intergrax is under **active private R&D**. Security fixes are applied to the **current main branch** only.

| Version | Supported |
|---------|-----------|
| `main` (latest) | Yes |
| Older commits / tags | Best effort |

---

## Reporting a vulnerability

**Do not** open a public GitHub issue for security vulnerabilities.

### Contact

| Channel | Details |
|---------|---------|
| **Email** | jakbu.czarnecki.83@gmail.com |
| **Subject** | `[Intergrax Security]` brief description |

### What to include

1. Description of the vulnerability and potential impact
2. Steps to reproduce (proof of concept if available)
3. Affected component (Nexus runtime, integration, tool, application, …)
4. Suggested fix or mitigation (if known)
5. Your contact information for follow-up

### Response timeline

| Stage | Target |
|-------|--------|
| Acknowledgment | Within 5 business days |
| Initial assessment | Within 10 business days |
| Fix or mitigation plan | Depends on severity |

We will coordinate disclosure timing with you. Credit will be given if desired and appropriate.

---

## Security architecture (overview)

Intergrax implements security as a **control plane** within the Harness AI platform. Canonical references:

| Topic | Document |
|-------|----------|
| Policy engine | [docs/project/architecture/intergrax_runtime_architecture.md §42.11](docs/project/architecture/intergrax_runtime_architecture.md) |
| Security control plane | [docs/project/technical/guides/AGENT_CREATION_GUIDE.md Appendix S](docs/project/technical/guides/AGENT_CREATION_GUIDE.md) |
| Production hardening | [docs/project/architecture/intergrax_runtime_architecture.md Phase U](docs/project/architecture/intergrax_runtime_architecture.md) |
| Harness audit (security layers) | [docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md](docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md) |

### Key security mechanisms

- **PolicyEngine** — pre-run, pre-tool, post-tool governance hooks
- **ToolRuntime** — unified tool gateway with policy, trace, and idempotency
- **Tier boundaries** — agents cannot bypass Nexus to access integrations directly
- **Human-in-the-loop (HITL)** — governance gates for sensitive operations
- **Trace & audit** — observability for security-relevant events
- **Cost governance** — budget controls for LLM and tool usage

---

## Security best practices for contributors

### Secrets

- **Never** commit API keys, tokens, passwords, or `.env` files
- Use environment variables — see integration docs in [docs/project/architecture/INTEGRATIONS.md](docs/project/architecture/INTEGRATIONS.md)
- Rotate credentials if accidentally exposed

### Dependencies

- Dependencies managed via `uv` / `pyproject.toml`
- Report supply-chain concerns to the security email above

### Agent and tool safety

- New tools must go through `ToolRuntime` with policy hooks
- Do not bypass `PolicyEngine` for convenience
- Follow [docs/project/technical/guides/AGENT_CREATION_GUIDE.md Appendix S](docs/project/technical/guides/AGENT_CREATION_GUIDE.md) for security wiring

### Infrastructure

- Local Docker backends: [infra/README.md](infra/README.md) — do not expose to public networks in development
- Lab harness presets: [docs/project/technical/guides/HARNESS_ENVIRONMENT.md](docs/project/technical/guides/HARNESS_ENVIRONMENT.md)

---

## Scope

### In scope

- Nexus runtime (`intergrax/runtime/`)
- ToolRuntime and PolicyEngine
- Integration connectors (`intergrax/integrations/`)
- Tool and skill execution paths
- Application hosts (`applications/`)
- Authentication/authorization in application APIs
- Data handling in RAG and memory subsystems

### Out of scope

- Third-party LLM provider security (report to the provider)
- User-deployed infrastructure misconfiguration
- Social engineering
- Denial of service against public endpoints not operated by the project

---

## License & confidentiality

Intergrax is proprietary software. See [LICENSE](LICENSE) and the collaboration model in [COLLABORATION.md](docs/project/community/COLLABORATION.md). Security reports are handled confidentially.
