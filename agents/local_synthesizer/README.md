# LocalSynthesizerAgent

Produces reports, emails, and estimates from retrieved evidence — writes only to shadow workspace.

**Architecture:** [ARCHITECTURE.md](docs/ARCHITECTURE.md) · **Plan:** [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
**Host:** [`applications/local_workspace_application/`](../../applications/local_workspace_application/)  
**Capability:** `local.workspace.synthesize`

## Role in LKW pipeline

```text
evidence + template → LLM → workspace.write_file (shadow)
```

Requires `metadata={"shadow_workspace": True}` on the Nexus task.

## Quick start

```bash
uv run pytest agents/local_synthesizer/tests -q
```

## Templates (planned)

| `synthesis_template` | Output |
|---------------------|--------|
| `email` | Client/supplier email draft |
| `report` | Structured markdown report |
| `estimate` | Cost estimate from gathered figures |
| `custom` | Free-form per user guidelines |

## Implementation status

| Wave | Scope |
|------|-------|
| LKW.0 | Scaffold + architecture (**Done**) |
| LKW.1 | Basic synthesis + shadow write | Planned |

## Authoring

See [`../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md) Appendix B (shadow workspace).
