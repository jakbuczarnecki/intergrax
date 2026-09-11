# Enterprise Execution Scale & Resilience — W4-D Provider Cancellation Inventory

**Task:** W4-D — Provider native cancellation & adapter termination boundary  
**Status:** INVENTORY COMPLETE

## Provider matrix (ETAP 1)

| Provider | Supports cancel | API type | Ownership |
|----------|-----------------|----------|-----------|
| OpenAI (+ compat: Groq, vLLM, OpenRouter, Azure OpenAI) | Yes (stream abort + registry) | HTTP/SDK | Adapter (`providers/openai/cancellation.py`) |
| Claude | Yes (stream abort) | Anthropic SDK | Adapter (`providers/claude/cancellation.py`) |
| Gemini / Vertex Gemini | Yes (stream abort) | Google SDK | Adapter (`providers/gemini/cancellation.py`) |
| Mistral | Yes (stream abort) | Mistral SDK | Adapter (`providers/mistral/cancellation.py`) |
| Bedrock | Stream abort only (`supports_native_cancel=False`) | AWS SDK | Adapter (`providers/aws_bedrock/cancellation.py`) |
| Ollama | No native (`supports_native_cancel=False`); stream close only | Local HTTP | Adapter (`providers/ollama/cancellation.py`) |
| Runtime tools | No (`ToolExecutorTerminationPort`; optional `Future.cancel` pre-start) | Thread pool executor | Tool boundary (`runtime/nexus/tools/tool_operation_termination.py`) |

## Ports

| Port | Location | Scope |
|------|----------|-------|
| `ExternalOperationCancellationPort` | `contracts/external_operation_cancellation.py` | W4-C intent signal |
| `ExternalOperationTerminationPort` | `contracts/external_operation_termination.py` | W4-D physical termination |
| `ExternalOperationCapabilities` | same | Capability discovery |

## Seam wiring

- `resolve_llm_provider_external_operation_seam(slug)` — `runtime/external_operations/provider_cancellation.py`
- `bind_external_operation_ports` on `LLMAdapter` — termination + stream registry
- Observe helper — `runtime/external_operations/operation_termination.py`
