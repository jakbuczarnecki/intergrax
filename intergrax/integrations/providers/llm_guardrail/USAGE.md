# LLM guardrail integrations (M.12)

Vendor LLM safety scanners for Tier-3 hosts. Agents **must not** import these SDKs — use `IntegrationProfile.llm_guardrail` + `GuardrailProfile`.

## Quick start

```python
from intergrax.integrations.registry.presets import harness_guardrail_stack
from intergrax.applications.contracts.environment_profile import GuardrailProfile

integration_profile = harness_guardrail_stack(primary="llm_guard", semantic="presidio")
guardrail_profile = GuardrailProfile(enabled=True, scan_input=True, scan_output=True)
```

## Optional dependencies

```bash
# Presidio (shipped extra — no torch conflict)
uv sync --extra integrations-guardrails

# Heavy vendors — separate venv recommended (torch/docling pin conflict)
pip install llm-guard guardrails-ai nemoguardrails
```

## Environment variables

| Slug | Variables |
|------|-----------|
| `openguardrails` | `INTERGRAX_OPENGUARDRAILS_API_KEY`, `INTERGRAX_OPENGUARDRAILS_BASE_URL` |
| `lakera` | `INTERGRAX_LAKERA_API_KEY`, `INTERGRAX_LAKERA_BASE_URL` |
| `azure_content_safety` | `INTERGRAX_AZURE_CONTENT_SAFETY_*` |
| `nemo_guardrails` | `INTERGRAX_NEMO_COLANG_CONFIG_PATH` or `GuardrailProfile.colang_config_path` |
| `bedrock_guardrails` | `INTERGRAX_BEDROCK_GUARDRAIL_POLICY_ID` or profile field |
| `llama_guard` | `INTERGRAX_LLAMA_GUARD_INFERENCE_URL` or `GuardrailProfile.inference_slug` |

## Verification

```bash
uv run pytest tests/unit/integrations/providers/llm_guardrail/ -m gate -q
python scripts/maintenance/check_harness_guardrail_wiring.py
```

Canon: `docs/project/architecture/INTEGRATIONS.md` §47 · ADR-GR-001
