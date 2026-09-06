# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Provider-owned LLM adapter registration modules."""

from intergrax.llm_adapters.providers.registrations.builtin import register_builtin_llm_adapters

__all__ = ["register_builtin_llm_adapters"]
