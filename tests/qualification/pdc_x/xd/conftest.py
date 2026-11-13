# © Artur Czarnecki. All rights reserved.

"""Keep the B-P6 XD tests from leaking global LLM state into other suites.

The tests import provider adapter modules and resolve builtin registrations. Other suites
(``test_p2_002_c_provider_registration``) assume provider modules are imported lazily and the
registry is untouched, so both are restored after every test.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

_PROVIDER_MODULE_PREFIX = "intergrax.llm_adapters.providers."


@pytest.fixture(autouse=True)
def _isolate_llm_global_state() -> Iterator[None]:
    modules_before = set(sys.modules)
    factories = dict(LLMAdapterRegistry._factories)
    seams = dict(LLMAdapterRegistry._external_operation_seam_factories)
    capabilities = dict(LLMAdapterRegistry._external_operation_capabilities)
    try:
        yield
    finally:
        LLMAdapterRegistry._factories.clear()
        LLMAdapterRegistry._factories.update(factories)
        LLMAdapterRegistry._external_operation_seam_factories.clear()
        LLMAdapterRegistry._external_operation_seam_factories.update(seams)
        LLMAdapterRegistry._external_operation_capabilities.clear()
        LLMAdapterRegistry._external_operation_capabilities.update(capabilities)
        for name in set(sys.modules) - modules_before:
            if name.startswith(_PROVIDER_MODULE_PREFIX) and name.endswith("_adapter"):
                sys.modules.pop(name, None)
