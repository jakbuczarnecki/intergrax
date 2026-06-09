# © Artur Czarnecki. All rights reserved.

"""CI gate — guardrail contract + bridge importable (M-P12-WIRE.3)."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from intergrax.integrations.contracts.llm_guardrail import (  # noqa: F401
            GuardrailScanResult,
            LlmGuardrailBackend,
        )
        from intergrax.applications._shared.guardrail_runtime_bridge import (  # noqa: F401
            resolve_guardrail_wiring_options,
        )
        from intergrax.applications._shared.application_guardrail_middleware import (  # noqa: F401
            LlmGuardrailMiddleware,
        )
        from intergrax.applications._shared.guardrail_assembly_resolver import (  # noqa: F401
            assert_guardrail_assembly_valid,
        )
        from intergrax.applications._shared.guardrail_wiring import (  # noqa: F401
            wire_application_guardrail,
        )
        from intergrax.integrations.providers.llm_guardrail._factory import (  # noqa: F401
            create_guardrail_backend,
        )
        from intergrax.integrations.providers.llm_guardrail.register_all import (  # noqa: F401
            register_llm_guardrail_integrations,
        )
    except ImportError as exc:
        print(f"check_harness_guardrail_wiring: import failed: {exc}")
        return 1

    from intergrax.integrations.providers.llm_guardrail._factory import create_chained_guardrail_backend

    register_llm_guardrail_integrations(override=True)
    backend = create_guardrail_backend("llm_guard")
    if not backend.health_check():
        print("check_harness_guardrail_wiring: backend health_check failed")
        return 1
    chained = create_chained_guardrail_backend("llm_guard", "presidio")
    if not chained.health_check():
        print("check_harness_guardrail_wiring: chained backend health_check failed")
        return 1
    print("check_harness_guardrail_wiring: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
