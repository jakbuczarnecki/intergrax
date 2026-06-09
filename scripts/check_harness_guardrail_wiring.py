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
        from intergrax.applications._shared.guardrail_wiring import (  # noqa: F401
            wire_application_guardrail,
        )
    except ImportError as exc:
        print(f"check_harness_guardrail_wiring: import failed: {exc}")
        return 1
    print("check_harness_guardrail_wiring: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
