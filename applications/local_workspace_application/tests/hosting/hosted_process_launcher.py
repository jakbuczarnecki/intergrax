# © Artur Czarnecki. All rights reserved.

"""Test-only LKW hosting subprocess entry with one seeded active projection."""

from __future__ import annotations

import json

from local_workspace_application.hosting.__main__ import _exit_code, _safe_result_payload
from local_workspace_application.hosting.foreground import run_local_workspace_hosted_application
from local_workspace_application.tests.lkw_ac3_projection import (
    create_lkw_hosted_test_process_composition,
)


def main() -> int:
    composition = create_lkw_hosted_test_process_composition(seed_active_projection=True)
    result = run_local_workspace_hosted_application(process_composition=composition)
    print(
        json.dumps(
            _safe_result_payload(result),
            sort_keys=True,
        ),
        flush=True,
    )
    return _exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
