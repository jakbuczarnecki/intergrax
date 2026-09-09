"""Module entry point: python -m ...storage_bootstrap.operator"""

from __future__ import annotations

import sys

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.cli import (
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
