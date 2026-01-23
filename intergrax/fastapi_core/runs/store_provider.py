# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.fastapi_core.runs.store_base import RunStore


def get_run_store(store: RunStore) -> RunStore:
    """
    Dependency wrapper for RunStore.

    Concrete implementation is injected via app configuration.
    """
    return store
