# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.ltm.contracts import (
    LtmSearchInput,
    LtmSearchOutput,
    LtmWriteFactInput,
    LtmWriteFactOutput,
)
from intergrax.tools.providers.ltm.service import ltm_search, ltm_write_fact


class LtmSearchHandler(ServiceToolHandler[LtmSearchInput, LtmSearchOutput]):
    _service = ltm_search


class LtmWriteFactHandler(ServiceToolHandler[LtmWriteFactInput, LtmWriteFactOutput]):
    _service = ltm_write_fact
