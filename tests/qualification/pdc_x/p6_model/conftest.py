# © Artur Czarnecki. All rights reserved.

"""Restore global LLM registry state after every B-P6-M test (same isolation as the XD suite)."""

from __future__ import annotations

from tests.qualification.pdc_x.xd.conftest import _isolate_llm_global_state as _isolate_llm_global_state
