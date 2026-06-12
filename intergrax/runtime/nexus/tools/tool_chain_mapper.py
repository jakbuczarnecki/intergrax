# © Artur Czarnecki. All rights reserved.

"""Map chain step inputs from user query and prior step outputs (TOOL-ENG-20)."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from intergrax.runtime.nexus.tools.tool_chain_spec import (
    USER_QUERY_SOURCE,
    ChainInputSource,
    ChainStep,
    FieldRef,
)
from intergrax.tools.core.contracts import ToolContract


def resolve_mapping_value(
    source: ChainInputSource,
    *,
    user_query: str,
    prior_outputs: Sequence[BaseModel],
) -> object:
    if isinstance(source, FieldRef):
        if source.step >= len(prior_outputs):
            raise ValueError(f"FieldRef step {source.step} out of range (have {len(prior_outputs)} outputs)")
        payload = prior_outputs[source.step].model_dump()
        if source.field not in payload:
            raise KeyError(f"Field '{source.field}' missing from step {source.step} output")
        return payload[source.field]
    if source == USER_QUERY_SOURCE:
        return user_query
    return source


def build_chain_step_input(
    step: ChainStep,
    *,
    contract: ToolContract,
    user_query: str,
    prior_outputs: Sequence[BaseModel],
) -> BaseModel:
    if not step.input_mappings:
        from intergrax.runtime.nexus.tools.tool_input_defaults import default_tool_input

        default = default_tool_input(contract, user_query)
        if default is None:
            raise ValueError(f"Chain step {step.tool_id} has no input_mappings and no default input")
        return default

    kwargs: dict[str, object] = {}
    for field_name, source in step.input_mappings.items():
        kwargs[field_name] = resolve_mapping_value(
            source,
            user_query=user_query,
            prior_outputs=prior_outputs,
        )
    return contract.input_schema.model_validate(kwargs)
