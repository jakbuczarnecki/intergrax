# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional

from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
from intergrax.tools._shared.output import limit_tool_output

# --- Optional Pydantic (as before) ---
try:
    from pydantic import BaseModel, ValidationError  # type: ignore
except Exception:  # no pydantic or another import error
    class BaseModel:  # lightweight stub
        def __init__(self, **kwargs): pass
        def model_dump(self): return {}
        @classmethod
        def model_json_schema(cls): return {"type": "object", "properties": {}, "required": []}
    class ValidationError(Exception):
        pass

__all__ = ["ToolBase", "ToolRegistry", "_limit_tool_output", "limit_tool_output"]

_limit_tool_output = limit_tool_output


def _deprecated_tools_base(name: str) -> None:
    warnings.warn(
        f"intergrax.tools.tools_base.{name} is deprecated; use intergrax.tools.registry "
        f"and intergrax.tools.exporters (Phase O.7).",
        DeprecationWarning,
        stacklevel=3,
    )


class ToolBase:
    """
    **Deprecated (Phase O.7).** Use ``ToolContract`` + ``ToolRegistry`` from
    ``intergrax.tools.registry`` instead.

    Legacy base class for ad-hoc tools in deprecated ``ChatAgent``.

Removal timeline: after Phase Q-R.12 (catalog ``tool_ids`` only). Do not import
from new agents — use ``ToolProvider`` + ``ToolRegistry`` (see ``docs/TOOLS.md``).
    """
    name: str = "tool"
    description: str = "No description"
    schema_model: Optional[type[BaseModel]] = None
    strict_validation: bool = True  # option to disable 'additionalProperties=False'

    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns an OpenAI-compatible JSON Schema 'parameters' (type=object, properties, required).
        By default, derived from Pydantic v2: model_json_schema().
        """
        if self.schema_model is None:
            # fallback: empty object
            return {"type": "object", "properties": {}, "required": []}

        raw = self.schema_model.model_json_schema()
        props = raw.get("properties", {})
        required = raw.get("required", [])

        parameters = {
            "type": "object",
            "properties": props,
            "required": required,
        }

        # Optionally enforce strict validation
        if self.strict_validation:
            parameters["additionalProperties"] = False

        return parameters

    def run(self, 
            run_id:Optional[str] = None,
            llm_usage_tracker: Optional[LLMUsageTracker] = None,
            **kwargs) -> Any:
        """Each tool must override this method."""
        raise NotImplementedError

    def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates arguments with Pydantic (if schema_model is available).
        """
        if self.schema_model is None:
            return args
        try:
            return self.schema_model(**args).model_dump()
        except ValidationError as e:
            raise ValueError(f"{self.name}: invalid arguments: {e}") from e

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Builds the JSON object for the 'tools' field in the OpenAI Responses API.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters(),
            },
        }


class ToolRegistry:
    """
    **Deprecated (Phase O.7).** Use ``intergrax.tools.registry.ToolRegistry``.

    Legacy registry for ``ToolBase`` instances (OpenAI Responses API export).
    """
    def __init__(self):
        _deprecated_tools_base("ToolRegistry")
        self._tools: Dict[str, ToolBase] = {}

    def register(self, tool: ToolBase):
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolBase:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def list(self) -> List[ToolBase]:
        return list(self._tools.values())

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Returns tools in a format compatible with the OpenAI Responses API."""
        return [t.to_openai_schema() for t in self._tools.values()]
