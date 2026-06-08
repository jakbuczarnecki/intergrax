# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.skill_tool.contracts import SkillResolveInput, SkillResolveOutput
from intergrax.tools.providers.skill_tool.service import skill_resolve


class SkillResolveHandler(ServiceToolHandler[SkillResolveInput, SkillResolveOutput]):
    _service = skill_resolve
