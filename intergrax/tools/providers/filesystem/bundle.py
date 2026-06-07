# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.filesystem.contracts import (
    FilesystemGlobInput,
    FilesystemGlobOutput,
    FilesystemListInput,
    FilesystemListOutput,
    FilesystemReadTextInput,
    FilesystemReadTextOutput,
    FilesystemStatInput,
    FilesystemStatOutput,
    FilesystemWriteTextInput,
    FilesystemWriteTextOutput,
)
from intergrax.tools.providers.filesystem.handlers import (
    FilesystemGlobHandler,
    FilesystemListHandler,
    FilesystemReadTextHandler,
    FilesystemStatHandler,
    FilesystemWriteTextHandler,
)
from intergrax.tools.providers.filesystem.service import (
    FILESYSTEM_GLOB_TOOL_ID,
    FILESYSTEM_LIST_TOOL_ID,
    FILESYSTEM_READ_TEXT_TOOL_ID,
    FILESYSTEM_STAT_TOOL_ID,
    FILESYSTEM_WRITE_TEXT_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

FILESYSTEM_BUNDLE_ID = "filesystem"
FILESYSTEM_TOOL_IDS: tuple[str, ...] = (
    FILESYSTEM_LIST_TOOL_ID,
    FILESYSTEM_GLOB_TOOL_ID,
    FILESYSTEM_READ_TEXT_TOOL_ID,
    FILESYSTEM_STAT_TOOL_ID,
    FILESYSTEM_WRITE_TEXT_TOOL_ID,
)


def register_filesystem_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=FILESYSTEM_LIST_TOOL_ID,
            name=FILESYSTEM_LIST_TOOL_ID,
            description="List entries in an allowlisted directory (read-only user filesystem browse).",
            description_short="List allowlisted directory.",
            input_schema=FilesystemListInput,
            output_schema=FilesystemListOutput,
            error_mapping={},
            side_effects=False,
            category="filesystem",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("filesystem", "browse", "read_only"),
        ),
        FilesystemListHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=FILESYSTEM_GLOB_TOOL_ID,
            name=FILESYSTEM_GLOB_TOOL_ID,
            description="Glob file paths under an allowlisted root (read-only).",
            description_short="Glob allowlisted paths.",
            input_schema=FilesystemGlobInput,
            output_schema=FilesystemGlobOutput,
            error_mapping={},
            side_effects=False,
            category="filesystem",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("filesystem", "browse", "read_only"),
        ),
        FilesystemGlobHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=FILESYSTEM_READ_TEXT_TOOL_ID,
            name=FILESYSTEM_READ_TEXT_TOOL_ID,
            description="Read UTF-8 text from an allowlisted file with a byte cap (read-only).",
            description_short="Read allowlisted text file.",
            input_schema=FilesystemReadTextInput,
            output_schema=FilesystemReadTextOutput,
            error_mapping={},
            side_effects=False,
            category="filesystem",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("filesystem", "read_only"),
        ),
        FilesystemReadTextHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=FILESYSTEM_STAT_TOOL_ID,
            name=FILESYSTEM_STAT_TOOL_ID,
            description="Return metadata for an allowlisted path (size, mtime, type).",
            description_short="Stat allowlisted path.",
            input_schema=FilesystemStatInput,
            output_schema=FilesystemStatOutput,
            error_mapping={},
            side_effects=False,
            category="filesystem",
            risk_level=ToolRiskLevel.LOW,
            tags=("filesystem", "metadata", "read_only"),
        ),
        FilesystemStatHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=FILESYSTEM_WRITE_TEXT_TOOL_ID,
            name=FILESYSTEM_WRITE_TEXT_TOOL_ID,
            description="Write UTF-8 text to an allowlisted file with a byte cap (creates parent dirs when enabled).",
            description_short="Write allowlisted text file.",
            input_schema=FilesystemWriteTextInput,
            output_schema=FilesystemWriteTextOutput,
            error_mapping={},
            side_effects=True,
            category="filesystem",
            risk_level=ToolRiskLevel.HIGH,
            tags=("filesystem", "write"),
        ),
        FilesystemWriteTextHandler(ctx),
    )
