# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.workspace.contracts import (
    WorkspaceArtifactOutput,
    WorkspaceListFilesInput,
    WorkspaceListFilesOutput,
    WorkspaceReadFileInput,
    WorkspaceReadFileOutput,
    WorkspaceSnapshotInput,
    WorkspaceSnapshotOutput,
    WorkspaceWriteFileInput,
)
from intergrax.tools.providers.workspace.handlers import (
    WorkspaceListFilesHandler,
    WorkspaceReadFileHandler,
    WorkspaceSnapshotHandler,
    WorkspaceWriteFileHandler,
)
from intergrax.tools.providers.workspace.service import (
    WORKSPACE_LIST_FILES_TOOL_ID,
    WORKSPACE_READ_FILE_TOOL_ID,
    WORKSPACE_SNAPSHOT_TOOL_ID,
    WORKSPACE_WRITE_FILE_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

WORKSPACE_BUNDLE_ID = "workspace"
WORKSPACE_TOOL_IDS: tuple[str, ...] = (
    WORKSPACE_WRITE_FILE_TOOL_ID,
    WORKSPACE_READ_FILE_TOOL_ID,
    WORKSPACE_LIST_FILES_TOOL_ID,
    WORKSPACE_SNAPSHOT_TOOL_ID,
)


def register_workspace_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=WORKSPACE_WRITE_FILE_TOOL_ID,
            name=WORKSPACE_WRITE_FILE_TOOL_ID,
            description="Write a UTF-8 text file into the isolated shadow workspace for this task.",
            description_short="Write shadow workspace file.",
            input_schema=WorkspaceWriteFileInput,
            output_schema=WorkspaceArtifactOutput,
            error_mapping={},
            side_effects=True,
            category="workspace",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("workspace", "shadow", "filesystem"),
        ),
        WorkspaceWriteFileHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=WORKSPACE_READ_FILE_TOOL_ID,
            name=WORKSPACE_READ_FILE_TOOL_ID,
            description="Read a UTF-8 text file from the shadow workspace.",
            description_short="Read shadow workspace file.",
            input_schema=WorkspaceReadFileInput,
            output_schema=WorkspaceReadFileOutput,
            error_mapping={},
            side_effects=False,
            category="workspace",
            risk_level=ToolRiskLevel.LOW,
            tags=("workspace", "shadow", "filesystem"),
        ),
        WorkspaceReadFileHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=WORKSPACE_LIST_FILES_TOOL_ID,
            name=WORKSPACE_LIST_FILES_TOOL_ID,
            description="List artifact files currently stored in the shadow workspace.",
            description_short="List shadow workspace files.",
            input_schema=WorkspaceListFilesInput,
            output_schema=WorkspaceListFilesOutput,
            error_mapping={},
            side_effects=False,
            category="workspace",
            risk_level=ToolRiskLevel.LOW,
            tags=("workspace", "shadow", "filesystem"),
        ),
        WorkspaceListFilesHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=WORKSPACE_SNAPSHOT_TOOL_ID,
            name=WORKSPACE_SNAPSHOT_TOOL_ID,
            description="Capture a point-in-time snapshot of all shadow workspace files.",
            description_short="Snapshot shadow workspace.",
            input_schema=WorkspaceSnapshotInput,
            output_schema=WorkspaceSnapshotOutput,
            error_mapping={},
            side_effects=False,
            category="workspace",
            risk_level=ToolRiskLevel.LOW,
            tags=("workspace", "shadow", "snapshot"),
        ),
        WorkspaceSnapshotHandler(ctx),
    )
