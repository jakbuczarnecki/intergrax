# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.tools.providers.gitlab.bundle import GITLAB_BUNDLE_ID, register_gitlab_tools
from intergrax.tools.registry.catalog import ToolBundleEntry, ToolBundleStatus, register_tool_bundle


def register_gitlab_tool_bundle(*, override: bool = False) -> None:
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=GITLAB_BUNDLE_ID,
            tool_ids=("gitlab.create_issue",),
            register=register_gitlab_tools,
            status=ToolBundleStatus.BETA,
            description="GitLab issue tracker tools.",
        ),
        override=override,
    )
