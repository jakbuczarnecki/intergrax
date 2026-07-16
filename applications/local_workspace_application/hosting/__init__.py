# © Artur Czarnecki. All rights reserved.

"""LKW Application Hosting adoption surface (APP-HOST-8A/8C)."""

from local_workspace_application.hosting.foreground import (
    run_local_workspace_hosted_application,
)
from local_workspace_application.hosting.profile import (
    build_local_workspace_hosted_profile,
)

__all__ = [
    "build_local_workspace_hosted_profile",
    "run_local_workspace_hosted_application",
]
