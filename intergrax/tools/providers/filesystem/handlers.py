# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.filesystem.contracts import (
    FilesystemGlobInput,
    FilesystemGlobOutput,
    FilesystemListInput,
    FilesystemListOutput,
    FilesystemReadTextInput,
    FilesystemReadTextOutput,
    FilesystemStatInput,
    FilesystemStatOutput,
)
from intergrax.tools.providers.filesystem.service import (
    filesystem_glob,
    filesystem_list,
    filesystem_read_text,
    filesystem_stat,
)


class FilesystemListHandler(ServiceToolHandler[FilesystemListInput, FilesystemListOutput]):
    _service = filesystem_list


class FilesystemGlobHandler(ServiceToolHandler[FilesystemGlobInput, FilesystemGlobOutput]):
    _service = filesystem_glob


class FilesystemReadTextHandler(ServiceToolHandler[FilesystemReadTextInput, FilesystemReadTextOutput]):
    _service = filesystem_read_text


class FilesystemStatHandler(ServiceToolHandler[FilesystemStatInput, FilesystemStatOutput]):
    _service = filesystem_stat
