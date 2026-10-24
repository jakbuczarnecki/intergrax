# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional bounded relational row fetch — physical consumption ceiling at driver cursor."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class RelationalStoreBoundedQueryPort(Protocol):
    """
    Incremental SQL row consumption with row/byte ceilings.

    Implementations must stop fetching once limits are reached (no full-result materialization).
    """

    def fetch_query_bounded(
        self,
        sql: str,
        params: Sequence[Any] = (),
        *,
        max_rows: int,
        max_result_bytes: int,
    ) -> tuple[tuple[Mapping[str, Any], ...], bool, str]:
        """Return ``(rows, truncated, limit_reason)`` with limit_reason ``row_limit`` | ``byte_limit`` | ````."""


__all__ = ["RelationalStoreBoundedQueryPort"]
