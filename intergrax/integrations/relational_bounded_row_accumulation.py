# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Incremental relational row accumulation with row/UTF-8 JSON byte ceilings."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping


class RelationalResultByteLimitExceeded(RuntimeError):
    def __init__(self, code: str, *, detail: str = "") -> None:
        self.code = code
        super().__init__(detail or code)


def _estimate_rows_json_bytes(rows: list[dict[str, Any]]) -> int:
    return len(json.dumps(rows, default=str, separators=(",", ":")).encode("utf-8"))


def accumulate_bounded_sql_rows(
    row_iter: Iterable[Mapping[str, Any]],
    *,
    max_rows: int,
    max_result_bytes: int,
) -> tuple[list[dict[str, Any]], bool, str]:
    if max_rows < 1 or max_result_bytes < 1:
        raise RelationalResultByteLimitExceeded("invalid_bounds")

    accumulated: list[dict[str, Any]] = []
    truncated = False
    reason = ""

    for row in row_iter:
        if len(accumulated) >= max_rows:
            truncated = True
            reason = "row_limit"
            break
        dict_row = dict(row)
        trial = accumulated + [dict_row]
        if _estimate_rows_json_bytes(trial) > max_result_bytes:
            if not accumulated:
                raise RelationalResultByteLimitExceeded(
                    "database_result_byte_limit",
                    detail="single_row_exceeds_max_result_bytes",
                )
            truncated = True
            reason = "byte_limit"
            break
        accumulated.append(dict_row)

    return accumulated, truncated, reason


__all__ = [
    "RelationalResultByteLimitExceeded",
    "accumulate_bounded_sql_rows",
]
