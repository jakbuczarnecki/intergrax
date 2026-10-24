# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded HTTP response body accumulation at transport read boundary."""

from __future__ import annotations


def read_response_text_bounded(
    chunks: bytes,
    *,
    max_response_bytes: int,
) -> tuple[str, bool]:
    if len(chunks) <= max_response_bytes:
        return chunks.decode("utf-8", errors="replace"), False
    truncated_bytes = chunks[:max_response_bytes]
    return truncated_bytes.decode("utf-8", errors="ignore"), True


def accumulate_response_bytes(
    chunk_iter,
    *,
    max_response_bytes: int,
) -> tuple[bytes, bool]:
    if max_response_bytes < 1:
        raise ValueError("invalid_bounds")
    body = bytearray()
    truncated = False
    for chunk in chunk_iter:
        if not chunk:
            continue
        remaining = max_response_bytes - len(body)
        if remaining <= 0:
            truncated = True
            break
        if len(chunk) > remaining:
            body.extend(chunk[:remaining])
            truncated = True
            break
        body.extend(chunk)
    return bytes(body), truncated


__all__ = ["accumulate_response_bytes", "read_response_text_bounded"]
