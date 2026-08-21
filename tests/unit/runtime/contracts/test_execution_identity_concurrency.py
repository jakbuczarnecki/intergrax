# © Artur Czarnecki. All rights reserved.

"""Concurrency isolation for execution identity (full gate only — not CI smoke)."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    require_active_execution_identity,
    reset_active_execution_identity,
    transition_active_execution_identity,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_concurrent_execution_identity_isolation() -> None:
    run_r1 = mint_run_id()
    run_r2 = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_b1 = mint_attempt_id()
    gate = asyncio.Event()
    results: dict[str, tuple[RunId, AttemptId]] = {}

    async def coroutine_a() -> None:
        token = bind_active_execution_identity(run_id=run_r1, attempt_id=attempt_a1)
        try:
            gate.set()
            await asyncio.sleep(0.05)
            results["a"] = require_active_execution_identity()
        finally:
            reset_active_execution_identity(token)

    async def coroutine_b() -> None:
        await gate.wait()
        token = bind_active_execution_identity(run_id=run_r2, attempt_id=attempt_b1)
        try:
            await asyncio.sleep(0.05)
            results["b"] = require_active_execution_identity()
        finally:
            reset_active_execution_identity(token)

    await asyncio.gather(coroutine_a(), coroutine_b())
    assert results["a"] == (run_r1, attempt_a1)
    assert results["b"] == (run_r2, attempt_b1)


@pytest.mark.asyncio
async def test_concurrent_retry_transition_isolation() -> None:
    run_r1 = mint_run_id()
    run_r2 = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_b1 = mint_attempt_id()
    gate = asyncio.Event()
    results: dict[str, AttemptId | tuple[RunId, AttemptId]] = {}

    async def coroutine_a() -> None:
        token = bind_active_execution_identity(run_id=run_r1, attempt_id=attempt_a1)
        try:
            gate.set()
            await asyncio.sleep(0.05)
            results["a"] = transition_active_execution_identity()
        finally:
            reset_active_execution_identity(token)

    async def coroutine_b() -> None:
        await gate.wait()
        token = bind_active_execution_identity(run_id=run_r2, attempt_id=attempt_b1)
        try:
            await asyncio.sleep(0.05)
            results["b_before"] = require_active_execution_identity()[1]
            await asyncio.sleep(0.05)
            results["b_after"] = require_active_execution_identity()[1]
        finally:
            reset_active_execution_identity(token)

    await asyncio.gather(coroutine_a(), coroutine_b())
    assert results["a"] != attempt_a1
    assert results["b_before"] == attempt_b1
    assert results["b_after"] == attempt_b1
