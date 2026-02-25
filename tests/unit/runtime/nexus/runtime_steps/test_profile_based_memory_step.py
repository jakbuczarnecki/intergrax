# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from intergrax.runtime.nexus.runtime_steps.profile_based_memory_step import ProfileBasedMemoryStep
from tests._support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


class _FakeSessionManager:
    def __init__(self, user_instr=None, org_instr=None):
        self.user_instr = user_instr
        self.org_instr = org_instr

    async def get_user_profile_instructions_for_session(self, **kwargs):
        return self.user_instr

    async def get_org_profile_instructions_for_session(self, **kwargs):
        return self.org_instr


@pytest.mark.asyncio
async def test_profile_step_requires_session():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = None

    with pytest.raises(AssertionError, match=r"Session must exist before memory layer\."):
        await ProfileBasedMemoryStep().run(state)


@pytest.mark.asyncio
async def test_profile_step_disabled_flags_no_use():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {})()
    state.context.config.enable_user_profile_memory = False
    state.context.config.enable_org_profile_memory = False
    state.context.session_manager = _FakeSessionManager(
        user_instr="USER",
        org_instr="ORG"
    )

    await ProfileBasedMemoryStep().run(state)

    assert state.profile_user_instructions is None
    assert state.profile_org_instructions is None
    assert getattr(state, "used_user_profile", False) is False


@pytest.mark.asyncio
async def test_profile_step_user_profile_used():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {})()
    state.context.config.enable_user_profile_memory = True
    state.context.config.enable_org_profile_memory = False
    state.context.session_manager = _FakeSessionManager(
        user_instr="  USER PROFILE  ",
        org_instr=None
    )

    await ProfileBasedMemoryStep().run(state)

    assert state.profile_user_instructions == "USER PROFILE"
    assert state.profile_org_instructions is None
    assert state.used_user_profile is True


@pytest.mark.asyncio
async def test_profile_step_org_profile_used():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {})()
    state.context.config.enable_user_profile_memory = False
    state.context.config.enable_org_profile_memory = True
    state.context.session_manager = _FakeSessionManager(
        user_instr=None,
        org_instr=" ORG PROFILE "
    )

    await ProfileBasedMemoryStep().run(state)

    assert state.profile_user_instructions is None
    assert state.profile_org_instructions == "ORG PROFILE"
    assert state.used_user_profile is True


@pytest.mark.asyncio
async def test_profile_step_both_profiles_used():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {})()
    state.context.config.enable_user_profile_memory = True
    state.context.config.enable_org_profile_memory = True
    state.context.session_manager = _FakeSessionManager(
        user_instr="USER",
        org_instr="ORG"
    )

    await ProfileBasedMemoryStep().run(state)

    assert state.profile_user_instructions == "USER"
    assert state.profile_org_instructions == "ORG"
    assert state.used_user_profile is True


@pytest.mark.asyncio
async def test_profile_step_empty_strings_not_used():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {})()
    state.context.config.enable_user_profile_memory = True
    state.context.config.enable_org_profile_memory = True
    state.context.session_manager = _FakeSessionManager(
        user_instr="   ",
        org_instr="  "
    )

    await ProfileBasedMemoryStep().run(state)

    assert state.profile_user_instructions is None
    assert state.profile_org_instructions is None    
