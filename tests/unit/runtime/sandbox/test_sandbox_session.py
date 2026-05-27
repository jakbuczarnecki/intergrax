# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import (
    SANDBOX_TOOL_NAME,
    requires_sandbox_tool,
)
from intergrax.runtime.sandbox.session import SandboxSession

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_requires_sandbox_tool():
    assert requires_sandbox_tool(SANDBOX_TOOL_NAME) is True
    assert requires_sandbox_tool("echo.basic") is False


def test_sandbox_session_echo_and_files(tmp_path):
    session = SandboxSession.create(
        tmp_path,
        tenant_id="t1",
        task_id="task-1",
    )
    echo = session.execute("echo", {"message": "hello sandbox"})
    assert echo.success is True
    assert echo.output["message"] == "hello sandbox"

    write = session.execute(
        "write_file",
        {"path": "notes/out.txt", "content": "stored safely"},
    )
    assert write.success is True

    read = session.execute("read_file", {"path": "notes/out.txt"})
    assert read.success is True
    assert read.output["content"] == "stored safely"

    listing = session.execute("list_files", {})
    assert listing.output["files"] == ["notes/out.txt"]
    assert len(session.audit_log) == 4


def test_sandbox_session_denies_unknown_operation(tmp_path):
    session = SandboxSession.create(tmp_path, tenant_id="t1", task_id="task-2")
    result = session.execute("run_shell", {"cmd": "rm -rf /"})
    assert result.success is False
    assert "operation_not_allowed" in (result.error or "")


def test_sandbox_session_cancel_is_interruptible(tmp_path):
    session = SandboxSession.create(tmp_path, tenant_id="t1", task_id="task-3")
    session.cancel()
    result = session.execute("echo", {"message": "blocked"})
    assert result.success is False
    assert result.error == "sandbox_cancelled"


def test_sandbox_session_manager_cleanup(tmp_path):
    manager = SandboxSessionManager(root=tmp_path)
    session = manager.open_or_create(tenant_id="t1", task_id="task-4")
    session.execute("write_file", {"path": "a.txt", "content": "x"})
    assert session.exists_on_disk()

    assert manager.cleanup(session.session_id) is True
    assert not session.exists_on_disk()
