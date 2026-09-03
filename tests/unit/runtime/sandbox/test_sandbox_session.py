# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_TOOL_NAME
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.core.contracts import ToolIsolationRequirement
from intergrax.tools.providers.sandbox.bundle import sandbox_exec_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_requires_sandbox_tool_uses_contract_not_tool_id() -> None:
    contract = sandbox_exec_contract()
    assert contract.tool_id == SANDBOX_TOOL_NAME
    assert contract.isolation_requirement is ToolIsolationRequirement.SANDBOX
    assert contract.requires_sandbox_isolation is True


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
