import asyncio, os, sys, time
from pathlib import Path
from unittest.mock import AsyncMock, patch
ROOT = Path(os.environ["AUDIT_ROOT"]); sys.path[:0] = [str(ROOT), str(ROOT / "applications")]
from fastmcp import Client
from fastmcp.exceptions import ToolError
from intergrax.applications._shared import mcp_host_execution_transport_cancel as tc
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.execution.root_execution_transport_detachment_supervision import build_host_root_execution_transport_detachment_supervisor
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from tests.qualification.mcp_server_x.test_mcp_server_x_p7_failure_injection_l2 import _boundary, _identity_resolution, _reference_server
NEXUS = "intergrax.applications._shared.mcp_nexus_server"

class RaisingHost:
    """On governed cancel the host coroutine ends by raising CancelledError (Task state: cancelled)."""
    def __init__(self, mode):
        self.mode = mode
        self.port = AsyncMock(spec=HostTaskExecutionPort); self.port.execute = AsyncMock(side_effect=self._exec)
    async def _exec(self, task, **_):
        run_id = mint_run_id(); await ActiveTaskRegistry.register(task, run_id)
        try:
            while not CancellationCoordinator.is_requested(task.metadata): await asyncio.sleep(0.02)
            if self.mode == "cancelled": raise asyncio.CancelledError()
            raise RuntimeError("SECRET-HOST-FAILURE-hunter2")
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, run_id)

async def one(mode):
    ActiveTaskRegistry.clear_for_tests()
    host = RaisingHost(mode); sup = build_host_root_execution_transport_detachment_supervisor()
    mcp = _reference_server(host.port, task_control_mutation_boundary=_boundary(), max_call_seconds=0.3)
    with patch(f"{NEXUS}.resolve_mcp_run_agent_identity", return_value=_identity_resolution()), \
         patch(f"{NEXUS}.resolve_transport_detachment_supervision_from_request", return_value=sup), \
         patch.object(tc, "_HOST_DRAIN_AFTER_TRANSPORT_CANCEL_S", 0.25):
        async with Client(mcp) as c:
            t0 = time.monotonic()
            try:
                r = await asyncio.wait_for(c.call_tool("run_agent", {"message": "x"}), timeout=10)
                print(mode, "-> RESULT (no error!)", str(r.data)[:150])
            except ToolError as e:
                print(mode, "-> ToolError:", str(e)[:200])
            except asyncio.TimeoutError:
                print(mode, "-> CLIENT TIMEOUT 10s (no response)")
            except BaseException as e:
                print(mode, "->", type(e).__name__, str(e)[:200])
            print(mode, "elapsed", round(time.monotonic() - t0, 2))
async def main():
    for m in ("cancelled", "failed"): await one(m)
asyncio.run(main())
