import pytest
from intergrax.fastapi_core.runs.models import RunStatus
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from testing_support.builder import DummyRunStore

pytestmark = pytest.mark.unit

def test_cancel_run_transitions_running_to_canceled() -> None:
    store = DummyRunStore()
    service = DefaultRunService(store=store, execution_adapter=None)

    run = store.create()
    store.update_status(run.run_id, RunStatus.RUNNING)

    service.cancel_run(run.run_id)

    final = store.get(run.run_id)
    assert final.status == RunStatus.CANCELED


def test_cancel_run_fails_if_not_running() -> None:
    store = DummyRunStore()
    service = DefaultRunService(store=store, execution_adapter=None)

    run = store.create()
    store.update_status(run.run_id, RunStatus.COMPLETED)

    try:
        service.cancel_run(run.run_id)
    except Exception:
        pass
    else:
        assert False, "Cancel should fail for non-running run"
