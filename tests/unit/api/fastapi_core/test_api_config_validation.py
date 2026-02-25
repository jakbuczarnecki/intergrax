import pytest

from intergrax.fastapi_core.config import ApiConfig

pytestmark = pytest.mark.unit

def test_config_rejects_run_service_and_execution_adapter():
    config = ApiConfig(
        api_prefix="/",
        run_service=object(),
        execution_adapter=object(),
    )

    with pytest.raises(ValueError):
        config.validate()
