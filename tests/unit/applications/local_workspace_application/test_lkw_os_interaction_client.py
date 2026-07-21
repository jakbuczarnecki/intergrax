# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CLIENT = (
    _PROJECT_ROOT
    / "applications"
    / "local_workspace_application"
    / "scripts"
    / "invoke-lkw-interaction.py"
)
_PS1 = _CLIENT.with_name("invoke-lkw-interaction.ps1")
_LINUX_SH = _CLIENT.with_name("invoke-lkw-interaction-linux.sh")
_MACOS_SH = _CLIENT.with_name("invoke-lkw-interaction-macos.sh")


def _load_client():
    module_name = "lkw_os_interaction_client"
    spec = importlib.util.spec_from_file_location(module_name, _CLIENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _base_argv(**overrides: str) -> list[str]:
    values = {
        "--os-family": "windows",
        "--adapter-id": "lkw.windows_powershell",
        "--source": "windows_powershell",
        "--wrapper-runtime": "windows_powershell",
        "--message": "hello",
        "--tenant-id": "t1",
        "--user-id": "u1",
        "--metadata-json": "{}",
        "--timeout-seconds": "5",
        "--base-url": "http://127.0.0.1:8020",
    }
    values.update({f"--{k.replace('_', '-')}": v for k, v in overrides.items()})
    argv: list[str] = []
    for key, value in values.items():
        argv.extend([key, value])
    return argv


def test_accepted_os_contracts() -> None:
    client = _load_client()
    for family, adapter, source, wrapper in (
        (
            "windows",
            "lkw.windows_powershell",
            "windows_powershell",
            "windows_powershell",
        ),
        ("linux", "lkw.linux_shell", "linux_shell", "posix_sh"),
        ("macos", "lkw.macos_shell", "macos_shell", "posix_sh"),
    ):
        contract = client.validate_os_contract(
            os_family=family,
            adapter_id=adapter,
            source=source,
            wrapper_runtime=wrapper,
        )
        assert contract.os_family == family
        assert contract.adapter_id == adapter


@pytest.mark.parametrize(
    ("os_family", "adapter_id", "source", "wrapper_runtime"),
    [
        ("windows", "lkw.linux_shell", "windows_powershell", "windows_powershell"),
        ("linux", "lkw.windows_powershell", "linux_shell", "posix_sh"),
        ("macos", "lkw.macos_shell", "linux_shell", "posix_sh"),
        ("windows", "lkw.windows_powershell", "windows_powershell", "posix_sh"),
        ("linux", "lkw.linux_shell", "linux_shell", "windows_powershell"),
        (
            "unknown",
            "lkw.windows_powershell",
            "windows_powershell",
            "windows_powershell",
        ),
        ("windows", "", "windows_powershell", "windows_powershell"),
        ("windows", "lkw.windows_powershell", "", "windows_powershell"),
        ("windows", "lkw.windows_powershell", "windows_powershell", ""),
    ],
)
def test_os_contract_mismatch_fails_closed(
    os_family: str,
    adapter_id: str,
    source: str,
    wrapper_runtime: str,
) -> None:
    client = _load_client()
    with pytest.raises(client.ClientError) as exc_info:
        client.validate_os_contract(
            os_family=os_family,
            adapter_id=adapter_id,
            source=source,
            wrapper_runtime=wrapper_runtime,
        )
    assert exc_info.value.error_id in {
        client.ERROR_OS_CONTRACT_MISMATCH,
        client.ERROR_UNSUPPORTED_OS_FAMILY,
        client.ERROR_INVALID_ADAPTER_INPUT,
    }
    assert exc_info.value.exit_code == client.EXIT_INVALID_INPUT


def test_unknown_runtime_os_fails_closed() -> None:
    client = _load_client()
    with pytest.raises(client.ClientError) as exc_info:
        client.detect_runtime_os_family("FreeBSD")
    assert exc_info.value.error_id == client.ERROR_UNSUPPORTED_OS_FAMILY


def test_runtime_os_mismatch_fails_closed() -> None:
    client = _load_client()
    with pytest.raises(client.ClientError) as exc_info:
        client.validate_runtime_os_matches_declared(
            "linux", runtime_os_family="windows"
        )
    assert exc_info.value.error_id == client.ERROR_RUNTIME_OS_MISMATCH


def test_metadata_validation() -> None:
    client = _load_client()
    assert client.parse_metadata_json('{"a":1}') == {"a": 1}
    assert client.parse_metadata_json("") == {}
    for bad in ("[]", '"x"', "1", "null", "true", "{bad"):
        with pytest.raises(client.ClientError) as exc_info:
            client.parse_metadata_json(bad)
        assert exc_info.value.error_id == client.ERROR_INVALID_ADAPTER_INPUT


def test_url_trailing_slash_and_tenant_encoding() -> None:
    client = _load_client()
    assert client.resolve_base_url("http://127.0.0.1:8020/") == "http://127.0.0.1:8020"
    url = client.build_intake_url(
        base_url="http://127.0.0.1:8020",
        tenant_id="tenant a/b",
    )
    assert url.startswith("http://127.0.0.1:8020/v1/interactions/intake?")
    assert "execute=true" in url
    assert "tenant=tenant%20a%2Fb" in url


def test_http_not_called_after_contract_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _load_client()
    called = {"http": False}

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        called["http"] = True
        raise AssertionError("http should not be called")

    monkeypatch.setattr(client, "post_interaction_intake", _boom)
    monkeypatch.setattr(
        client,
        "validate_runtime_os_matches_declared",
        lambda *_a, **_k: "windows",
    )
    argv = _base_argv()
    idx = argv.index("--adapter-id")
    argv[idx + 1] = "lkw.linux_shell"
    code = client.run_client(argv)
    assert code == client.EXIT_INVALID_INPUT
    assert called["http"] is False


def test_success_and_failure_json_contracts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _load_client()
    monkeypatch.setattr(
        client,
        "validate_runtime_os_matches_declared",
        lambda *_a, **_k: "windows",
    )
    monkeypatch.setattr(
        client,
        "post_interaction_intake",
        lambda **_kwargs: {"state": "completed", "task_id": "t1"},
    )
    code = client.run_client(_base_argv())
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out.strip())
    assert payload["schema_version"] == client.SCHEMA_VERSION
    assert payload["result"] == "PASS"
    assert payload["client_runtime"] == "python"
    assert payload["wrapper_runtime"] == "windows_powershell"
    assert payload["endpoint"] == "/v1/interactions/intake"
    assert payload["execute"] is True
    assert "message" not in payload
    assert "metadata" not in payload
    assert "password" not in json.dumps(payload).lower()

    monkeypatch.setattr(
        client,
        "post_interaction_intake",
        lambda **_kwargs: (_ for _ in ()).throw(
            client.ClientError(
                client.ERROR_INTERACTION_REQUEST_FAILED,
                exit_code=client.EXIT_REQUEST_FAILED,
                http_status=500,
            )
        ),
    )
    code = client.run_client(_base_argv())
    captured = capsys.readouterr()
    assert code == client.EXIT_REQUEST_FAILED
    fail = json.loads(captured.err.strip())
    assert fail["result"] == "FAIL"
    assert fail["error_id"] == client.ERROR_INTERACTION_REQUEST_FAILED
    assert fail["http_status"] == 500


def test_timeout_and_invalid_response_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _load_client()

    class _TimeoutOpener:
        def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
            raise TimeoutError("timeout")

    monkeypatch.setattr(client.urllib.request, "urlopen", _TimeoutOpener())
    with pytest.raises(client.ClientError) as exc_info:
        client.post_interaction_intake(
            url="http://127.0.0.1:8020/v1/interactions/intake?execute=true&tenant=t",
            payload={"message": "x"},
            timeout_seconds=1,
        )
    assert exc_info.value.error_id == client.ERROR_INTERACTION_REQUEST_FAILED

    class _BadJsonResponse:
        status = 200

        def read(self) -> bytes:
            return b"not-json"

        def __enter__(self) -> _BadJsonResponse:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(
        client.urllib.request,
        "urlopen",
        lambda *_a, **_k: _BadJsonResponse(),
    )
    with pytest.raises(client.ClientError) as exc_info:
        client.post_interaction_intake(
            url="http://127.0.0.1:9/v1/interactions/intake?execute=true&tenant=t",
            payload={"message": "x"},
            timeout_seconds=1,
        )
    assert exc_info.value.error_id == client.ERROR_INTERACTION_RESPONSE_INVALID


def test_http_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _load_client()

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise HTTPError(
            "http://example",
            503,
            "unavailable",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr(client.urllib.request, "urlopen", _raise)
    with pytest.raises(client.ClientError) as exc_info:
        client.post_interaction_intake(
            url="http://127.0.0.1:9/v1/interactions/intake?execute=true&tenant=t",
            payload={"message": "x"},
            timeout_seconds=1,
        )
    assert exc_info.value.http_status == 503

    def _url_error(*_args: Any, **_kwargs: Any) -> Any:
        raise URLError("down")

    monkeypatch.setattr(client.urllib.request, "urlopen", _url_error)
    with pytest.raises(client.ClientError) as exc_info:
        client.post_interaction_intake(
            url="http://127.0.0.1:9/v1/interactions/intake?execute=true&tenant=t",
            payload={"message": "x"},
            timeout_seconds=1,
        )
    assert exc_info.value.error_id == client.ERROR_INTERACTION_REQUEST_FAILED


def test_wrappers_are_thin_and_delegate() -> None:
    ps1 = _PS1.read_text(encoding="utf-8")
    assert "invoke-lkw-interaction.py" in ps1
    for forbidden in (
        "Invoke-RestMethod",
        "Invoke-WebRequest",
        "ConvertTo-Json",
        "ConvertFrom-Json",
        "Invoke-Expression",
        "Start-Process",
    ):
        assert forbidden not in ps1
    assert "--os-family" in ps1
    assert "windows" in ps1
    assert "lkw.windows_powershell" in ps1

    for path, family, adapter in (
        (_LINUX_SH, "linux", "lkw.linux_shell"),
        (_MACOS_SH, "macos", "lkw.macos_shell"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "invoke-lkw-interaction.py" in text
        assert f"--os-family {family}" in text
        assert adapter in text
        for forbidden in (
            "curl",
            "wget",
            "jq",
            "docker",
            "ProofReceipt",
            "ConvertTo-Json",
        ):
            assert forbidden not in text


def test_shared_client_does_not_invoke_os_wrappers() -> None:
    text = _CLIENT.read_text(encoding="utf-8")
    assert "invoke-lkw-interaction.ps1" not in text
    assert "invoke-lkw-interaction-linux.sh" not in text
    assert "invoke-lkw-interaction-macos.sh" not in text
    assert "shell=True" not in text
    assert "requests" not in text
    assert "httpx" not in text
    assert "Invoke-RestMethod" not in text
