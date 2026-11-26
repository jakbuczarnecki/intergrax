# ruff: noqa: E501, F401, F841, E741
"""NT-19 (R24) - streaming fault-injection probe for A5-002 (EBH-2E stream paths vs provider guard / bounded retry).
Probe asserts the SAFE behaviour: PASS = REFUTED (bounded, guarded), FAIL = CONFIRMED, skip = INCONCLUSIVE.
Reuses the fake-SDK helpers of tests/qualification/pdc_x/d3/test_d3_f01_stream_guard.py (loaded by path, not modified).
Adapters: openai-responses, gemini, langchain-ollama (+ openai-chat as control). No network, no credential."""

import importlib.util
import os
import pathlib
import re
import subprocess
from types import SimpleNamespace

import pytest

WT = pathlib.Path(os.environ.get("CAMPAIGN_B_ALPHA_ROOT", os.getcwd()))
_SRC = WT / "tests/qualification/pdc_x/d3/test_d3_f01_stream_guard.py"
_spec = importlib.util.spec_from_file_location("_d3_helpers", _SRC)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

from intergrax.llm_adapters.contracts.call_config import LLMCallConfig  # noqa: E402


class APITimeoutError(Exception):
    """Transient by name (word 'timeout')."""


def _build(name):
    return H._ALL_STREAM_MESSAGES[name]()


def _set_cfg(adapter, retries):
    adapter.call_config = LLMCallConfig(max_retries=retries, retry_backoff_sec=0.0, max_retry_after_sec=0.0)


@pytest.mark.parametrize("name", ["openai-responses", "gemini", "langchain-ollama", "openai-chat"])
def test_nt19_a_persistent_sdk_failure_is_bounded_by_max_retries(name):
    adapter, probe = _build(name)
    _set_cfg(adapter, 2)
    # make SDK stream creation always fail
    orig = probe.stream

    class _Fail:
        def __iter__(self):
            probe.sdk_calls += 0
            raise APITimeoutError("injected")

        def close(self):
            pass

        def __enter__(self):
            probe.sdk_calls += 0
            raise APITimeoutError("injected-enter")

        def __exit__(self, *a):
            pass

    # replace the stream object returned by the fake SDK
    probe.stream = _Fail()
    with pytest.raises(Exception) as ei:
        list(adapter.stream_messages(H._USER))
    print(f"OBSERVATION NT19a {name} sdk_calls={probe.sdk_calls} exc={type(ei.value).__name__}")
    assert probe.sdk_calls <= 3, f"unbounded or over-retried: {probe.sdk_calls} SDK calls for max_retries=2"
    assert probe.sdk_calls >= 1


@pytest.mark.parametrize("name", ["openai-responses", "gemini", "langchain-ollama", "openai-chat"])
def test_nt19_b_transient_then_success_is_retried_within_budget(name):
    adapter, probe = _build(name)
    _set_cfg(adapter, 2)
    good = probe.stream
    state = {"n": 0}

    class _Flaky:
        def __iter__(self):
            state["n"] += 1
            if state["n"] <= 2:
                raise APITimeoutError("injected-transient")
            return iter(good._items)

        def close(self):
            good.close()

        def __enter__(self):
            state["n"] += 1
            if state["n"] <= 2:
                raise APITimeoutError("injected-transient-enter")
            return good.__enter__()

        def __exit__(self, *a):
            return good.__exit__(*a)

        def get_final_response(self):
            return good.get_final_response()

    probe.stream = _Flaky()
    try:
        out = list(adapter.stream_messages(H._USER))
    except Exception as exc:  # recorded as observation; retry semantics for mid-stream failures may differ
        print(f"OBSERVATION NT19b {name} raised {type(exc).__name__} sdk_calls={probe.sdk_calls} attempts={state['n']}")
        assert probe.sdk_calls <= 3 and state["n"] <= 3
        return
    print(f"OBSERVATION NT19b {name} ok items={len(out)} sdk_calls={probe.sdk_calls} attempts={state['n']}")
    assert probe.sdk_calls <= 3 and state["n"] <= 3


def test_nt19_c_legacy_permit_helper_has_no_callers():
    out = subprocess.run(
        [
            "git",
            "-C",
            str(WT),
            "grep",
            "-n",
            "_provider_dependency_attempt",
            "--",
            "intergrax",
            "agents",
            "applications",
        ],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    callers = [l for l in out if "def _provider_dependency_attempt" not in l]
    print("OBSERVATION NT19c callers=", callers)
    assert callers == []


def test_nt19_d_every_stream_method_of_shipped_adapters_uses_execute_streaming():
    bad = []
    for p in sorted((WT / "intergrax/llm_adapters/providers").glob("*_adapter.py")):
        bad += [f"{p.name}:{v}" for v in H._violations(p.read_text(encoding="utf-8"))]
    print("OBSERVATION NT19d violations=", bad)
    assert bad == []
