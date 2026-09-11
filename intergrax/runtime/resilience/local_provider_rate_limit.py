# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Process-local provider rate limiting (W2-C)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import DefaultDict

from intergrax.contracts.provider_rate_limit import (
    ProviderRateLimitExceededError,
    ProviderRateLimitIdentity,
    ProviderRateLimitOverloadMode,
    ProviderRateLimitPolicy,
    ProviderRateLimitPolicyMissingError,
    ProviderRateLimitPort,
    ProviderRateLimitWaitTimeoutError,
)


@dataclass
class _RateWindowState:
    call_timestamps: deque[float] = field(default_factory=deque)


class LocalProviderRateLimit(ProviderRateLimitPort):
    """Fixed-window calls-per-minute limiter per ``ProviderRateLimitIdentity``."""

    __slots__ = ("_lock", "_policies", "_states")

    def __init__(
        self,
        policies: Mapping[ProviderRateLimitIdentity, ProviderRateLimitPolicy] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._states: DefaultDict[ProviderRateLimitIdentity, _RateWindowState] = defaultdict(
            _RateWindowState
        )
        self._policies: dict[ProviderRateLimitIdentity, ProviderRateLimitPolicy] = {}
        if policies is not None:
            for identity, policy in policies.items():
                self.register_policy(identity, policy)

    def register_policy(
        self,
        identity: ProviderRateLimitIdentity,
        policy: ProviderRateLimitPolicy,
    ) -> None:
        if not isinstance(identity, ProviderRateLimitIdentity):
            raise TypeError("identity must be ProviderRateLimitIdentity")
        if not isinstance(policy, ProviderRateLimitPolicy):
            raise TypeError("policy must be ProviderRateLimitPolicy")
        self._policies[identity] = policy

    def reset(self, identity: ProviderRateLimitIdentity | None = None) -> None:
        with self._lock:
            if identity is None:
                self._states.clear()
                return
            self._states.pop(identity, None)

    def acquire_for_physical_attempt(
        self,
        identity: ProviderRateLimitIdentity,
        policy: ProviderRateLimitPolicy,
    ) -> None:
        if not isinstance(identity, ProviderRateLimitIdentity):
            raise TypeError("identity must be ProviderRateLimitIdentity")
        if not isinstance(policy, ProviderRateLimitPolicy):
            raise TypeError("policy must be ProviderRateLimitPolicy")
        registered = self._policies.get(identity)
        if registered is not None:
            policy = registered
        limit = policy.calls_per_minute
        if limit < 1:
            raise ValueError("calls_per_minute must be >= 1")
        deadline: float | None = None
        if policy.overload_mode is ProviderRateLimitOverloadMode.WAIT_WITH_TIMEOUT:
            if policy.wait_timeout_seconds is None:
                raise ValueError("wait_timeout_seconds required for WAIT_WITH_TIMEOUT")
            deadline = time.monotonic() + float(policy.wait_timeout_seconds)
        while True:
            if self._try_consume(identity, limit):
                return
            if policy.overload_mode is ProviderRateLimitOverloadMode.REJECT:
                raise ProviderRateLimitExceededError("provider rate limit exceeded")
            if deadline is not None and time.monotonic() >= deadline:
                raise ProviderRateLimitWaitTimeoutError(
                    "provider rate limit wait timed out"
                )
            time.sleep(0.01)

    def _try_consume(self, identity: ProviderRateLimitIdentity, limit: int) -> bool:
        now = time.monotonic()
        window_start = now - 60.0
        with self._lock:
            st = self._states[identity]
            while st.call_timestamps and st.call_timestamps[0] < window_start:
                st.call_timestamps.popleft()
            if len(st.call_timestamps) >= limit:
                return False
            st.call_timestamps.append(now)
            return True

    def require_registered_policy(
        self,
        identity: ProviderRateLimitIdentity,
    ) -> ProviderRateLimitPolicy:
        policy = self._policies.get(identity)
        if policy is None:
            raise ProviderRateLimitPolicyMissingError(
                "no provider rate limit policy configured for identity"
            )
        return policy
