from collections.abc import Sequence
from typing import Any
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.llm_adapters.registry.failover_policy import default_failover_policy
from intergrax.runtime.cancellation.coordinator import CooperativeCancellationAbort
U=[ChatMessage(role="user",content="x")]
class A(BaseLLMAdapter):
    provider="p"; model="m"
    def __init__(self, exc=None, name="p"):
        super().__init__(); self.exc=exc; self.provider=name; self.model="m"; self.n=0
    @property
    def context_window_tokens(self): return 1000
    def generate_messages(self, messages, **_):
        self.n+=1
        if self.exc: raise self.exc
        return LLMAdapterResponse(content="ok", provider=self.provider, model="m")
def run(label, exc, policy=None):
    a=A(exc,"a"); b=A(None,"b")
    ch=FailoverLLMAdapter([a,b], failover_policy=policy or default_failover_policy())
    try:
        r=ch.generate_messages(U); out="returned "+r.content
    except BaseException as e:
        out="raised "+type(e).__name__
    print(f"{label:48} primary_calls={a.n} secondary_calls={b.n} -> {out}")
try: abort=CooperativeCancellationAbort("cancelled")
except TypeError: abort=CooperativeCancellationAbort()
run("CooperativeCancellationAbort (default policy)", abort)
run("KeyboardInterrupt (default policy)", KeyboardInterrupt())
class GenerateContentError(Exception): pass
run("class named GenerateContentError (bad request)", GenerateContentError("400 bad input"))
class SeparateTenantError(Exception): pass
run("class named SeparateTenantError", SeparateTenantError("x"))
class E429(Exception): status_code=429
run("status_code=429", E429("rl"))
class Auth(Exception): status_code=401
run("status_code=401", Auth("bad key"))
