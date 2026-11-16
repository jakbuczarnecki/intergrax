from types import SimpleNamespace as NS
from unittest.mock import MagicMock
import intergrax.runtime.execution.deadline_provider_guard as g
import intergrax.llm_adapters.governance.quota as q
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter
from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
U=[ChatMessage(role="user",content="x")]
class Denied(RuntimeError): pass
def deny(*a,**k): raise Denied("protected-work admission denied")
g.assert_protected_provider_call_allowed = deny
def stream_ctx(**kw):
    class C:
        def __enter__(s): return iter([NS(type="response.output_text.delta", delta="hi")])
        def __exit__(s,*a): return False
    return C()
def run(label, fn):
    try:
        out=fn(); print(f"{label:55} -> ALLOWED ({out})")
    except Denied: print(f"{label:55} -> denied by protected-work guard")
    except Exception as e: print(f"{label:55} -> {type(e).__name__}: {str(e)[:70]}")
# Responses adapter
c=MagicMock(); c.responses.stream.side_effect=stream_ctx
a=OpenAIChatResponsesAdapter(client=c, model="gpt-4o-mini")
run("openai_responses.stream_messages", lambda: [e.kind.value for e in a.stream_messages(U)])
c2=MagicMock(); a2=OpenAIChatResponsesAdapter(client=c2, model="gpt-4o-mini")
run("openai_responses.generate_messages", lambda: a2.generate_messages(U).content)
# Chat completions adapter streaming (uses _execute_streaming)
c3=MagicMock(); c3.chat.completions.create.return_value=iter([NS(choices=[NS(delta=NS(content="x",refusal=None,tool_calls=None),finish_reason=None)],usage=None)])
a3=OpenAIChatCompletionsAdapter(client=c3, model="gpt-4o-mini", provider=LLMProvider.OPENAI)
run("openai_chat.stream_messages (control)", lambda: [e.kind.value for e in a3.stream_messages(U)])
