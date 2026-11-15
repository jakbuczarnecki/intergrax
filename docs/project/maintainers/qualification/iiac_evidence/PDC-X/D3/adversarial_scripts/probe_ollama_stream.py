from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from intergrax.llm.messages import ChatMessage
try:
    from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
except Exception as e:
    print("IMPORT FAIL", type(e).__name__, e); raise SystemExit
from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver
U=[ChatMessage(role="user",content="x")]
def gen():
    yield NS(content="par"); yield NS(content="tial")
    raise ConnectionError("transport closed by cancel")
chat=MagicMock(); chat.stream.return_value=gen(); chat.invoke.return_value=NS(content="FULL REISSUED ANSWER")
res=OllamaModelCapabilityResolver(show_model=lambda m: NS(capabilities=["tools","completion"]))
a=LangChainOllamaAdapter(chat=chat, model="llama3.1:latest", capability_resolver=res)
evs=list(a.stream_messages(U))
print("invoke_calls=", chat.invoke.call_count)
print([ (e.kind.value, e.delta_content) for e in evs if not e.is_final], "FINAL:", [e.response.content for e in evs if e.is_final])
