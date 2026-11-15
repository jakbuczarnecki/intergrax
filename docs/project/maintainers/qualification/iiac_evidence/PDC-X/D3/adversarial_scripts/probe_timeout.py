import time
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
U=[ChatMessage(role="user",content="x")]
def slow(**kw):
    time.sleep(1.0)
    from openai.types.chat import ChatCompletion
    return ChatCompletion.model_validate({"id":"c1","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"late","refusal":None}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})
client=MagicMock(); client.chat.completions.create.side_effect=slow
a=OpenAIChatCompletionsAdapter(client=client, model="gpt-4o-mini", provider=LLMProvider.OPENAI, timeout_sec=0.1, max_retries=0)
print("call_config.timeout_sec=", a.call_config.timeout_sec)
t=time.time()
try:
    r=a.generate_messages(U); print("returned", repr(r.content), "after", round(time.time()-t,2),"s")
except BaseException as e:
    print("raised", type(e).__name__, "after", round(time.time()-t,2))
print("sdk call kwargs:", {k:v for k,v in client.chat.completions.create.call_args.kwargs.items() if "time" in k})
