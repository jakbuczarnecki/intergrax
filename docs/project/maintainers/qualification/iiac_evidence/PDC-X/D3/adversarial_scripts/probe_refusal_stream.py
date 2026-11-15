from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
U=[ChatMessage(role="user",content="x")]
def chunk(content=None, refusal=None, finish=None):
    return NS(choices=[NS(delta=NS(content=content, refusal=refusal, tool_calls=None), finish_reason=finish)], usage=None)
cases={
 "stream_refusal_delta":[chunk(refusal="I can't help with that."), chunk(finish="stop")],
 "stream_content_filter":[chunk(finish="content_filter")],
}
for name,chunks in cases.items():
    client=MagicMock()
    client.chat.completions.create.return_value=iter(chunks)
    a=OpenAIChatCompletionsAdapter(client=client, model="gpt-4o-mini", provider=LLMProvider.OPENAI)
    evs=list(a.stream_messages(U))
    fin=[e for e in evs if e.is_final][-1].response
    print(name, "finish=",fin.finish_reason, "refusal=",fin.refusal, "content=",repr(fin.content))
