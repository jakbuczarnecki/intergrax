# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Type, Union, Literal
import json
import time

# Your components
from intergrax.memory.conversational_memory import ConversationalMemory
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.tools.tools_agent import ToolsAgent, ToolsAgentConfig
from intergrax.tools.tools_base import ToolRegistry
from intergrax.rag.rag_answerer import RagAnswerer

Route = Literal["rag", "tools", "general"]

@dataclass
class ChatRouterConfig:
    """LLM router configuration (descriptive, not hard rules)."""
    use_llm_router: bool = True
    router_max_tokens: int = 256
    router_temperature: float = 0.0
    tools_description: str = (
        "TOOLS provide live, actionable operations such as calculations, "
        "summaries of numeric data, or calling external services/APIs."
    )
    general_description: str = (
        "GENERAL uses the base LLM without external tools or vector stores."
    )
    allow_override: bool = True

@dataclass
class RagComponent:
    """Single RAG endpoint with a description used for routing decisions."""
    name: str
    answerer: RagAnswerer
    description: str
    priority: int = 100

@dataclass
class ChatAgentConfig:
    """Global chat-agent settings."""
    temperature: float = 0.2
    max_answer_tokens: Optional[int] = None
    router: ChatRouterConfig = field(default_factory=ChatRouterConfig)
    pass_memory_to_general: bool = True

class ChatAgent:
    """
    One API “like ChatGPT”:
      - the model (LLM) decides the ROUTE (RAG / TOOLS / GENERAL),
      - for RAG the model chooses which RagComponent to use, based on descriptions,
      - accepts a ToolRegistry (lazy creation of intergraxToolsAgent),
      - handles memory, streaming, structured output,
      - returns a stable result: {answer, tool_traces, sources, summary, messages, output_structure, stats, route, rag_component}.
    """

    def __init__(
        self,
        llm: LLMAdapter,
        *,
        memory: Optional[ConversationalMemory] = None,
        tools: Optional[ToolRegistry] = None,
        tools_config: Optional[ToolsAgentConfig] = None,
        rag_components: Optional[Iterable[RagComponent]] = None,
        config: Optional[ChatAgentConfig] = None,
    ):
        self.llm = llm
        self.memory = memory
        self.cfg = config or ChatAgentConfig()

        self._tools_registry = tools
        self._tools_agent: Optional[ToolsAgent] = None
        self._tools_config = tools_config or ToolsAgentConfig()
        self._rag_catalog: List[RagComponent] = sorted(list(rag_components or []), key=lambda rc: rc.priority)

    # --------------------------
    #  Public API
    # --------------------------

    def run(
        self,
        question: str,
        *,
        stream: bool = False,
        force_route: Optional[Route] = None,
        allowed_tools: Optional[List[str]] = None,
        allowed_vectorstores: Optional[List[str]] = None,   # names of RagComponent
        output_model: Optional[Type] = None,
        summarize: bool = False,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,  # e.g., "auto"|"none"|OpenAI schema
        routing_context: Optional[str] = None,
        # tool usage policy
        tool_usage: Literal["auto", "required", "none"] = "auto",
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stable result shape:
        {
          "answer": str,
          "output_structure": Any|None,
          "tool_traces": list,
          "sources": list,
          "summary": str|None,
          "messages": list[ChatMessage],
          "stats": dict,
          "route": "rag"|"tools"|"general",
          "rag_component": str|None
        }
        """
        t0 = time.perf_counter()

        # store the question in memory
        if self.memory is not None:
            self.memory.add("user", question)

        # tool policy influences routing
        tools_enabled = not (tool_usage == "none")

        # 1) routing decision via LLM (with descriptions and tools_enabled flag)
        route, rag_name = self._decide_route_via_llm(
            question=question,
            force_route=force_route,
            allowed_vectorstores=allowed_vectorstores,
            routing_context=routing_context,
            tools_enabled=tools_enabled,
            run_id=run_id
        )

        # Enforce policy after router decision
        if route == "tools" and not tools_enabled:
            route, rag_name = "general", None

        if tool_usage == "required" and route != "tools":
            # user REQUIRES tools: if no tools → error,
            # if tools exist → try tools regardless of router decision
            if not self._tools_registry or len(self._tools_registry.list()) == 0:
                raise RuntimeError("No tools registered. Tool usage policy = 'required'. Aborting.")
            route, rag_name = "tools", None

        # 2) execution
        if route == "rag":
            result = self._do_rag(
                question=question,
                rag_name=rag_name,
                allowed_vectorstores=allowed_vectorstores,
                stream=stream,
                summarize=summarize,
                output_model=output_model,
                run_id=run_id
            )
            result["route"] = "rag"
            result["stats"] = {**result.get("stats", {}), "router_s": round(time.perf_counter() - t0, 4)}
            return result

        if route == "tools":
            result = self._do_tools(
                question=question,
                allowed_tools=allowed_tools,
                output_model=output_model,
                stream=stream,
                tool_choice=tool_choice or "auto",
                tool_usage=tool_usage,  # pass policy to _do_tools
                run_id=run_id,
            )
            result["route"] = "tools"
            result["stats"] = {**result.get("stats", {}), "router_s": round(time.perf_counter() - t0, 4)}
            result["rag_component"] = None
            return result

        # general
        result = self._do_general(
            question=question, 
            output_model=output_model, 
            stream=stream, 
            summarize=summarize, 
            run_id=run_id
        )
        result["route"] = "general"
        result["stats"] = {**result.get("stats", {}), "router_s": round(time.perf_counter() - t0, 4)}
        result["rag_component"] = None
        return result

    # --------------------------
    #  LLM routing
    # --------------------------

    def _decide_route_via_llm(
        self,
        *,
        question: str,
        force_route: Optional[Route],
        allowed_vectorstores: Optional[List[str]],
        routing_context: Optional[str],
        tools_enabled: bool,
        run_id: Optional[str] = None
    ) -> tuple[Route, Optional[str]]:
        router_cfg = self.cfg.router

        # ---- manual override ----
        if force_route and router_cfg.allow_override:
            if force_route == "rag":
                chosen = self._choose_rag_name_default(allowed_vectorstores)
                return "rag", chosen
            if force_route == "tools" and not tools_enabled:
                return "general", None
            return force_route, None

        if not router_cfg.use_llm_router:
            if allowed_vectorstores:
                return "rag", self._choose_rag_name_default(allowed_vectorstores)
            return "general", None

        # ---- Build component catalogs ----
        rag_options = [
            {"name": rc.name, "description": rc.description}
            for rc in self._rag_catalog
            if (not allowed_vectorstores or rc.name in allowed_vectorstores)
        ]

        tools_available = (
            tools_enabled
            and (self._tools_registry is not None)
            and (len(self._tools_registry.list()) > 0)
        )
        tool_options = []
        if tools_available:
            try:
                for t in self._tools_registry.list():
                    tool_options.append({
                        "name": getattr(t, "name", "<unnamed>"),
                        "description": getattr(t, "description", ""),
                    })
            except Exception:
                tool_options = []

        # ---- Router prompt ----
        sys_txt = (
            "You are a strict routing model responsible for choosing how to answer a user's query.\n"
            "Available routes:\n"
            "- RAG: use a vector-store retriever for company documents, policies, or structured internal knowledge.\n"
            f"- TOOLS: ({'ENABLED' if tools_enabled else 'DISABLED'}; AVAILABLE={len(tool_options) if tools_available else 0}) "
            f"use an agent equipped with specific function tools (e.g., calculations, weather lookup, external API calls). {router_cfg.tools_description}\n"
            f"- GENERAL: respond directly using only your own knowledge and reasoning. {router_cfg.general_description}\n\n"
            "Choose TOOLS only if at least one of the available tools is clearly relevant to the user's question.\n"
            "Your output must be a strict JSON object: {\"route\": \"RAG\"|\"TOOLS\"|\"GENERAL\", \"rag_component\": string|null}."
        )
        if routing_context:
            sys_txt += f"\nContext: {routing_context}"

        rag_catalog_txt = json.dumps(rag_options, ensure_ascii=False, indent=2)
        tools_catalog_txt = json.dumps(tool_options, ensure_ascii=False, indent=2)

        # ---- Few-shot examples to bias the model ----
        examples = [
            {
                "q": "What is the weather in Warsaw?",
                "explanation": "A weather-related question. The available tool get_weather is relevant -> TOOLS.",
                "out": {"route": "TOOLS", "rag_component": None},
            },
            {
                "q": "Calculate 25,000 net + 23% VAT.",
                "explanation": "A numerical or tax calculation -> TOOLS if a calculator-like tool exists.",
                "out": {"route": "TOOLS", "rag_component": None},
            },
            {
                "q": "What is intergrax's privacy policy?",
                "explanation": "A company policy or documentation request -> RAG (use intergrax_docs).",
                "out": {"route": "RAG", "rag_component": "intergrax_docs"},
            },
            {
                "q": "Tell me in general what intergrax is.",
                "explanation": "A general question not tied to any vector store or tool -> GENERAL.",
                "out": {"route": "GENERAL", "rag_component": None},
            },
        ]
        ex_txt = json.dumps(examples, ensure_ascii=False, indent=2)

        usr_txt = (
            "Decide which route should handle the following user query.\n\n"
            f"User query:\n{question}\n\n"
            f"RAG components:\n{rag_catalog_txt}\n\n"
            f"Available tools:\n{tools_catalog_txt}\n\n"
            "Routing rules:\n"
            "- If the question refers to company documentation, policies, or terms -> use RAG.\n"
            "- If the question asks for an operation that a listed tool can perform (e.g., weather, calculations, API lookup) -> use TOOLS.\n"
            "- Only choose TOOLS if at least one available tool is relevant to the question.\n"
            "- Otherwise, choose GENERAL.\n\n"
            f"Reference examples:\n{ex_txt}\n\n"
            "Output STRICT JSON ONLY, for example: {\"route\":\"RAG\",\"rag_component\":\"intergrax_docs\"}"
        )

        msgs = [
            ChatMessage(role="system", content=sys_txt),
            ChatMessage(role="user", content=usr_txt),
        ]

        raw = ""
        try:
            raw = self.llm.generate_messages(
                msgs,
                temperature=router_cfg.router_temperature,
                max_tokens=router_cfg.router_max_tokens,
                run_id=run_id
            ).strip()

            obj = json.loads(self._extract_json_block(raw))
            route = str(obj.get("route", "")).strip().lower()
            rag_name = obj.get("rag_component")

            if route not in ("rag", "tools", "general"):
                raise ValueError("invalid route")

            if route == "tools" and not tools_available:
                return "general", None

            if route != "rag":
                rag_name = None
            else:
                if not rag_name:
                    rag_name = self._choose_rag_name_default(allowed_vectorstores)

            return route, rag_name

        except Exception:
            if allowed_vectorstores:
                return "rag", self._choose_rag_name_default(allowed_vectorstores)
            return "general", None



    def _extract_json_block(self, text: str) -> str:
        """Extracts the first JSON block (from ```json ...``` or raw)."""
        t = text.strip()
        if t.startswith("```"):
            # remove backticks
            t = t.strip("`")
            # after an optional ```json label
            parts = t.split("\n", 1)
            if len(parts) == 2 and parts[0].lower().startswith("json"):
                t = parts[1]
        # models sometimes add comments — attempt to sanitize
        t = t.strip()
        # if the response has prefixes/suffixes, try to locate the braces
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return t[start:end+1]
        return t

    def _choose_rag_name_default(self, allowed: Optional[List[str]]) -> Optional[str]:
        for rc in self._rag_catalog:
            if not allowed or rc.name in allowed:
                return rc.name
        return None

    # --------------------------
    #  Execution paths
    # --------------------------

    def _do_tools(
        self,
        *,
        question: str,
        allowed_tools: Optional[List[str]],
        output_model: Optional[Type],
        stream: bool,
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        tool_usage: Literal["auto", "required", "none"] = "auto",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """TOOLS path with policy control (auto/required/none)."""

        # no tools → react per policy
        if not self._tools_registry or len(self._tools_registry.list()) == 0:
            msg = "[intergraxChatAgent][tools] No tools registered."
            if tool_usage == "required":
                raise RuntimeError(f"{msg} Tool usage policy = 'required'. Aborting.")
            return self._do_general(question=question, output_model=output_model, stream=stream, run_id=run_id)

        # lazy init
        if self._tools_agent is None:
            self._tools_agent = ToolsAgent(
                llm=self.llm,
                tools=self._tools_registry,
                memory=self.memory,
                config=self._tools_config,
            )

        # temporary tools filter
        backup_registry = None
        if allowed_tools:
            backup_registry = self._tools_registry
            filtered = ToolRegistry()
            for t in self._tools_registry.list():
                if t.name in allowed_tools:
                    filtered.register(t)
            self._tools_registry = filtered
            self._tools_agent.tools = filtered

        try:
            res = self._tools_agent.run(
                input_data=question,
                context=None,
                stream=stream,
                tool_choice=tool_choice,
                output_model=output_model,
                run_id=run_id
            )

            if self.memory and res.get("answer"):
                self.memory.add("assistant", res["answer"])

            return {
                "answer": res.get("answer", ""),
                "tool_traces": res.get("tool_traces", []),
                "sources": [],
                "summary": None,
                "messages": res.get("messages", []),
                "output_structure": res.get("output_structure"),
                "stats": {},
                "rag_component": None,                
            }

        except KeyError as e:
            # missing specific tool (e.g., calculate_vat)
            missing = None
            msg = str(e)
            if "Unknown tool:" in msg:
                missing = msg.split("Unknown tool:", 1)[1].strip().strip("'").strip()

            if tool_usage == "required":
                # hard stop (per policy)
                raise RuntimeError(f"Tool '{missing or 'unknown'}' required by LLM is not registered.")

            # in auto mode — soft fallback
            general = self._do_general(question=question, output_model=output_model, stream=stream)
            general.setdefault("tool_traces", []).append({
                "status": "error",
                "error": "missing_tool",
                "tool": missing or "unknown",
                "note": "Planner requested a tool that is not registered. Fell back to GENERAL.",
            })
            general.setdefault("stats", {})["tool_fallback"] = "missing_tool"
            return general

        except Exception as e:
            # other tool runtime errors
            if tool_usage == "required":
                raise RuntimeError(f"Tool execution failed under 'required' policy: {e}")

            general = self._do_general(question=question, output_model=output_model, stream=stream)
            general.setdefault("tool_traces", []).append({
                "status": "error",
                "error": "tool_runtime_error",
                "note": f"Exception during tool run: {type(e).__name__}",
            })
            general.setdefault("stats", {})["tool_fallback"] = "runtime_error"
            return general

        finally:
            if backup_registry is not None:
                self._tools_registry = backup_registry
                self._tools_agent.tools = backup_registry

    def _do_rag(
        self,
        *,
        question: str,
        rag_name: Optional[str],
        allowed_vectorstores: Optional[List[str]],
        stream: bool,
        summarize: bool,
        output_model: Optional[Type],
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rc = None
        if rag_name:
            rc = next((x for x in self._rag_catalog if x.name == rag_name), None)
        if rc is None:
            # if no specified RAG — choose default allowed one
            chosen = self._choose_rag_name_default(allowed_vectorstores)
            rc = next((x for x in self._rag_catalog if x.name == chosen), None)

        if rc is None:
            return self._do_general(question=question, output_model=output_model, stream=stream, summarize=summarize)

        res = rc.answerer.run(
            question=question,
            where=None,             # inject where-clauses inside the answerer if needed
            stream=stream,
            summarize=summarize,
            output_model=output_model,
            run_id=run_id,
        )

        if self.memory and res.get("answer"):
            self.memory.add("assistant", res["answer"])

        return {
            "answer": res.get("answer", ""),
            "tool_traces": [],
            "sources": res.get("sources", []),
            "summary": res.get("summary"),
            "messages": res.get("messages", []),
            "output_structure": res.get("output_structure"),
            "stats": res.get("stats", {}),
            "rag_component": rc.name,
        }

    def _do_general(
        self,
        *,
        question: str,
        output_model: Optional[Type],
        stream: bool,
        summarize: bool = False,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        msgs: List[ChatMessage] = [ChatMessage(role="system", content="You are a helpful, concise assistant.")]
        if self.memory and self.cfg.pass_memory_to_general:
            msgs.extend(self.memory.get_all())
        if not (len(msgs) and msgs[-1].role == "user"):
            msgs.append(ChatMessage(role="user", content=question))

        if stream:
            parts: List[str] = []
            for p in self.llm.stream_messages(
                msgs, 
                temperature=self.cfg.temperature, 
                max_tokens=self.cfg.max_answer_tokens,
                run_id=run_id,
            ):
                parts.append(p or "")
            answer = "".join(parts)
        else:
            answer = self.llm.generate_messages(
                msgs, 
                temperature=self.cfg.temperature, 
                max_tokens=self.cfg.max_answer_tokens,
                run_id=run_id
            )

        output_obj = None
        if (output_model is not None) and (not stream) and hasattr(self.llm, "generate_structured"):
            try:
                output_obj = self.llm.generate_structured(
                    msgs, 
                    output_model, 
                    temperature=self.cfg.temperature, 
                    max_tokens=self.cfg.max_answer_tokens,
                    run_id=run_id,
                )
            except Exception:
                output_obj = None

        summary_txt = None
        if summarize:
            try:
                s_msgs = [
                    ChatMessage(role="system", content="Summarize briefly."),
                    ChatMessage(role="user", content=answer),
                ]
                summary_txt = self.llm.generate_messages(
                    s_msgs, 
                    max_tokens=256,
                    run_id=run_id)
            except Exception:
                summary_txt = None

        if self.memory:
            payload = answer + (("\n\n" + summary_txt) if summary_txt else "")
            self.memory.add("assistant", payload)

        return {
            "answer": answer,
            "tool_traces": [],
            "sources": [],
            "summary": summary_txt,
            "messages": msgs,
            "output_structure": output_obj,
            "stats": {},
            "rag_component": None,
        }

    # --------------------------
    #  RAG/TOOLS registration
    # --------------------------

    def register_rag(self, comp: RagComponent) -> None:
        self._rag_catalog.append(comp)
        self._rag_catalog.sort(key=lambda rc: rc.priority)

    def register_rag_many(self, comps: Iterable[RagComponent]) -> None:
        self._rag_catalog.extend(list(comps))
        self._rag_catalog.sort(key=lambda rc: rc.priority)

    def set_tools(self, tools: ToolRegistry, config: Optional[ToolsAgentConfig] = None) -> None:
        self._tools_registry = tools
        if config:
            self._tools_config = config
        self._tools_agent = None  # lazy re-init
