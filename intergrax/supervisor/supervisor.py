# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Literal, Iterable, Callable
import json, re, math

from intergrax.llm_adapters.llm_adapter import LLMAdapter

from .supervisor_components import Component, ComponentContext, ComponentResult, PipelineState
from .supervisor_prompts import SupervisorPromptPack, SupervisorPromptPack as _DefaultPack

NeedsDict = Dict[str, bool]
Intent = Literal["qa", "procedural", "creative", "decision", "unknown"]

@dataclass
class PlanStep:
    id: str
    title: str
    goal: str
    method: str                 # "RAG" | "TOOL" | "GENERAL"
    component: Optional[str]
    inputs: List[str]
    outputs: List[str]
    success_criteria: List[str]
    fallback: str
    depends_on: List[str] = field(default_factory=list)
    why_method: Optional[str] = None
    why_component: Optional[str] = None
    notes: Optional[str] = None
    # assignment diagnostics
    router_score: float = 0.0
    router_notes: str = ""
    low_confidence: bool = False

@dataclass
class Plan:
    intent: Intent
    needs: NeedsDict
    confidence: float
    assumptions: List[str]
    steps: List[PlanStep] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    iteration_policy: Dict[str, Any] = field(default_factory=lambda: {"max_passes": 1, "improve_if": []})

@dataclass
class SupervisorConfig:
    llm_adapter: LLMAdapter
    temperature: float = 0.1
    max_tokens: int = 1200
    heuristic_enable: bool = True
    prompts: SupervisorPromptPack = field(default_factory=_DefaultPack)
    components: List[Component] = field(default_factory=list)
    debug: bool = False
    llm_response_format_json: bool = True
    pass_temperature: bool = False
    pass_max_tokens: bool = False
    resources: Dict[str, Any] = field(default_factory=dict)
    # "finally" components — always executed at the end
    finally_components: List[str] = field(default_factory=list)
    # fallback component used only when none could be assigned
    fallback_component: Optional[str] = None         # e.g., "Generalny"
    fallback_on_miss: bool = True                    # whether to enable fallback

    # Two-stage knobs
    planner_mode: Literal["one_shot", "two_stage"] = "two_stage"
    plan_retries: int = 2
    assign_retries: int = 2
    assign_self_consistency: int = 3
    assign_threshold: float = 0.5
    skip_on_low_confidence: bool = False
    assign_temperature: float = 0.3   # NEW: mild creativity helps the model commit

    # Semantic fallback (no keywords/tags required)
    semantic_fallback: bool = True
    semantic_weight: float = 0.15     # how much to boost score if semantic match is strong

    # Legacy post-fix (kept for compatibility)
    auto_fix_assignments: bool = False
    method_guards: bool = False
    auto_fix_passes: int = 1

class IntergraxSupervisor:
    def __init__(self, cfg: SupervisorConfig):
        self.cfg = cfg
        self.llm = cfg.llm_adapter
        self._prompts = cfg.prompts
        self._debug = bool(cfg.debug)
        self._components: Dict[str, Component] = {c.name: c for c in (cfg.components or [])}
        self.last_plan_meta: Dict[str, Any] = {}

    def make_context(self) -> ComponentContext:
        # Build a clean component execution context
        return ComponentContext(
            llm_adapter=self.cfg.llm_adapter,
            resources=getattr(self.cfg, "resources", {}) or {}
        )

    def register_components(self, comps: Iterable[Component]):
        for c in comps:
            self._components[c.name] = c

    def list_components(self) -> List[Component]:
        return list(self._components.values())

    def _is_available(self, c: Component) -> bool:
        return getattr(c, "available", True)

    def set_prompts(self, *, plan_system: Optional[str] = None, plan_user_template: Optional[str] = None):
        if plan_system is not None:
            self._prompts.plan_system = plan_system
        if plan_user_template is not None:
            self._prompts.plan_user_template = plan_user_template

    # --------- LLM helpers ---------
    def _extract_text(self, raw: Any) -> str:
        """
        Robustly extract assistant text from various adapter return types.
        This avoids empty {} after str(raw), which caused score=0.0 everywhere.
        """
        try:
            # common shape: {"choices":[{"message":{"role":"assistant","content":"..."}}]}
            if isinstance(raw, dict):
                ch = raw.get("choices")
                if isinstance(ch, list) and ch:
                    msg = ch[0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
            # some adapters return SimpleNamespace/objects with .content
            content = getattr(raw, "content", None)
            if isinstance(content, str):
                return content
            # some return list of messages
            if isinstance(raw, list) and raw:
                maybe = getattr(raw[-1], "content", None)
                if isinstance(maybe, str):
                    return maybe
            # fallback to string
            return str(raw)
        except Exception:
            return str(raw)

    def _safe_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            t = text.strip()
            if t.startswith("```"):
                t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.MULTILINE)
            if not t.startswith("{"):
                match = re.findall(r"\{.*\}", t, flags=re.DOTALL)
                if match:
                    t = max(match, key=len)
            return json.loads(t)
        except Exception:
            return None

    # --------- Public: plan ---------
    def plan(self, query: str, meta: Optional[Dict[str, Any]] = None,run_id: Optional[str] = None,) -> Plan:
        self.last_plan_meta = {}
        meta = meta or {}
        if self.cfg.planner_mode == "two_stage":
            plan = self._llm_plan_two_stage(query, run_id=run_id)
            if plan:
                self.last_plan_meta.setdefault("source", "stepwise")
                self.last_plan_meta["analysis"] = self.analyze_plan(plan)
                return plan

        # fallback one-shot
        catalog = self._render_catalog_for_prompt()
        plan = self._llm_plan_one_shot(query, catalog)
        if plan:
            self.last_plan_meta.setdefault("source", "llm")
            self.last_plan_meta["analysis"] = self.analyze_plan(plan)
            return plan
        if self.cfg.heuristic_enable:
            plan = self._heuristic_plan(query, meta)
            self.last_plan_meta.setdefault("source", "heuristic")
            self.last_plan_meta["analysis"] = self.analyze_plan(plan)
            return plan
        plan = self._minimal_plan()
        self.last_plan_meta.setdefault("source", "minimal_fallback")
        self.last_plan_meta["analysis"] = self.analyze_plan(plan)
        return plan

    # --------- Two-stage: decompose -> per-step assign ---------
    def _llm_plan_two_stage(self, query: str,run_id: Optional[str] = None,) -> Optional[Plan]:
        decomp = self._llm_decompose(query)
        if not decomp:
            return None
        catalog_rows = self._render_catalog_rows()
        for s in decomp.steps:
            self._assign_step_with_llm(s, catalog_rows, run_id=run_id)
        # recompute needs
        has_rag = any(x.method == "RAG" for x in decomp.steps)
        has_tools = any(x.method == "TOOL" for x in decomp.steps)
        has_general = any(x.method == "GENERAL" for x in decomp.steps)
        decomp.needs = {"rag": has_rag, "tools": has_tools, "general": has_general}
        return decomp

    def _llm_decompose(self, query: str, run_id: Optional[str] = None) -> Optional[Plan]:
        """
        Stage 1: get a clean step decomposition (no components bound).
        Strong JSON contract + retry if JSON parsing fails.
        """
        sys = (
            self._prompts.plan_system
            + "\n\nIMPORTANT: Return ONLY a single strict JSON object per schema. No prose, no explanations."
        )
        usr = (
            self._prompts.plan_user_template
            .replace("{query}", query)
            .replace("{catalog}", "- (catalog hidden at this stage)")
        )

        best: Optional[Plan] = None
        for _ in range(max(1, self.cfg.plan_retries)):
            messages = [
                SimpleNamespace(role="system", content=sys),
                SimpleNamespace(role="user", content=usr),
            ]
            call_kwargs: Dict[str, Any] = {"messages": messages}
            if self.cfg.pass_temperature:
                call_kwargs["temperature"] = self.cfg.temperature
            if self.cfg.pass_max_tokens:
                call_kwargs["max_tokens"] = self.cfg.max_tokens

            try:
                raw = self.llm.generate_messages(run_id=run_id, **call_kwargs)
            except TypeError:
                raw = self.llm.generate_messages(run_id=run_id, messages=messages)

            raw_text = self._extract_text(raw)
            if self._debug:
                self.last_plan_meta["raw_response_decompose"] = raw_text[:4000]

            data = self._safe_json(raw_text)
            if not data:
                # one corrective nudge
                corrective = "You must return ONLY valid JSON per schema, with double quotes and no trailing commas."
                raw2 = self.llm.generate_messages(
                    run_id=run_id,
                    messages=[
                    SimpleNamespace(role="system", content=sys),
                    SimpleNamespace(role="user", content=usr + "\n\n" + corrective),
                ])
                raw_text = self._extract_text(raw2)
                data = self._safe_json(raw_text)

            if not data:
                continue

            plan = self._parse_plan(data)
            # sanitize steps
            for i, s in enumerate(plan.steps, start=1):
                if not s.id:
                    s.id = f"S{i}"
                if not (s.title or "").strip():
                    s.title = f"Step {i}"
                if not (s.goal or "").strip():
                    s.goal = s.title
                s.component = None
                s.method = s.method if s.method in ("RAG", "TOOL", "GENERAL") else "GENERAL"
            best = plan
            break
        return best

    def _assign_step_with_llm(self, step: PlanStep, catalog_rows: List[Dict[str, Any]], run_id: Optional[str] = None,) -> None:
        """
        Stage 2: assign METHOD+COMPONENT per step using the catalog.
        Uses self-consistency voting and a semantic fallback if the JSON is empty/low-scored.
        """
        catalog_json = json.dumps(catalog_rows, ensure_ascii=False)
        # Strong JSON-only protocol with a micro few-shot baked-in via rules
        assign_system = (
            "You are an Assignment Adjudicator.\n"
            "Given ONE step (title, goal, inputs, outputs) and a catalog of components "
            "(name, description, use_when, outputs), select the best METHOD among: RAG | TOOL | GENERAL, "
            "and the best COMPONENT when METHOD != GENERAL.\n"
            "Return ONLY a JSON object with keys: method, component, score, reason.\n"
            "Constraints:\n"
            "- component MUST be null for GENERAL; must be a catalog name for RAG/TOOL.\n"
            "- Prefer components whose declared outputs overlap with step.outputs; use descriptions/use_when to disambiguate.\n"
            "- Be decisive; give score in [0.0,1.0].\n"
            "- Absolutely no text besides the single JSON object."
        )
        step_json = json.dumps({
            "id": step.id,
            "title": step.title,
            "goal": step.goal or step.title,
            "inputs": step.inputs,
            "outputs": step.outputs
        }, ensure_ascii=False)

        user_prompt = (
            "STEP:\n"
            f"{step_json}\n\n"
            "CATALOG:\n"
            f"{catalog_json}\n\n"
            "Return JSON only."
        )

        votes: Dict[str, int] = {}
        best = {"method": "GENERAL", "component": None, "score": 0.0, "reason": ""}

        rounds = max(1, self.cfg.assign_retries)
        per_round = max(1, self.cfg.assign_self_consistency)

        for _ in range(rounds):
            for _ in range(per_round):
                msgs = [
                    SimpleNamespace(role="system", content=assign_system),
                    SimpleNamespace(role="user", content=user_prompt),
                ]
                # Slight non-zero temperature helps avoid 'empty JSON' pathologies
                try:
                    raw = self.llm.generate_messages(messages=msgs, run_id=run_id, temperature=self.cfg.assign_temperature)
                except TypeError:
                    raw = self.llm.generate_messages(messages=msgs, run_id=run_id)
                data = self._safe_json(self._extract_text(raw)) or {}
                method = (str(data.get("method") or "GENERAL")).upper()
                comp = data.get("component")
                comp = str(comp) if isinstance(comp, str) else None
                score = float(data.get("score", 0.0))
                reason = data.get("reason", "")

                if comp and comp in self._components and self._is_available(self._components[comp]):
                    votes[comp] = votes.get(comp, 0) + 1

                if score > best["score"]:
                    best = {"method": method, "component": comp, "score": score, "reason": reason}

        # Majority vote consolidation
        if votes:
            winner = max(votes.items(), key=lambda kv: kv[1])[0]
            if best["component"] == winner:
                final = best
            else:
                final = {"method": best["method"], "component": winner,
                         "score": max(best["score"], 0.51), "reason": best.get("reason", "majority vote")}
        else:
            final = best

        # --- Semantic fallback (no keywords/tags needed) ---
        # If model didn't select a usable component or score is too low,
        # try cosine similarity between step text and catalog texts using an embed_fn passed in resources.
        if self.cfg.semantic_fallback:
            need_semantic = (final["component"] is None) or (final["score"] < self.cfg.assign_threshold)
            if need_semantic:
                pick_name, pick_score, pick_reason = self._semantic_route_component(step, catalog_rows)
                if pick_name:
                    # Blend LLM score with semantic evidence
                    blended = max(final["score"], min(1.0, pick_score + self.cfg.semantic_weight))
                    if blended > final["score"]:
                        final = {
                            "method": "TOOL",                 # semantic route implies a concrete tool pick
                            "component": pick_name,
                            "score": blended,
                            "reason": f"{final.get('reason','')}; semantic:{pick_reason}".strip("; ")
                        }

        # Apply final decision to step (respect threshold/skip flag)
        step.router_score = float(final["score"])
        step.router_notes = final.get("reason") or ""
        low_conf = step.router_score < float(self.cfg.assign_threshold)
        step.low_confidence = bool(low_conf)

        if (not low_conf or not self.cfg.skip_on_low_confidence) and final["component"]:
            step.component = final["component"]
            step.method = final["method"] if final["method"] in ("RAG", "TOOL", "GENERAL") else "GENERAL"
        else:
            step.component = None
            step.method = final["method"] if (not low_conf and final["method"] in ("RAG", "TOOL", "GENERAL")) else "GENERAL"

    # --------- Semantic router (embedding-based, no keywords) ---------
    def _semantic_route_component(self, step: PlanStep, catalog_rows: List[Dict[str, Any]]) -> (Optional[str], float, str): # type: ignore
        """
        Uses a user-provided embed_fn in cfg.resources['embed_fn'] to compute cosine similarity
        between step text and each component's (name+description+use_when+outputs).
        Returns (component_name, score[0..1], reason) or (None, 0.0, "").
        """
        embed_fn = (self.cfg.resources or {}).get("embed_fn")
        if not callable(embed_fn):
            return None, 0.0, ""

        def _vec(text: str) -> List[float]:
            try:
                v = embed_fn(text or "")
                return list(v) if v is not None else []
            except Exception:
                return []

        def _cos(a: List[float], b: List[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.0
            num = sum(x*y for x, y in zip(a, b))
            da = math.sqrt(sum(x*x for x in a))
            db = math.sqrt(sum(y*y for y in b))
            if da == 0.0 or db == 0.0:
                return 0.0
            return max(0.0, min(1.0, num / (da * db)))

        step_text = " ".join([
            step.title or "",
            step.goal or "",
            " ".join(step.inputs or []),
            " ".join(step.outputs or [])
        ]).strip()
        v_step = _vec(step_text)
        if not v_step:
            return None, 0.0, ""

        best_name, best_sim = None, 0.0
        for row in catalog_rows:
            name = str(row.get("name") or "")
            if not name or name not in self._components or not self._is_available(self._components[name]):
                continue
            blob = " ".join([
                name,
                str(row.get("description") or ""),
                str(row.get("use_when") or ""),
                " ".join(row.get("outputs") or []),
            ]).strip()
            v_comp = _vec(blob)
            sim = _cos(v_step, v_comp)
            if sim > best_sim:
                best_name, best_sim = name, sim

        if best_name and best_sim >= 0.35:  # modest floor; tune if needed
            return best_name, float(best_sim), f"cosine={best_sim:.2f}"
        return None, 0.0, ""

    # --------- One-shot fallback ---------
    def _llm_plan_one_shot(self, query: str, catalog_text: str,run_id: Optional[str] = None,) -> Optional[Plan]:
        try:
            sys = (
                self._prompts.plan_system
                + "\n\nIMPORTANT: Return ONLY a single strict JSON object per schema. No prose."
            )
            usr = (
                self._prompts.plan_user_template
                .replace("{query}", query)
                .replace("{catalog}", catalog_text)
            )
            messages = [
                SimpleNamespace(role="system", content=sys),
                SimpleNamespace(role="user", content=usr),
            ]
            call_kwargs: Dict[str, Any] = {"messages": messages}
            if self.cfg.pass_temperature:
                call_kwargs["temperature"] = self.cfg.temperature
            if self.cfg.pass_max_tokens:
                call_kwargs["max_tokens"] = self.cfg.max_tokens

            try:
                raw = self.llm.generate_messages(run_id=run_id, **call_kwargs)
            except TypeError:
                raw = self.llm.generate_messages(messages=messages, run_id=run_id)

            raw_text = self._extract_text(raw)
            if self._debug:
                self.last_plan_meta["raw_response"] = raw_text[:4000]

            data = self._safe_json(raw_text)
            if not data:
                return None

            plan = self._parse_plan(data)
            for step in plan.steps:
                if step.component and step.component not in self._components:
                    step.method = "GENERAL"
                    step.component = None
            return plan
        except Exception as e:
            if self._debug:
                self.last_plan_meta["llm_exception"] = repr(e)
            return None

    # --------- Catalog rendering ---------
    def _render_catalog_for_prompt(self) -> str:
        lines = []
        for c in self._components.values():
            if not self._is_available(c):
                continue
            ex = f" | examples: {', '.join(c.examples)}" if getattr(c, "examples", None) else ""
            lines.append(f"- name: {c.name} | desc: {c.description} | use_when: {c.use_when}{ex}")
        if not lines:
            return "- (no components available)"
        return "\n".join(lines)

    def _render_catalog_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for c in self._components.values():
            if not self._is_available(c):
                continue
            rows.append({
                "name": c.name,
                "description": getattr(c, "description", ""),
                "use_when": getattr(c, "use_when", ""),
                "outputs": getattr(c, "outputs", []) or getattr(c, "allowed_outputs", []) or [],
            })
        return rows

    # --------- Parse plan JSON ---------
    def _parse_plan(self, data: Dict[str, Any]) -> Plan:
        steps: List[PlanStep] = []
        for s in data.get("steps", []):
            steps.append(PlanStep(
                id=s.get("id", f"S{len(steps)+1}"),
                title=s.get("title", ""),
                goal=s.get("goal", ""),
                method=s.get("method", "GENERAL"),
                component=s.get("component"),
                inputs=s.get("inputs", []),
                outputs=s.get("outputs", []),
                success_criteria=s.get("success_criteria", []),
                fallback=s.get("fallback", "Retry or ask user"),
                depends_on=s.get("depends_on", []),
                why_method=s.get("why_method"),
                why_component=s.get("why_component"),
                notes=s.get("notes"),
            ))
        return Plan(
            intent=data.get("intent", "unknown"),
            needs=data.get("needs", {"rag": False, "tools": False, "general": True}),
            confidence=float(data.get("confidence", 0.5)),
            assumptions=data.get("assumptions", []),
            steps=steps,
            risks=data.get("risks", []),
            iteration_policy=data.get("iteration_policy", {"max_passes": 1, "improve_if": []})
        )

    # --------- Heuristic/minimal fallbacks ---------
    def _heuristic_plan(self, query: str, meta: Dict[str, Any]) -> Plan:
        steps = [
            PlanStep(
                id="S1",
                title="Synthesis (general)",
                goal="Draft an initial answer using general reasoning.",
                method="GENERAL",
                component=None,
                inputs=["query"],
                outputs=["draft_answer"],
                success_criteria=["relevance", "clarity"],
                fallback="Ask user for clarification."
            )
        ]
        return Plan(
            intent="unknown",
            needs={"rag": False, "tools": False, "general": True},
            confidence=0.3,
            assumptions=[],
            steps=steps,
            risks=["Heuristic fallback used"],
            iteration_policy={"max_passes": 1, "improve_if": ["user dissatisfied"]}
        )

    def _minimal_plan(self) -> Plan:
        return Plan(
            intent="unknown",
            needs={"general": True},
            confidence=0.2,
            assumptions=[],
            steps=[
                PlanStep(
                    id="S1",
                    title="General draft",
                    goal="Provide minimal answer",
                    method="GENERAL",
                    component=None,
                    inputs=["query"],
                    outputs=["draft_answer"],
                    success_criteria=["clarity"],
                    fallback="Ask user for more info."
                )
            ],
            risks=["No context"],
            iteration_policy={"max_passes": 1}
        )

    # --------- Printing & diagnostics ---------
    def _preview(self, val, n: int = 400) -> str:
        try:
            s = json.dumps(val, ensure_ascii=False)
        except Exception:
            s = str(val)
        return s if len(s) <= n else s[:n] + "…"

    def _per_step_index(self, analysis: dict) -> dict:
        rows = analysis.get("per_step", []) if analysis else []
        return {r.get("id"): r for r in rows}

    def _print_step_common(self, idx: int, step: "PlanStep", comp_obj, comp_found: bool, analysis_row: dict | None, print_fn=print):
        print_fn(f"{idx}. {step.title or step.id}")
        goal_txt = (step.goal or "").strip()
        if not goal_txt or goal_txt == "-":
            goal_txt = (step.notes or "").strip() or (step.title or "").strip() or "(no goal)"
        print_fn(f"   Goal: {goal_txt}")
        print_fn(f"   How to execute: method={step.method}, depends_on={step.depends_on or []}")
        print_fn(f"   Component: {step.component or '—'}")
        print_fn(f"   Component found: {comp_found}")
        if hasattr(step, "router_score"):
            print_fn(f"   Assign score: {step.router_score:.2f} | low_confidence={bool(getattr(step,'low_confidence',False))}")
        print_fn(f"   Inputs: {step.inputs or []}")
        print_fn(f"   Outputs: {step.outputs or []}")
        print_fn(f"   Success criteria: {step.success_criteria or []}")
        print_fn(f"   Fallback: {step.fallback or '-'}")
        wm = getattr(step, "why_method", None) or (analysis_row or {}).get("why_method")
        wc = getattr(step, "why_component", None) or (analysis_row or {}).get("why_component")
        if wm or wc:
            print_fn("   Rationale:")
            if wm: print_fn(f"     why_method: {wm}")
            if wc: print_fn(f"     why_component: {wc}")

    def print_plan(self, plan: "Plan", *, print_fn: Callable[[str], None] = print) -> None:
        needs_str = f"rag={plan.needs.get('rag', False)}, tools={plan.needs.get('tools', False)}, general={plan.needs.get('general', False)}"
        print_fn(f"INTENT: {plan.intent} | Needs: {needs_str} | Confidence: {plan.confidence:.2f}")
        if plan.assumptions:
            print_fn("Assumptions: " + "; ".join(plan.assumptions))

        analysis = (self.last_plan_meta or {}).get("analysis") or self.analyze_plan(plan)
        per_idx = self._per_step_index(analysis)

        for i, s in enumerate(plan.steps, start=1):
            comp_obj = self._components.get(s.component) if s.component else None
            comp_found = bool(comp_obj and self._is_available(comp_obj))
            self._print_step_common(i, s, comp_obj, comp_found, per_idx.get(s.id), print_fn=print_fn)

        meta = self.last_plan_meta or {}
        print_fn("\n--- Supervisor diagnostics ---")
        print_fn("Source: " + str(meta.get("source", "unknown")))

        dag = analysis.get("dag", {})
        violations = dag.get("violations", []) or []

        has_rag = any(s.method == "RAG" for s in plan.steps)
        has_tools = any(s.method == "TOOL" for s in plan.steps)
        has_general = any(s.method == "GENERAL" for s in plan.steps)
        mm = []
        if bool(plan.needs.get("rag", False)) != has_rag:
            mm.append(f"needs.rag={plan.needs.get('rag')} vs steps_have_rag={has_rag}")
        if bool(plan.needs.get("tools", False)) != has_tools:
            mm.append(f"needs.tools={plan.needs.get('tools')} vs steps_have_tools={has_tools}")
        if bool(plan.needs.get("general", False)) != has_general:
            mm.append(f"needs.general={plan.needs.get('general')} vs steps_have_general={has_general}")

        print_fn("Needs consistency: " + ("OK" if not mm else "; ".join(mm)))
        print_fn("DAG violations: " + ("none" if not violations else str(violations)))

    # --------- Execute ---------
    def execute_plan(
        self,
        plan: "Plan",
        *,
        query: str = "",
        entry_artifacts: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        print_fn: Callable[[str], None] = print,
        max_output_len: int = 400
    ) -> Dict[str, Any]:
        ctx = self.make_context()
        state: PipelineState = {
            "query": query or "",
            "artifacts": dict(entry_artifacts or {}),
            "step_status": {},
            "debug_logs": [],
            "last_output": None,
            "cursor": {},
            "terminated": False,
            "terminated_by": None,
            "terminate_reason": None,
            "finally_status": {}
        }

        analysis = self.analyze_plan(plan)
        per_idx = self._per_step_index(analysis)

        for idx, step in enumerate(plan.steps, start=1):
            comp = self._components.get(step.component) if step.component else None
            comp_found = bool(comp and self._is_available(comp))

            # Runtime fallback only when missing
            used_fallback = False
            if (not comp_found) and self.cfg.fallback_on_miss and self.cfg.fallback_component:
                fb = self._components.get(self.cfg.fallback_component)
                if fb and self._is_available(fb):
                    comp = fb
                    comp_found = True
                    used_fallback = True
                    state["debug_logs"].append(
                        f"[{step.id}] fallback component used: '{self.cfg.fallback_component}'"
                    )

            if verbose:
                self._print_step_common(idx, step, comp, comp_found, per_idx.get(step.id), print_fn=print_fn)
                if used_fallback:
                    print_fn(f"   (Default component used: {self.cfg.fallback_component})")

            # inputs presence info
            missing_inputs = []
            for inp in (step.inputs or []):
                key = (inp or "").strip().rstrip("?")
                if not key:
                    continue
                if key in state["artifacts"]:
                    continue
                missing_inputs.append(key)
            if missing_inputs and verbose:
                print_fn(f"   Note: missing inputs: {missing_inputs} (treated as user input or skipped)")

            if not comp_found:
                state["step_status"][step.id] = {"status": "error", "error": "component_not_found"}
                state["debug_logs"].append(f"[{step.id}] component '{step.component}' not found/available and no fallback")
                if verbose:
                    print_fn(f"   Invocation: skipped (component unavailable)")
                continue

            try:
                result: ComponentResult = comp.run(state, ctx)
            except Exception as e:
                result = ComponentResult(ok=False, logs=[f"exception: {e}"], meta={"error": str(e)})

            if result.logs:
                state["debug_logs"].extend([f"[{step.id}:{comp.name}] {l}" for l in result.logs])
            if result.produces:
                state["artifacts"].update(result.produces)
            state["last_output"] = result.output if result.output is not None else state.get("last_output")

            step_meta = dict(result.meta or {})
            state["step_status"][step.id] = {"status": "ok" if result.ok else "error", **step_meta}

            stop_flag = bool(step_meta.get("stop") or step_meta.get("halt") or step_meta.get("terminate"))
            if stop_flag:
                reason = step_meta.get("reason") or "component_requested_stop"
                state["terminated"] = True
                state["terminated_by"] = step.id
                state["terminate_reason"] = reason
                state["debug_logs"].append(f"[{step.id}:{comp.name}] PIPELINE STOP: {reason}")

            if verbose:
                status_txt = "OK" if result.ok else "ERROR"
                out_preview = self._preview(result.output, n=max_output_len) if (result.output is not None) else "-"
                prod_preview = self._preview(result.produces, n=max_output_len) if result.produces else "-"
                print_fn(f"   Invocation: {status_txt}")
                print_fn(f"   Result (output): {out_preview}")
                print_fn(f"   Artifacts (produces): {prod_preview}")
                if result.logs:
                    print_fn(f"   Logs: {', '.join(result.logs)}")
                if stop_flag:
                    print_fn(f"   *** PIPELINE TERMINATED *** reason: {reason}")

            if stop_flag:
                break

        self._run_finally_components(state, ctx, verbose=verbose, print_fn=print_fn, max_output_len=max_output_len)
        return state

    def _run_finally_components(
        self,
        state: PipelineState,
        ctx: ComponentContext,
        *,
        verbose: bool,
        print_fn: Callable[[str], None],
        max_output_len: int
    ):
        if not self.cfg.finally_components:
            return

        print_fn("\n=== FINALLY (components always executed) ===")
        for i, name in enumerate(self.cfg.finally_components, start=1):
            comp = self._components.get(name)
            comp_found = bool(comp and self._is_available(comp))
            print_fn(f"{i}. {name}")
            print_fn("   Goal: final component / cleanup / aggregate report")
            print_fn("   How to execute: method=GENERAL, depends_on=[]")
            print_fn(f"   Component: {name}")
            print_fn(f"   Component found: {comp_found}")
            print_fn(f"   Inputs: {list((state.get('artifacts') or {}).keys())[:6]}{' …' if len((state.get('artifacts') or {}))>6 else ''}")
            print_fn("   Outputs: ['*per component*']")
            print_fn("   Success criteria: ['runs without exception']")
            print_fn("   Fallback: '-'")

            if not comp_found:
                state["finally_status"][name] = {"status": "error", "error": "component_not_found"}
                print_fn("   Invocation: skipped (component unavailable)")
                continue

            try:
                result: ComponentResult = comp.run(state, ctx)
            except Exception as e:
                result = ComponentResult(ok=False, logs=[f"exception: {e}"], meta={"error": str(e)})

            if result.logs:
                state["debug_logs"].extend([f"[FINALLY:{name}] {l}" for l in result.logs])
            if result.produces:
                state["artifacts"].update(result.produces)

            state["finally_status"][name] = {"status": "ok" if result.ok else "error", **(result.meta or {})}

            status_txt = "OK" if result.ok else "ERROR"
            out_preview = self._preview(result.output, n=max_output_len) if (result.output is not None) else "-"
            prod_preview = self._preview(result.produces, n=max_output_len) if result.produces else "-"
            print_fn(f"   Invocation: {status_txt}")
            print_fn(f"   Result (output): {out_preview}")
            print_fn(f"   Artifacts (produces): {prod_preview}")

    # --------- Static analysis ---------
    def analyze_plan(self, plan: Plan) -> Dict[str, Any]:
        available = {n: self._is_available(c) for n, c in self._components.items()}
        produced_by, violations = {}, []
        step_index = {s.id: i for i, s in enumerate(plan.steps)}

        for s in plan.steps:
            for out in s.outputs:
                key = (out or "").strip()
                if not key:
                    continue
                if key in produced_by:
                    violations.append(f"Duplicate output '{key}' in {produced_by[key]} and {s.id}")
                else:
                    produced_by[key] = s.id

        for s in plan.steps:
            for dep in s.depends_on:
                if dep not in step_index:
                    violations.append(f"{s.id} depends_on unknown {dep}")
                elif step_index[dep] >= step_index[s.id]:
                    violations.append(f"{s.id} depends_on non-prior step {dep}")

        per_step = []
        for s in plan.steps:
            comp = self._components.get(s.component) if s.component else None
            per_step.append({
                "id": s.id,
                "method": s.method,
                "component": s.component,
                "available": bool(comp and self._is_available(comp)),
                "why_method": s.why_method or self._infer_why_method(s),
                "why_component": s.why_component or self._infer_why_component(s)
            })

        return {
            "dag": {"produced_by": produced_by, "violations": violations},
            "components_available": available,
            "per_step": per_step
        }

    def _infer_why_method(self, s: PlanStep) -> str:
        if s.method == "TOOL":
            return "Domain operation handled by a specialized tool."
        if s.method == "RAG":
            return "Requires domain retrieval or knowledge search."
        return "General reasoning step."

    def _infer_why_component(self, s: PlanStep) -> Optional[str]:
        return f"Chosen by planner: {s.component}" if s.component else None
