import subprocess, pathlib, sys, json, re
WT = pathlib.Path("D:/Projekty/_wt/intergrax/ia-d3-mut")
OUT = pathlib.Path("D:/Projekty/intergrax/.claude/worktrees/pdc-x-ia-d3-d10590/.tmp/session/pdc-ia-d3/mutation")
D3 = ["tests/qualification/pdc_x/d3"]
ECAP = ["tests/qualification/pdc_x/p6_ecapfix"]
MUTANTS = [
 ("M01_R022_failover_tool_streaming_always_true","intergrax/llm_adapters/registry/failover_adapter.py",
  "return any(read_feature_flag(entry.adapter, LLMFeature.TOOL_STREAMING) for entry in self._chain)","return True", D3),
 ("M02_R006_base_tool_stream_downgrades_to_empty","intergrax/llm_adapters/base/base_llm_adapter.py",
  'raise self._capability_mismatch(LLMFeature.TOOL_STREAMING, "Tools streaming is not supported by this adapter.")',"return iter(())", D3),
 ("M03_R015_failover_falls_back_to_first_member","intergrax/llm_adapters/registry/failover_adapter.py",
  "        primary = self._chain[0].adapter\n        raise LLMCapabilityMismatchError(\n            feature,","        return adapters[0]\n        primary = self._chain[0].adapter\n        raise LLMCapabilityMismatchError(\n            feature,", D3),
 ("M04_R004_failover_passes_chain_twice","intergrax/llm_adapters/registry/failover_adapter.py",
  "for index, entry in enumerate(active_chain):","for index, entry in enumerate(active_chain + active_chain):", D3),
 ("M05_R009_payload_does_not_upgrade_to_refusal","intergrax/llm_adapters/contracts/finish_reason.py",
  "if text is not None and finish_reason is not LLMFinishReason.ERROR:","if False:", D3),
 ("M06_R011_swap_terminated_transport","intergrax/llm_adapters/contracts/capabilities.py",
  "    if capabilities.supports_native_cancel:\n        return CancelCapability.TERMINATED","    if capabilities.supports_native_cancel:\n        return CancelCapability.TRANSPORT_CLOSED", D3),
 ("M07_R023_legacy_table_beats_catalog","intergrax/llm_adapters/registry/context_window.py",
  "    exact = cat.lookup_exact(normalized_model)\n    if exact is not None:\n        return exact.context_window_tokens\n","    if legacy_windows is not None and normalized_model in legacy_windows:\n        return int(legacy_windows[normalized_model])\n    exact = cat.lookup_exact(normalized_model)\n    if exact is not None:\n        return exact.context_window_tokens\n", D3),
 ("M08_R015_undeclared_flag_counts_as_supported","intergrax/llm_adapters/contracts/capabilities.py",
  "    if not callable(probe):\n        return False\n    return probe() is True","    if not callable(probe):\n        return True\n    return probe() is True", D3),
 ("M09_R009_drop_safety_alias","intergrax/llm_adapters/contracts/finish_reason.py",
  '        "safety": LLMFinishReason.CONTENT_FILTER,\n',"", D3),
 ("M10_F49_openai_dispatch_inverted","intergrax/integrations/providers/embedding_provider/openai/register.py",
  "if kwargs.get(\"integration_category\") == IntegrationCategory.EMBEDDING_PROVIDER:","if kwargs.get(\"integration_category\") != IntegrationCategory.EMBEDDING_PROVIDER:", ECAP),
 ("M11_F50_openai_enabled_hardwired_true","intergrax/integrations/providers/embedding_provider/openai/register.py",
  "enabled=bool(kwargs.get(\"enabled\", False))","enabled=True", ECAP),
 ("M12_R011_stream_registry_not_closed","intergrax/llm_adapters/_shared/provider_stream_admission.py",
  "                stream_registry.register(operation_id, _close_transport)","                pass", D3),
 ("M13_R022_stream_flag_false_on_native_ollama","intergrax/llm_adapters/providers/native_ollama_adapter.py",
  "    def supports_streaming(self) -> bool:\n        return True","    def supports_streaming(self) -> bool:\n        return False", D3),
 ("M14_R006_catalog_overlay_tool_stream_flag_true","intergrax/llm_adapters/registry/catalog_capabilities.py",
  "require_declared_features(self, LLMFeature.TOOL_STREAMING)","pass", D3),
]
only = sys.argv[1:]
results=[]
for name, rel, old, new, tests in MUTANTS:
    if only and not any(name.startswith(o) for o in only): continue
    p = WT/rel
    src = p.read_text(encoding="utf-8")
    if old not in src:
        results.append((name,"NO_MATCH")); print(name,"NO_MATCH",flush=True); continue
    p.write_text(src.replace(old,new,1),encoding="utf-8")
    try:
        r = subprocess.run(["uv","run","--extra","integrations-duckdb","pytest",*tests,"-q","-p","no:randomly","-x"],cwd=WT,capture_output=True,text=True,timeout=550)
        tail = (r.stdout or "").strip().splitlines()[-3:]
        verdict = "SURVIVED" if r.returncode==0 else "KILLED"
        (OUT/f"{name}.log").write_text(r.stdout[-4000:], encoding="utf-8")
    except subprocess.TimeoutExpired:
        verdict="TIMEOUT"; tail=[]
    finally:
        p.write_text(src,encoding="utf-8")
    results.append((name,verdict)); print(name,verdict,tail[-1] if tail else "",flush=True)
json.dump(results, open(OUT/"results.json","w"), indent=1)
