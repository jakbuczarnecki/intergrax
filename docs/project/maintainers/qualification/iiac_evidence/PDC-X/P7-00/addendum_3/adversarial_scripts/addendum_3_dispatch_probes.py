"""P7-00 addendum 3: read-only probes of the ollama catalog row dispatch (auditor-owned).

Usage: python addendum_3_dispatch_probes.py <checkout_root> <out_json>
Runs in-process against the given checkout (head 0fc8badd8 or base 324d9dba3) and records, per probe, the observed
behaviour as plain data so the two runs can be diffed. Nothing is written into the checkout, no network access is made.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))

from intergrax.integrations.contracts.base import IntegrationCategory  # noqa: E402
from intergrax.integrations.registry import catalog as cat  # noqa: E402
from intergrax.integrations.registry.factory import resolve  # noqa: E402

out: dict[str, object] = {"root": str(root)}


def describe(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def attempt(fn):  # noqa: ANN001
    try:
        value = fn()
        return {"ok": describe(value)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def fresh_full_catalog() -> None:
    from intergrax.integrations.registry.bootstrap import (
        register_default_integrations,
        reset_default_integrations_state,
    )

    cat.clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")


# ---- P1: every (slug, category) pair of the default full catalog, resolved with no config
fresh_full_catalog()
matrix: dict[str, object] = {}
for entry in sorted(cat.iter_entries(), key=lambda e: e.slug):
    for category in entry.categories:
        key = f"{entry.slug}|{category.value}"
        matrix[key] = attempt(lambda e=entry, c=category: resolve(c, e.slug))
out["P1_resolve_matrix"] = matrix
out["P1_multi_category_rows"] = {
    e.slug: [c.value for c in e.categories] for e in cat.iter_entries() if len(e.categories) > 1
}
out["P1_ollama_row"] = {
    "categories": [c.value for c in cat.get_entry("ollama").categories],
    "factory": getattr(cat.get_entry("ollama").factory, "__qualname__", "?"),
    "spec_categories": sorted(str(s.category) for s in cat.get_entry("ollama").contract_specs),
}

# ---- P2: factory called directly with category shapes and extra keyword arguments
fresh_full_catalog()
factory = cat.get_entry("ollama").factory
E = IntegrationCategory.EMBEDDING_PROVIDER
M = IntegrationCategory.MODEL_SERVING_RUNTIME
shapes = {
    "enum_embedding": {"integration_category": E},
    "str_embedding": {"integration_category": "embedding_provider"},
    "enum_model_serving": {"integration_category": M},
    "str_model_serving": {"integration_category": "model_serving_runtime"},
    "none_category": {"integration_category": None},
    "no_category": {},
    "unknown_category_str": {"integration_category": "relational_store"},
    "embedding_enabled_true": {"integration_category": E, "enabled": True},
    "embedding_enabled_str_false": {"integration_category": E, "enabled": "false"},
    "embedding_enabled_zero": {"integration_category": E, "enabled": 0},
    "embedding_extra_keys": {"integration_category": E, "base_url": "http://example.invalid:1", "api_key": "x"},
}
p2: dict[str, object] = {}
for name, kwargs in shapes.items():
    res = attempt(lambda k=kwargs: factory(**k))
    if "ok" in res:
        inst = factory(**kwargs)
        cfg = getattr(inst, "config", None)
        res["enabled"] = getattr(cfg, "enabled", "n/a")
        res["config_base_url"] = getattr(cfg, "base_url", "n/a")
        res["provider_id"] = str(getattr(inst, "provider_id", getattr(inst, "integration_id", "n/a")))
    p2[name] = res
out["P2_factory_direct"] = p2

# ---- P3: wrapper forwards everything that is not the embedding category to the owning factory (spy factory)
try:
    from intergrax.integrations.providers.embedding_provider.ollama import register as reg

    spy_calls: list[dict[str, object]] = []

    def spy(**kwargs: object):  # noqa: ANN202
        spy_calls.append(dict(kwargs))
        return object()

    wrapped = reg._dispatching_factory(spy)  # type: ignore[attr-defined]  # noqa: SLF001
    forwarded: dict[str, object] = {}
    for name, kwargs in shapes.items():
        spy_calls.clear()
        res = attempt(lambda k=kwargs: wrapped(**k))
        forwarded[name] = {"spy_called": bool(spy_calls), "spy_kwargs": sorted(spy_calls[0]) if spy_calls else [], **res}
    out["P3_wrapper_forwarding"] = forwarded
except Exception as exc:  # noqa: BLE001  (base checkout has no wrapper)
    out["P3_wrapper_forwarding"] = {"absent": f"{type(exc).__name__}: {exc}"}

# ---- P4: catalog revision after the registration sequence, and idempotency / ordering
from intergrax.integrations.providers.embedding_provider.ollama.register import (  # noqa: E402
    register_ollama_embedding_provider_integration as reg_emb,
)
from intergrax.integrations.providers.model_serving_runtime.ollama.register import (  # noqa: E402
    register_ollama_integration as reg_own,
)

cat.clear_catalog()
before = cat.read_catalog_revision_under_lock()
reg_own()
mid = cat.read_catalog_revision_under_lock()
reg_emb()
after = cat.read_catalog_revision_under_lock()
out["P4_revision"] = {
    "gen_before": before.generation,
    "gen_after_owning": mid.generation,
    "gen_after_embedding": after.generation,
    "digest_changed_by_embedding": mid.state_digest != after.state_digest,
}
out["P4_second_embedding_registration"] = attempt(lambda: reg_emb())
cat.clear_catalog()
out["P4_embedding_without_owning_row"] = attempt(lambda: reg_emb())
out["P4_catalog_after_failed_embedding"] = cat.list_slugs()
cat.clear_catalog()
reg_own()
reg_emb()
reg_own(override=True)
row = cat.get_entry("ollama")
out["P4_owning_override_after_embedding"] = {
    "categories": [c.value for c in row.categories],
    "resolve_embedding": attempt(lambda: resolve(E, "ollama")),
}

# ---- P5: isolated registration sink (as used by catalog reload) leaves the global catalog untouched
cat.clear_catalog()
reg_own()
snapshot_before = {k: [c.value for c in v.categories] for k, v in cat.catalog_snapshot().items()}
sink: dict[str, object] = {}
try:
    with cat.isolated_catalog_registration(sink):  # type: ignore[arg-type]
        reg_emb()
    out["P5_sink"] = {"sink_slugs": sorted(sink), "global_unchanged": snapshot_before == {
        k: [c.value for c in v.categories] for k, v in cat.catalog_snapshot().items()
    }}
except Exception as exc:  # noqa: BLE001
    out["P5_sink"] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}

# ---- P6: transient window between augment and the wrapper registration (head only: two registration steps)
cat.clear_catalog()
reg_own()
seen: dict[str, object] = {}
try:
    import intergrax.integrations.providers.embedding_provider.ollama.register as regmod

    original = regmod.register_integration

    def observe(entry, *, override=False):  # noqa: ANN001, ANN202
        row_now = cat.get_entry("ollama")
        seen["categories_visible_before_wrapper"] = [c.value for c in row_now.categories]
        seen["resolve_embedding_in_window"] = attempt(lambda: resolve(E, "ollama"))
        original(entry, override=override)

    regmod.register_integration = observe  # type: ignore[assignment]
    try:
        reg_emb()
    finally:
        regmod.register_integration = original  # type: ignore[assignment]
    seen["resolve_embedding_after"] = attempt(lambda: resolve(E, "ollama"))
except Exception as exc:  # noqa: BLE001
    seen["absent"] = f"{type(exc).__name__}: {str(exc)[:160]}"
out["P6_transient_window"] = seen

# ---- P7: concurrent resolve while another thread registers (soak, 200 rounds)
bad = 0
rounds = 200
for _ in range(rounds):
    cat.clear_catalog()
    reg_own()
    results: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                row_now = cat.get_entry("ollama")
            except Exception:  # noqa: BLE001
                continue
            if E in row_now.categories:
                results.append(attempt(lambda: resolve(E, "ollama")).get("ok", "ERR"))

    t = threading.Thread(target=reader)
    t.start()
    reg_emb()
    stop.set()
    t.join()
    if any(r != "intergrax.integrations.providers.embedding_provider.ollama.integration.OllamaEmbeddingProviderIntegration" for r in results):
        bad += 1
out["P7_concurrent_resolve"] = {"rounds": rounds, "rounds_with_wrong_or_error_result": bad}

Path(sys.argv[2]).write_text(json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8")
print("done", sys.argv[2])
