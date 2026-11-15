"""P7-00 addendum 4: read-only probes of the shared-slug catalog rows (ollama and openai) at one checkout.

Usage: python addendum_4_dispatch_probes.py <checkout_root> <out_json>
In-process, no network. A placeholder value is set for OPENAI_API_KEY only inside this process to get past the
managed-retrieval configuration check of the owning openai factory (no client call is made).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))

from intergrax.integrations.contracts.base import IntegrationCategory  # noqa: E402
from intergrax.integrations.registry import catalog as cat  # noqa: E402
from intergrax.integrations.registry.bootstrap import (  # noqa: E402
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.factory import resolve  # noqa: E402

E = IntegrationCategory.EMBEDDING_PROVIDER
SLUGS = {
    "ollama": {
        "owning": ("intergrax.integrations.providers.model_serving_runtime.ollama.register", "register_ollama_integration"),
        "embedding": ("intergrax.integrations.providers.embedding_provider.ollama.register", "register_ollama_embedding_provider_integration"),
        "owning_category": IntegrationCategory.MODEL_SERVING_RUNTIME,
    },
    "openai": {
        "owning": ("intergrax.integrations.providers.managed_retrieval.openai.register", "register_openai_managed_retrieval_integration"),
        "embedding": ("intergrax.integrations.providers.embedding_provider.openai.register", "register_openai_embedding_provider_integration"),
        "owning_category": IntegrationCategory.MANAGED_RETRIEVAL,
    },
}
out: dict[str, object] = {"root": str(root)}


def describe(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def attempt(fn):  # noqa: ANN001, ANN201
    try:
        return {"ok": describe(fn())}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}


def fresh_full() -> None:
    cat.clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")


os.environ["OPENAI_API_KEY"] = "placeholder-not-a-real-key"

# P1: resolve matrix over every (slug, category) pair of the full default catalog
fresh_full()
matrix: dict[str, object] = {}
for entry in sorted(cat.iter_entries(), key=lambda e: e.slug):
    for category in entry.categories:
        matrix[f"{entry.slug}|{category.value}"] = attempt(lambda e=entry, c=category: resolve(c, e.slug))
out["P1_resolve_matrix"] = matrix
out["P1_multi_category_rows"] = {e.slug: [c.value for c in e.categories] for e in cat.iter_entries() if len(e.categories) > 1}
out["P1_embedding_rows"] = {e.slug: [c.value for c in e.categories] for e in cat.iter_entries() if E in e.categories}
out["P1_factory_names"] = {s: getattr(cat.get_entry(s).factory, "__qualname__", "?") for s in SLUGS}

for slug, cfg in SLUGS.items():
    res: dict[str, object] = {}
    owning = getattr(importlib.import_module(cfg["owning"][0]), cfg["owning"][1])
    emb_mod = importlib.import_module(cfg["embedding"][0])
    emb = getattr(emb_mod, cfg["embedding"][1])
    oc = cfg["owning_category"]
    # P2 factory called directly
    fresh_full()
    factory = cat.get_entry(slug).factory
    shapes = {
        "enum_embedding": {"integration_category": E},
        "str_embedding": {"integration_category": "embedding_provider"},
        "enum_owning": {"integration_category": oc},
        "none_category": {"integration_category": None},
        "no_category": {},
        "unknown_category_str": {"integration_category": "relational_store"},
        "embedding_enabled_true": {"integration_category": E, "enabled": True},
        "embedding_enabled_str_false": {"integration_category": E, "enabled": "false"},
        "embedding_enabled_zero": {"integration_category": E, "enabled": 0},
        "embedding_extra_keys": {"integration_category": E, "base_url": "http://example.invalid:1", "api_key": "x", "credential_ref": "ref"},
    }
    p2: dict[str, object] = {}
    for name, kwargs in shapes.items():
        r = attempt(lambda k=kwargs: factory(**k))
        if "ok" in r:
            inst = factory(**kwargs)
            cfgobj = getattr(inst, "config", None)
            r["enabled"] = getattr(cfgobj, "enabled", "n/a")
            r["base_url"] = getattr(cfgobj, "base_url", "n/a")
            r["credential_ref"] = getattr(cfgobj, "credential_ref", "n/a")
        p2[name] = r
    res["P2_factory_direct"] = p2
    # P3 wrapper against a spy owning factory
    calls: list[dict[str, object]] = []

    def spy(**kwargs: object):  # noqa: ANN202
        calls.append(dict(kwargs))
        return object()

    maker = getattr(emb_mod, "_dispatching_factory", None)
    p3: dict[str, object] = {}
    if maker is None:
        p3 = {"absent": "no _dispatching_factory in this checkout"}
    else:
        wrapped = maker(spy)
        for name, kwargs in shapes.items():
            calls.clear()
            r = attempt(lambda k=kwargs: wrapped(**k))
            p3[name] = {"spy_called": bool(calls), "spy_kwargs": sorted(calls[0]) if calls else [], **r}
    res["P3_wrapper_forwarding"] = p3
    # P4 revision and ordering
    cat.clear_catalog()
    g0 = cat.read_catalog_revision_under_lock()
    owning()
    g1 = cat.read_catalog_revision_under_lock()
    emb()
    g2 = cat.read_catalog_revision_under_lock()
    res["P4_revision"] = {"gen": [g0.generation, g1.generation, g2.generation], "digest_changed_by_embedding": g1.state_digest != g2.state_digest}
    res["P4_second_embedding_registration"] = attempt(lambda: emb())
    cat.clear_catalog()
    res["P4_embedding_without_owning_row"] = attempt(lambda: emb())
    res["P4_catalog_after_failed_embedding"] = cat.list_slugs()
    cat.clear_catalog()
    owning()
    emb()
    owning(override=True)
    res["P4_owning_override_after_embedding"] = {"categories": [c.value for c in cat.get_entry(slug).categories], "resolve_embedding": attempt(lambda: resolve(E, slug))}
    # P5 isolated registration sink
    cat.clear_catalog()
    owning()
    before = {k: [c.value for c in v.categories] for k, v in cat.catalog_snapshot().items()}
    sink: dict[str, object] = {}
    try:
        with cat.isolated_catalog_registration(sink):  # type: ignore[arg-type]
            emb()
        res["P5_sink"] = {"sink_slugs": sorted(sink), "global_unchanged": before == {k: [c.value for c in v.categories] for k, v in cat.catalog_snapshot().items()}}
    except Exception as exc:  # noqa: BLE001
        res["P5_sink"] = {"error": f"{type(exc).__name__}: {str(exc)[:150]}"}
    # P6 transient window between the two catalog writes
    cat.clear_catalog()
    owning()
    seen: dict[str, object] = {}
    original = getattr(emb_mod, "register_integration", None)

    def observe(entry, *, override=False):  # noqa: ANN001, ANN202
        seen["categories_visible_before_wrapper"] = [c.value for c in cat.get_entry(slug).categories]
        seen["resolve_embedding_in_window"] = attempt(lambda: resolve(E, slug))
        original(entry, override=override)

    if original is not None:
        emb_mod.register_integration = observe  # type: ignore[assignment]
    try:
        emb()
    finally:
        if original is not None:
            emb_mod.register_integration = original  # type: ignore[assignment]
    if original is None:
        seen["note"] = "single catalog write in this checkout (no register_integration in the embedding register module)"
    seen["resolve_embedding_after"] = attempt(lambda: resolve(E, slug))
    res["P6_transient_window"] = seen
    # P7 concurrent soak
    expected = f"intergrax.integrations.providers.embedding_provider.{slug}.integration."
    bad = 0
    for _ in range(200):
        cat.clear_catalog()
        owning()
        results: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                try:
                    row = cat.get_entry(slug)
                except Exception:  # noqa: BLE001
                    continue
                if E in row.categories:
                    results.append(attempt(lambda: resolve(E, slug)).get("ok", "ERR"))

        t = threading.Thread(target=reader)
        t.start()
        emb()
        stop.set()
        t.join()
        if any(not str(r).startswith(expected) for r in results):
            bad += 1
    res["P7_concurrent_resolve"] = {"rounds": 200, "rounds_with_wrong_or_error_result": bad}
    out[f"slug_{slug}"] = res

# P8 repeated / overriding default registration
fresh_full()
rg = {}
for label, kw in (("first_full", {}), ("second_full_noop", {}), ("third_full_override", {"override": True})):
    try:
        if label != "first_full":
            register_default_integrations(preset="full", **kw)
        rg[label] = {
            s: {
                "categories": [c.value for c in cat.get_entry(s).categories],
                "rows": cat.list_slugs().count(s),
                "embedding": attempt(lambda s=s: resolve(E, s)),
            }
            for s in SLUGS
        } | {"generation": cat.read_catalog_revision_under_lock().generation}
    except Exception as exc:  # noqa: BLE001
        rg[label] = f"{type(exc).__name__}: {str(exc)[:150]}"
out["P8_rebootstrap"] = rg
Path(sys.argv[2]).write_text(json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8")
print("done", sys.argv[2])
