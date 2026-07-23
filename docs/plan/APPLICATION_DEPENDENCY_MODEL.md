# Application Dependency Model — Plan

**Status:** Active (packaging architecture)  
**Architecture (1:1):** [`architecture/APPLICATION_DEPENDENCY_MODEL.md`](../architecture/APPLICATION_DEPENDENCY_MODEL.md)  
**Last updated:** 2026-07-23

---

## Goal

Make every Tier-3 application the source of truth for:

* Intergrax base dependency
* selected platform capability extras
* application-only third-party packages

Remove manual Dockerfile / script `--extra` assembly as the dependency contract.

## Delivery checklist

| Item | Status |
|------|--------|
| Root uv workspace members | done |
| Application `pyproject.toml` for every real Tier-3 host | done |
| Scaffold emits application `pyproject.toml` | done |
| Dockerfiles sync `--project applications/<app>` | done |
| Shared root `uv.lock` | done |
| Isolation proof (Slack vs non-Slack) | required in verification |
| Per-application lockfiles | deferred (monorepo phase) |
| Aggressive platform-base slimming | deferred (correctness first) |

## Non-goals

* Renaming application import packages
* Splitting applications into separate repositories
* Changing product / Slack / Ask runtime behavior
* Publishing Intergrax to PyPI in this task

## Follow-ups

* Move optional LLM / RAG / parser packages from root base into extras only when import-safe.
* Optional application extras for deployment overlays (observability / proof) when Docker variants diverge.
