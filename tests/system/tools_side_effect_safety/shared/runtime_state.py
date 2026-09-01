# © Artur Czarnecki. All rights reserved.

"""Minimal runtime state for proof tool invocations."""

from __future__ import annotations

from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class ProofRuntimeState:
  def __init__(
      self,
      *,
      tenant_id: str,
      run_id: str,
      task_id: str,
      policy_bundle: object | None = None,
      declarative_hitl_grant: DeclarativeHitlApprovalGrant | None = None,
  ) -> None:
      self._tenant_id = tenant_id
      self.run_id = run_id
      self.request = type("Req", (), {"task_id": task_id})()
      self.declarative_hitl_grant = declarative_hitl_grant
      self._policy_bundle = policy_bundle

  @property
  def tenant_id(self) -> str:
      return self._tenant_id

  @property
  def task_id(self) -> str:
      return self.request.task_id

  @property
  def context(self):
      return type(
          "Ctx",
          (),
          {"config": type("Cfg", (), {"policy_bundle": self._policy_bundle})()},
      )()

  def set_policy_bundle(self, bundle: object | None) -> None:
      self._policy_bundle = bundle

  def trace_event(
      self,
      *,
      component: TraceComponent | None = None,
      step: str = "",
      message: str = "",
      level: TraceLevel | None = None,
      payload: object | None = None,
      artifact_refs: list | None = None,
  ) -> None:
      del component, step, message, level, payload, artifact_refs
