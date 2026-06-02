# © Artur Czarnecki. All rights reserved.

"""Distributed modality worker entrypoints (Celery-aligned)."""

from intergrax.model_inference.workers.task_runner import run_modality_job

__all__ = ["run_modality_job"]
