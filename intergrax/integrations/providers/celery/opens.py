# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Celery openers — internal to the celery integration package.

Only this module may construct ``Celery`` apps and ``CeleryTaskQueue`` instances
for runtime wiring.
"""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.providers.celery.config import CeleryIntegrationConfig
from intergrax.queueing.contracts.task_queue import TaskQueue


def open_celery_app(
    config: CeleryIntegrationConfig,
    *,
    app: Optional[object] = None,
    task_always_eager: bool = False,
) -> object:
    if app is not None:
        return app
    from celery import Celery

    celery_app = Celery(
        config.app_name,
        broker=config.broker_url,
        backend=config.backend_url,
    )
    if task_always_eager:
        celery_app.conf.task_always_eager = True
        celery_app.conf.task_eager_propagates = True
        celery_app.conf.task_store_eager_result = True
    return celery_app


def open_celery_task_queue(
    config: CeleryIntegrationConfig,
    *,
    app: Optional[object] = None,
    task_always_eager: bool = False,
) -> TaskQueue:
    from intergrax.queueing.providers.celery.celery_task_queue import CeleryTaskQueue

    return CeleryTaskQueue(app=open_celery_app(config, app=app, task_always_eager=task_always_eager))
