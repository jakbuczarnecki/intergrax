# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.runtime.long_running.models import NotificationMessage, TaskCheckpoint
from intergrax.runtime.long_running.notification import (
    ENV_SLACK_WEBHOOK_URL,
    ENV_TEAMS_WEBHOOK_URL,
    LoggingNotificationAdapter,
    NotificationAdapter,
    SlackNotificationAdapter,
    TeamsNotificationAdapter,
    resolve_notification_adapter,
)
from intergrax.runtime.long_running.partial_results import (
    PartialResultSnapshot,
    build_task_progress_view,
    partial_result_from_checkpoint,
)
from intergrax.runtime.long_running.store import (
    DEFAULT_TASK_CHECKPOINTS_DB,
    ENV_TASK_CHECKPOINTS_DB,
    SQLiteTaskCheckpointStore,
    open_task_checkpoint_store,
    resolve_task_checkpoints_db_path,
)

__all__ = [
    "DEFAULT_SCHEDULER_POLL_SECONDS",
    "DEFAULT_TASK_CHECKPOINTS_DB",
    "ENV_SCHEDULER_POLL_SECONDS",
    "ENV_SLACK_WEBHOOK_URL",
    "ENV_TASK_CHECKPOINTS_DB",
    "ENV_TEAMS_WEBHOOK_URL",
    "LoggingNotificationAdapter",
    "LongRunningCoordinator",
    "LongRunningScheduler",
    "NotificationAdapter",
    "NotificationMessage",
    "PartialResultSnapshot",
    "ScheduledResume",
    "ScheduledResumeStatus",
    "SlackNotificationAdapter",
    "SQLiteTaskCheckpointStore",
    "TaskCheckpoint",
    "TaskResumeExecutor",
    "TeamsNotificationAdapter",
    "UnifiedTaskResumeExecutor",
    "build_task_progress_view",
    "open_task_checkpoint_store",
    "partial_result_from_checkpoint",
    "resolve_notification_adapter",
    "resolve_task_checkpoints_db_path",
]


def __getattr__(name: str):
    if name == "LongRunningCoordinator":
        from intergrax.runtime.long_running.coordinator import LongRunningCoordinator

        return LongRunningCoordinator
    if name in {"LongRunningScheduler", "TaskResumeExecutor", "UnifiedTaskResumeExecutor"}:
        from intergrax.runtime.long_running import scheduler as _scheduler

        return getattr(_scheduler, name)
    if name in {
        "DEFAULT_SCHEDULER_POLL_SECONDS",
        "ENV_SCHEDULER_POLL_SECONDS",
    }:
        from intergrax.runtime.long_running import scheduler as _scheduler

        return getattr(_scheduler, name)
    if name in {"ScheduledResume", "ScheduledResumeStatus"}:
        from intergrax.runtime.long_running.scheduled_resume import (
            ScheduledResume,
            ScheduledResumeStatus,
        )

        if name == "ScheduledResume":
            return ScheduledResume
        return ScheduledResumeStatus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
