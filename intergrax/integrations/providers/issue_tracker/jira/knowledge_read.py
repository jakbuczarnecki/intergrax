# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Private Jira knowledge-read models and protocol (JIRA-KNOWLEDGE-ADAPTER-1)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JIRA_ISSUES_SOURCE_KIND = "issues"
JIRA_PROJECT_SCOPE_TYPE = "jira_project"
JIRA_KNOWLEDGE_CURSOR_VERSION = "jira.issues.cursor.v1"

_JIRA_PROJECT_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,31}$")
_JIRA_ISSUE_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,31}-[1-9][0-9]*$")
_JIRA_REMOTE_ID_RE = re.compile(r"^[1-9][0-9]*$")
_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)


def validate_jira_project_key(project_key: str) -> str:
    cleaned = str(project_key).strip()
    if not _JIRA_PROJECT_KEY_RE.fullmatch(cleaned):
        raise ValueError("invalid Jira project key")
    return cleaned


def validate_jira_issue_key(issue_key: str) -> str:
    cleaned = str(issue_key).strip()
    if not _JIRA_ISSUE_KEY_RE.fullmatch(cleaned):
        raise ValueError("invalid Jira issue key")
    return cleaned


def issue_key_project_part(issue_key: str) -> str:
    validated_key = validate_jira_issue_key(issue_key)
    return validated_key.rsplit("-", 1)[0]


def validate_jira_knowledge_issue_project_scope(
    issue: JiraKnowledgeIssue,
    *,
    project_key: str,
) -> None:
    validated_project_key = validate_jira_project_key(project_key)
    if issue.project_key != validated_project_key:
        raise ValueError("Jira knowledge issue does not belong to requested project")
    if issue_key_project_part(issue.key) != validated_project_key:
        raise ValueError("Jira knowledge issue does not belong to requested project")


class JiraKnowledgeUser(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    account_id: str | None = None
    display_name: str | None = None
    active: bool | None = None

    @field_validator("account_id", "display_name", mode="before")
    @classmethod
    def _validate_optional_identity(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("identity field must be a string when provided")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("identity field must not be empty when provided")
        return cleaned


class JiraKnowledgeIssue(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    remote_id: str
    key: str
    summary: str
    description: str
    status_id: str | None = None
    status_name: str
    issue_type_id: str | None = None
    issue_type_name: str
    project_id: str
    project_key: str
    project_name: str
    priority_name: str | None = None
    labels: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    assignee: JiraKnowledgeUser | None = None
    reporter: JiraKnowledgeUser | None = None
    resolution_name: str | None = None
    created_at: datetime
    updated_at: datetime
    web_url: str

    @field_validator("remote_id")
    @classmethod
    def _validate_remote_id(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not _JIRA_REMOTE_ID_RE.fullmatch(cleaned):
            raise ValueError("remote_id must be a positive numeric Jira ID")
        return cleaned

    @field_validator("key")
    @classmethod
    def _validate_key(cls, value: str) -> str:
        return validate_jira_issue_key(value)

    @field_validator("project_key")
    @classmethod
    def _validate_project_key(cls, value: str) -> str:
        return validate_jira_project_key(value)

    @field_validator(
        "summary",
        "status_name",
        "issue_type_name",
        "project_id",
        "project_name",
        "web_url",
    )
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("field must be a non-empty string")
        return cleaned

    @field_validator("created_at", "updated_at")
    @classmethod
    def _validate_timezone_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if value.utcoffset() is None:
            raise ValueError("timestamp must have a defined UTC offset")
        return value.astimezone(timezone.utc)


class JiraKnowledgeIssuePage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    issues: tuple[JiraKnowledgeIssue, ...] = ()
    next_page_token: str | None = Field(default=None, repr=False)
    is_last: bool

    @field_validator("next_page_token", mode="before")
    @classmethod
    def _validate_next_page_token(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("next_page_token must be a string when provided")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("next_page_token must not be empty when provided")
        return cleaned

    @model_validator(mode="after")
    def _token_rules(self) -> JiraKnowledgeIssuePage:
        if not self.is_last and not self.next_page_token:
            raise ValueError("next_page_token is required when is_last is False")
        if self.is_last and self.next_page_token is not None:
            raise ValueError("next_page_token must be None when is_last is True")
        seen_ids: set[str] = set()
        for issue in self.issues:
            if issue.remote_id in seen_ids:
                raise ValueError("duplicate issue id on page")
            seen_ids.add(issue.remote_id)
        return self


@runtime_checkable
class JiraKnowledgeReadClient(Protocol):
    def search_knowledge_issues(
        self,
        *,
        project_key: str,
        next_page_token: str | None,
        limit: int,
    ) -> JiraKnowledgeIssuePage:
        ...

    def get_knowledge_issue(
        self,
        *,
        issue_key: str,
    ) -> JiraKnowledgeIssue:
        ...


def _parse_timestamp(raw: object, *, field_name: str) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} is required")
    text = raw.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    if len(text) >= 5 and text[-5] in "+-" and text[-3] != ":":
        text = f"{text[:-2]}:{text[-2:]}"
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} timestamp is invalid") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _parse_user(raw: object) -> JiraKnowledgeUser | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("user payload must be an object")
    account_id_raw = raw.get("accountId")
    display_name_raw = raw.get("displayName")
    active_raw = raw.get("active")
    account_id: str | None
    if account_id_raw is None:
        account_id = None
    elif not isinstance(account_id_raw, str):
        raise ValueError("user account id must be a string when provided")
    else:
        account_id = account_id_raw.strip()
        if not account_id:
            raise ValueError("user account id must not be empty when provided")
    display_name: str | None
    if display_name_raw is None:
        display_name = None
    elif not isinstance(display_name_raw, str):
        raise ValueError("user display name must be a string when provided")
    else:
        display_name = display_name_raw.strip()
        if not display_name:
            raise ValueError("user display name must not be empty when provided")
    active = bool(active_raw) if isinstance(active_raw, bool) else None
    if account_id is None and display_name is None and active is None:
        return None
    return JiraKnowledgeUser(
        account_id=account_id,
        display_name=display_name,
        active=active,
    )


def _string_tuple(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("labels must be a list")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("labels must contain non-empty strings")
        values.append(item.strip())
    return tuple(values)


def _component_names(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("components must be a list")
    names: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("component payload must be an object")
        name_raw = item.get("name")
        if not isinstance(name_raw, str) or not name_raw.strip():
            raise ValueError("component name is required")
        names.append(name_raw.strip())
    return tuple(names)


def parse_jira_knowledge_issue(
    payload: Mapping[str, Any],
    *,
    issue_url: str,
    plain_description: Any,
) -> JiraKnowledgeIssue:
    if not isinstance(payload, dict):
        raise ValueError("issue payload must be an object")
    remote_id_raw = payload.get("id")
    if remote_id_raw is None or not str(remote_id_raw).strip():
        raise ValueError("issue id is required")
    key_raw = payload.get("key")
    if not isinstance(key_raw, str) or not key_raw.strip():
        raise ValueError("issue key is required")
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("issue fields are required")

    status_obj = fields.get("status")
    if not isinstance(status_obj, dict):
        raise ValueError("issue status is required")
    status_name_raw = status_obj.get("name")
    if not isinstance(status_name_raw, str) or not status_name_raw.strip():
        raise ValueError("issue status name is required")
    status_id_raw = status_obj.get("id")
    status_id = str(status_id_raw).strip() if status_id_raw is not None else None

    issue_type_obj = fields.get("issuetype")
    if not isinstance(issue_type_obj, dict):
        raise ValueError("issue type is required")
    issue_type_name_raw = issue_type_obj.get("name")
    if not isinstance(issue_type_name_raw, str) or not issue_type_name_raw.strip():
        raise ValueError("issue type name is required")
    issue_type_id_raw = issue_type_obj.get("id")
    issue_type_id = str(issue_type_id_raw).strip() if issue_type_id_raw is not None else None

    project_obj = fields.get("project")
    if not isinstance(project_obj, dict):
        raise ValueError("issue project is required")
    project_id_raw = project_obj.get("id")
    project_key_raw = project_obj.get("key")
    project_name_raw = project_obj.get("name")
    if project_id_raw is None or not str(project_id_raw).strip():
        raise ValueError("issue project id is required")
    if not isinstance(project_key_raw, str) or not project_key_raw.strip():
        raise ValueError("issue project key is required")
    if not isinstance(project_name_raw, str) or not project_name_raw.strip():
        raise ValueError("issue project name is required")

    priority_obj = fields.get("priority")
    priority_name: str | None = None
    if isinstance(priority_obj, dict):
        priority_name_raw = priority_obj.get("name")
        if isinstance(priority_name_raw, str) and priority_name_raw.strip():
            priority_name = priority_name_raw.strip()

    resolution_obj = fields.get("resolution")
    resolution_name: str | None = None
    if isinstance(resolution_obj, dict):
        resolution_name_raw = resolution_obj.get("name")
        if isinstance(resolution_name_raw, str) and resolution_name_raw.strip():
            resolution_name = resolution_name_raw.strip()

    summary_raw = fields.get("summary")
    summary = str(summary_raw) if summary_raw is not None else ""

    return JiraKnowledgeIssue(
        remote_id=str(remote_id_raw).strip(),
        key=key_raw.strip(),
        summary=summary,
        description=str(plain_description(fields.get("description"))),
        status_id=status_id,
        status_name=status_name_raw.strip(),
        issue_type_id=issue_type_id,
        issue_type_name=issue_type_name_raw.strip(),
        project_id=str(project_id_raw).strip(),
        project_key=project_key_raw.strip(),
        project_name=project_name_raw.strip(),
        priority_name=priority_name,
        labels=_string_tuple(fields.get("labels")),
        components=_component_names(fields.get("components")),
        assignee=_parse_user(fields.get("assignee")),
        reporter=_parse_user(fields.get("reporter")),
        resolution_name=resolution_name,
        created_at=_parse_timestamp(fields.get("created"), field_name="created"),
        updated_at=_parse_timestamp(fields.get("updated"), field_name="updated"),
        web_url=issue_url,
    )


def parse_jira_knowledge_issue_page(
    payload: Mapping[str, Any],
    *,
    issue_url_builder: Any,
    plain_description: Any,
) -> JiraKnowledgeIssuePage:
    if not isinstance(payload, dict):
        raise ValueError("search response must be an object")
    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list):
        raise ValueError("issues must be a list")
    is_last_raw = payload.get("isLast")
    if not isinstance(is_last_raw, bool):
        raise ValueError("isLast must be a boolean")
    next_page_token_raw = payload.get("nextPageToken")
    next_page_token: str | None
    if next_page_token_raw is None:
        next_page_token = None
    elif isinstance(next_page_token_raw, str) and next_page_token_raw.strip():
        next_page_token = next_page_token_raw.strip()
    else:
        raise ValueError("nextPageToken must be a non-empty string when present")

    seen_ids: set[str] = set()
    issues: list[JiraKnowledgeIssue] = []
    for item in raw_issues:
        if not isinstance(item, dict):
            raise ValueError("issue payload must be an object")
        remote_id_raw = item.get("id")
        if remote_id_raw is None or not str(remote_id_raw).strip():
            raise ValueError("issue id is required")
        remote_id = str(remote_id_raw).strip()
        if remote_id in seen_ids:
            raise ValueError("duplicate issue id on page")
        seen_ids.add(remote_id)
        key_raw = item.get("key")
        if not isinstance(key_raw, str) or not key_raw.strip():
            raise ValueError("issue key is required")
        issue = parse_jira_knowledge_issue(
            item,
            issue_url=issue_url_builder(key_raw.strip()),
            plain_description=plain_description,
        )
        issues.append(issue)

    return JiraKnowledgeIssuePage(
        issues=tuple(issues),
        next_page_token=next_page_token,
        is_last=is_last_raw,
    )
