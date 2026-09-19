# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5G subprocess worker — fresh-process durable restart phases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intergrax.integrations.providers.relational_store.sqlite.opens import (
    open_organization_profile_store_at,
    open_task_memory_store_at,
)
from intergrax.runtime.organization.organization_profile import (
    OrganizationIdentity,
    OrganizationProfile,
)
from intergrax.runtime.organization.organization_profile_manager import OrganizationProfileManager
from intergrax.runtime.task_memory.coordinator import TaskMemoryCoordinator


def _load_fixture(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_write(db_path: Path, fixture: dict[str, object]) -> None:
    store = open_task_memory_store_at(db_path)
    try:
        TaskMemoryCoordinator.write(
            store,
            tenant_id=str(fixture["tenant_id"]),
            task_id=str(fixture["task_id"]),
            namespace=str(fixture["namespace"]),
            key=str(fixture["key"]),
            value=dict(fixture["value"]),
            provenance=dict(fixture.get("provenance") or {}),
        )
    finally:
        store.close()


def _task_read(db_path: Path, fixture: dict[str, object]) -> None:
    store = open_task_memory_store_at(db_path)
    try:
        loaded = TaskMemoryCoordinator.read(
            store,
            tenant_id=str(fixture["tenant_id"]),
            task_id=str(fixture["task_id"]),
            namespace=str(fixture["namespace"]),
            key=str(fixture["key"]),
        )
    finally:
        store.close()
    if loaded is None:
        print(json.dumps({"present": False}))
        return
    print(
        json.dumps(
            {
                "present": True,
                "value": loaded.value,
                "record_id": loaded.record_id,
                "provenance": loaded.provenance,
            },
        ),
    )


def _org_save(db_path: Path, fixture: dict[str, object]) -> None:
    store = open_organization_profile_store_at(db_path)
    manager = OrganizationProfileManager(store)
    try:
        org_id = str(fixture["organization_id"])
        profile = OrganizationProfile(
            identity=OrganizationIdentity(
                organization_id=org_id,
                name=str(fixture.get("name") or org_id),
            ),
            system_instructions=str(fixture["system_instructions"]),
            tags=list(fixture.get("tags") or []),
        )
        import asyncio

        asyncio.run(manager.save_profile(profile))
    finally:
        store.close()


def _org_read(db_path: Path, fixture: dict[str, object]) -> None:
    store = open_organization_profile_store_at(db_path)
    manager = OrganizationProfileManager(store)
    try:
        import asyncio

        loaded = asyncio.run(
            manager.get_profile(str(fixture["organization_id"])),
        )
    finally:
        store.close()
    print(
        json.dumps(
            {
                "system_instructions": loaded.system_instructions,
                "tags": loaded.tags,
                "name": loaded.identity.name,
            },
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "domain",
        choices=("task_write", "task_read", "org_save", "org_read"),
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args(argv)
    fixture = _load_fixture(args.fixture)
    if args.domain == "task_write":
        _task_write(args.db, fixture)
    elif args.domain == "task_read":
        _task_read(args.db, fixture)
    elif args.domain == "org_save":
        _org_save(args.db, fixture)
    elif args.domain == "org_read":
        _org_read(args.db, fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
