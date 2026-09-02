# © Artur Czarnecki. All rights reserved.

"""Shared scenario identifiers — business-oriented, not provider-specific."""

from __future__ import annotations

from datetime import UTC, datetime

TENANT_A = "e2e-tenant-a"
TENANT_B = "e2e-tenant-b"
WORKSPACE_A = "e2e-workspace-a"
WORKSPACE_B = "e2e-workspace-b"
PRINCIPAL_ALICE = "principal-alice"
PRINCIPAL_BOB = "principal-bob"
PRINCIPAL_DELEGATOR = "principal-delegator"
PRINCIPAL_DELEGATE = "principal-delegate"

OPERATION_MUTATE = "collaborative.resource.mutate"
OPERATION_READ = "collaborative.resource.read"
AUTHORITY_SCOPE_MUTATE = "resource.mutate"
AUTHORITY_SCOPE_READ = "resource.read"
RESOURCE_A = "resource-a"
RESOURCE_B = "resource-b"

POLICY_RULE_HITL = "runtime.hitl"
POLICY_BUNDLE_ID = "bundle-collab-e2e"
POLICY_BUNDLE_V1 = "1.0.0"
POLICY_BUNDLE_V2 = "2.0.0"
POLICY_BUNDLE_D1 = "sha256:" + ("11" * 32)
POLICY_BUNDLE_D2 = "sha256:" + ("22" * 32)

SIDE_EFFECT_SCOPE_1 = "side-effect-scope-1"
SIDE_EFFECT_SCOPE_2 = "side-effect-scope-2"
SIDE_EFFECT_DIGEST_1 = "sha256:" + ("ab" * 32)
SIDE_EFFECT_DIGEST_2 = "sha256:" + ("cd" * 32)

FIXED_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
