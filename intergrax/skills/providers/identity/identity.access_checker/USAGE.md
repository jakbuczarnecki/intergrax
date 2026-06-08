# `identity.access_checker`

**Bundle:** `identity` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Identity and tenancy verification for multi-tenant hosts.

## How it works

identity.* via IdentityProvider integration.

## How to use

identity_skill_profile(); wire identity_provider slug.

## What you get

Token verification without agent-local auth code.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `identity.verify_token` | Validate bearer token |
| `identity.get_user` | Resolve user profile |
| `identity.list_tenants` | List accessible tenants |

## Related skills

- `platform.secrets_flags`
- `hitl.approval_gate`
