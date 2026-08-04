<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Build with Intergrax — Builder Quick Start

This guide is for AI engineers and application developers who want to build or extend an application using the existing Intergrax foundation.

Intergrax is intended to support specialized applications built on reusable foundations. This page helps you choose a small, reviewable first build and identify the right ownership boundary before you go deeper.

It does not create a new scaffold or a new execution contract. All setup and verification steps below come from existing canonical repository documentation.

## At a glance

| Item | Meaning |
|------|---------|
| Audience | AI engineers and application developers |
| Goal | Begin one small, bounded application change |
| First decision | Extend an application, compose a new workflow, or evaluate a foundation |
| Expected outcome | Know where the change belongs and what to verify |
| Product trial | [LKW Quick Start](applications/local_workspace_application/docs/QUICKSTART.md) |
| Deeper builder guide | [BUILD_WITH_INTERGRAX.md](BUILD_WITH_INTERGRAX.md) |
| Broader evaluation | [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) |
| Technical depth | [Technical Documentation Map](docs/DOCUMENTATION_MAP.md) |

## Builder flow

```mermaid
flowchart LR
    A[Choose a concrete workflow] --> B[Choose the application boundary]
    B --> C[Reuse one Intergrax foundation]
    C --> D[Make one small change]
    D --> E[Run the nearest existing check]
    E --> F[Review the result and go deeper]
```

This is the recommended progression, not a claim that all applications use an identical implementation.

## Choose your starting route

### Extend Local Knowledge Workspace

Change or extend the existing specialized LKW application while preserving the separation between application workflow and reusable platform foundations.

Primary deeper references:

- [LKW application architecture](applications/local_workspace_application/docs/ARCHITECTURE.md)
- [BUILD_WITH_INTERGRAX.md](BUILD_WITH_INTERGRAX.md)

### Build a specialized application

Start with a concrete workflow and compose existing Intergrax capabilities around it rather than modifying the platform core first.

Primary references:

- [BUILD_WITH_INTERGRAX.md](BUILD_WITH_INTERGRAX.md)
- [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)

### Evaluate a foundation before building

Run an existing bounded evaluation before deciding whether a platform capability fits the intended application.

Primary reference:

- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

This guide does not duplicate the evaluation catalog.

## Where the change belongs

| Change type | Primary location | Reason |
|-------------|------------------|--------|
| Product workflow or application behavior | `applications/` | Keeps specialized product logic outside reusable platform foundations |
| Reusable capability or contract | `intergrax/` | Makes the capability reusable across applications |
| Evidence for changed behavior | Nearest existing test or proof boundary | Keeps verification close to the behavior |
| Public explanation | Owning public document | Prevents duplicated or drifting claims |

The application/platform boundary is a decision to make for each change; not every change automatically belongs in `intergrax/`.

## First builder checkpoint

1. Choose one concrete user workflow.
2. Select the existing application or reusable foundation closest to it.
3. Read only the nearest architecture document.
4. Prepare the repository using the current canonical public setup.
5. Make one small change inside the correct ownership boundary.
6. Run the nearest existing documented verification.
7. Continue to deeper references only when the first checkpoint is clear.

### Setup and verification truth rule

The public documents do not define one universal builder setup command. They expose route-specific setup instead.

For an existing bounded repository evaluation, [EVALUATION_GUIDE.md § 30-minute bounded technical evaluation](EVALUATION_GUIDE.md#30-minute-bounded-technical-evaluation) documents this exact setup and verification sequence:

```text
uv sync --extra dev
uv run intergrax doctor
uv run pytest -m gate -q
```

These commands are owned by the Evaluation Guide. They are a documented evaluation path, not a generic builder acceptance check, product trial, platform certification, or production-readiness claim. If the intended change has a closer documented check, use that route's own documentation instead.

## Progressive disclosure

Start here:
[BUILDER_QUICKSTART.md](BUILDER_QUICKSTART.md)

Choose and understand the builder route:
[BUILD_WITH_INTERGRAX.md](BUILD_WITH_INTERGRAX.md)

Run broader evaluations:
[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

Understand public architecture:
[ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)

Perform deep technical review:
[docs/DOCUMENTATION_MAP.md](docs/DOCUMENTATION_MAP.md)

Internal plans and maintainer controls are not normal first-step builder documentation. Use them only when the public route has made the next technical question clear.

## Current boundaries

- No project scaffold is promised.
- No stable universal public SDK is claimed.
- No fixed onboarding time is validated.
- Application-specific prerequisites may differ.
- Existing tests or proofs demonstrate bounded behavior, not universal production readiness.
- The builder route is distinct from the LKW product trial.
- No production-ready application template is claimed.

## Next actions

| Goal | Next action |
|------|-------------|
| Extend LKW | [LKW application architecture](applications/local_workspace_application/docs/ARCHITECTURE.md) |
| Plan an application | [BUILD_WITH_INTERGRAX.md](BUILD_WITH_INTERGRAX.md) |
| Run an evaluation | [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) |
| Understand the platform shape | [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) |
| Enter deep technical documentation | [docs/DOCUMENTATION_MAP.md](docs/DOCUMENTATION_MAP.md) |
| Try the LKW product instead | [LKW Quick Start](applications/local_workspace_application/docs/QUICKSTART.md) |
| Return to project overview | [README.md](README.md) |
