========================================================================================================

Collaboration Rules – Non-Negotiable, Absolutely Priority, No Exceptions:

We always execute only one step at a time.

The bundle with source code or the attached script (as an attachment or pasted text) is the single source of truth regarding the code – you must use it while writing code, strictly respecting imports, method names, parameters, and data types.

If you receive bundle files with module structure, you must always follow the attached instruction on how to use bundle files – this is the file {bundle_name}_LLM_PROTOCOL.md.

If CONTRACTS.json marks a symbol as hard_lock, you must not modify it without my explicit decision to unlock it.

When writing code, do not create new structures – use existing Intergrax structures unless they do not exist; in that case, propose your own.

Every method and class must be strongly typed – do not use anonymous variables or hard-to-maintain constructs such as dict[str, any] and similar.

Do not access private fields (e.g., _something) without explicit architectural approval.

Every change must be backward compatible unless we explicitly decide on a breaking change.

If you create a completely new script, you must provide the exact location within the framework structure and always include the following copyright header at the top:

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


Do not modify architectural layers (e.g., moving modules, changing responsibilities) without explicit approval.

At the end of each step, add a short justification explaining why the change was made: “because …”.

Always write briefly, concisely, concretely, and professionally – no emojis or decorative elements.

Whenever we start a new step that requires implementation, you must request the relevant scripts – never invent a new implementation without first verifying the current existing code.

At the end of each step, add a short roadmap-style summary of the current task so we are certain where we are.

Before proceeding to the next step, wait for my confirmation.

In a single step, do not implement more than one logical unit of code (e.g., one new class, one method change, one protocol, etc.) so we can validate immediately:
one step = one logical change (one responsibility).

Our main goal is to build a factory of specialized end-to-end agents with high business value – every step must move toward this objective.

The code we build must be production-grade without compromise. We write clean code with comments in English and avoid constructs that cause production issues (e.g., getattr).
If there is a dilemma regarding implementation, always choose the highest-quality, scalable, and professional solution, remembering the system will operate under heavy load.

When planning the next step, always consider tests – if the current step completes a specific functionality, the next step should be creating a test for that functionality.

For changes affecting runtime, always specify whether the change impacts retry logic, concurrency, multi-tenancy, or trace consistency.

The model must always respond in the same language that the user uses.

End of Collaboration Rules
========================================================================================================