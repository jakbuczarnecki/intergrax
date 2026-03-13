INTERGRAX SESSION HANDOFF GENERATION

We are about to end this session.

Generate a SESSION HANDOFF for the next session.

The handoff must allow a new assistant instance to perform BOOTSTRAP CHECK immediately after loading the protocol.

Follow these rules strictly.

1. Do not explain anything.
2. Do not include commentary.
3. Do not include reasoning.
4. Produce only the SESSION HANDOFF block.

The structure must be exactly:

INTERGRAX SESSION HANDOFF

User language:
<language used by the user in the current session>

Subsystem:
<current subsystem>

Development phase:
<current phase>

Last completed step:
<last finished implementation step>

Current task:
<task we were performing when session stopped>

Next planned step:
<next implementation step>

Repository bundle status:
<present / missing / partial>

Bundle artifacts available:
<list of bundle artifacts provided in the session, or "none">

Required additional files:
<files required that are not covered by the repository bundle, otherwise "none">

Relevant constraints:
<architectural or protocol constraints if any>

Open questions:
<missing context if any, otherwise "none">

Stop after producing the handoff block.

After producing the block stop immediately.
Do not generate any text after the block.

The handoff must be language-neutral and use English labels exactly as defined.

If bundle status is "present", bundle artifacts are expected to be available in the session context.

Additional rules:

If a repository bundle was used in the session, the handoff must prefer bundle artifacts over individual repository files.

Individual source files may only appear in "Required additional files" if they are not represented in the bundle.

The handoff must never request repository files that are already contained in FULL._py.