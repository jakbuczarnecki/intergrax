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

Required source files:
<minimal list of files required to continue>

Relevant constraints:
<architectural or protocol constraints if any>

Open questions:
<missing context if any, otherwise "none">

Stop after producing the handoff block.

After producing the block stop immediately.
Do not generate any text after the block.

The handoff must be language-neutral and use English labels exactly as defined.