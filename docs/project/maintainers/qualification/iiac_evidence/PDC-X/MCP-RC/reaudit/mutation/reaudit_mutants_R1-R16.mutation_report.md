# MCP-RC mutation report (head f1dde5a3e19924b82e902be18f11a21b64b822ba)

| mutant | invariant | status |
| --- | --- | --- |
| R1-sanitizer-passes-non-toolerror-through | sanitized_tool_errors converts every non-wire exception | killed |
| R2-custom-gate-bypassed-for-every-tool | custom host tools are gated unless explicitly public | killed |
| R3-outcome-unknown-reported-as-success | outcome_unknown after the bound is never a success | killed |
| R4-public-opt-out-inverted | only listed custom tools are public | killed |
| R5-toolerror-not-passed-through | wire ToolErrors keep their MCP code | killed |
| R6-host-cancelled-after-bound-propagates | host ending cancelled after the bound is MCP_DEADLINE_EXCEEDED | killed |
| R7-custom-gate-evaluated-after-handler | the principal gate runs before the custom handler | killed |
| R8-invoke-catalog-tool-unsanitized | invoke_catalog_tool never echoes raw exception text | survived |
| R9-run-agent-unsanitized | run_agent never echoes raw exception text | killed |
| R10-list-agents-unsanitized | list_agents never echoes raw exception text | killed |
| R11-custom-handler-unsanitized | custom handler exceptions are never echoed | killed |
| R12-custom-gate-evaluated-in-local-dev-mode | the custom gate uses the server identity mode | killed |
| R13-failed-host-mislabelled-cancelled | a failed host after the bound reports outcome=failed | killed |
| R14-describe-catalog-unsanitized | describe_catalog_tool never echoes raw exception text | killed |
| R15-list-catalog-unsanitized | list_catalog_tools never echoes raw exception text | killed |
| R16-bounded-observe-swallow-widened-to-base | observing the host never hides a non-host failure (control: expected equivalent/benign) | survived |
