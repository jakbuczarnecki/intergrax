# EQO-W3-WB06 — Configuration (band C)

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Band:** C isolated pytest (`eqo-w3`), shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Finding scope

| ID | Probe focus |
|----|-------------|
| A4-025 | CONFIG-X closed-world env scan is limited to inventoried surfaces; about 25 ambient enviro… |
| A4-042 | intergrax/globals has no owner and hardcodes Poland-specific import-time defaults |
| A5-002 | EBH-2E cert-time gap: Gemini, OpenAI Responses and LangChain Ollama stream paths bypassed … |

## EXE @ TRAIN_PIN

Prereg SHA256: `51d8f0b813cc4721650d73a3eab662433b00b9cf7d96ac1ab7607c34376ca38e` (`DES_PREREG.json`).

**Wave-6 rerun (2026-11-21):** 1 failed, 31 passed in 21.09s (exit 1). CONFIG-X qualification and adversarial gates green; `test_unresolved_capabilities_fail_closed` fails closed on missing optional extra `llm-langchain-ollama` (environment probe, not in prereg sync set).

## Disposition (pre-AUD)

| Finding | DIS (interim) |
|---------|----------------|
| A4-025 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A4-042 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |
| A5-002 | `BLOCKED — pytest slice failed on TRAIN_PIN; hypothesis remains open pending triage` |

Evidence log: `pytest_wb06_exe.log`. AUD session E pending for all rows.