# LKW-SLACK-WORKFLOW-1A — live proof

```text
Status: PARTIAL
Verified commit: 184cfa1d13625c6b445bb65bd79167f76e3d14e8
Date: 2026-07-23 (UTC)
Environment: Docker Compose project intergrax_lkw
Host/container: intergrax_lkw-local_workspace-1 (image local_workspace-application:latest rebuilt for companion code)
Anonymized team ID: team:T0B…HU7#d7c3d6395d
Anonymized human user ID: user:U0B…MQ9#8f0e4a35b0
Anonymized event ID: evt:Ev0…T1P#336d0ccb5f
Anonymized channel/thread ID: OPERATOR_CONFIRMED (bot token lacks im:read; API history not polled)
Authorization result: PASS (approved team/user; dedupe claim + Ask executed for configured tenant/workspace)
Dedupe result: first claim created; status=completed; ask_run_id bound
Acknowledgement count: 1 (operator confirmed ack text in thread; no acknowledgement_failed log)
Ask call count: 1 (exact live question)
Ask run_id: run_8f6824ee9dcd41a1b07e7ba1222d5b14
Ask typed status: completed
Final reply count: 1 (operator confirmed final reply + safe Sources label; dedupe mark_completed after send)
Same-thread result: PASS — operator confirmed ack + final under the same inbound DM thread; platform mapping (thread_id = message ts → chat.postMessage thread_ts)
Answer correctness: PASS (persisted answer matches expected verification code from safe source)
Citation count: 1
Safe file names: lkw_persistence_proof_20260720143245.txt
Unsafe fields observed: none in Slack-rendered path / persisted answer text (no source_path, excerpt, chunk/document IDs, stack traces, or tokens in operator-facing output)
Persisted run verification: GET /v1/local_workspace/asks/{run_id} → exists; workspace_id matches configured active workspace; status=completed; answer persisted; citation_count=1; file_name matches Slack safe label
Shutdown/optionality: PASS — controlled stop; restart with companion disabled; HTTP ready; core readiness ready; mcp disabled/healthy; slack_companion enabled=false detail=disabled
Secrets review: PASS — no tokens/API keys/.env contents/full payloads/answers/questions/excerpts/absolute paths recorded here
dedupe_live_redelivery: NOT_EXECUTED (no safe code-free live redelivery harness for the same Slack event_id)
dedupe_code_and_concurrency_tests: VERIFIED (unit/integration suite)
SUMMARY=PARTIAL
```

## Notes

- Happy-path Slack Ask workflow executed against a real approved DM after Socket Mode companion start.
- Operator visual confirmation (2026-07-24): acknowledgement, grounded final answer, and safe source label `lkw_persistence_proof_20260720143245.txt` in the same Slack thread.
- Image rebuild was required: running container image predated Slack companion settings/code.
- Local compose `.env` Qdrant/Ollama hosts were pointed at compose DNS for container networking (gitignored; not committed).
- Roadmap remains without `LIVE_VERIFIED` because live event redelivery was not executed.
