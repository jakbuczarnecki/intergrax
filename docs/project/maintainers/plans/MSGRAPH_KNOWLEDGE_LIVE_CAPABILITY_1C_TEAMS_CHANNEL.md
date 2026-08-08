# Microsoft Graph Teams Channel Live Capability 1C

STATUS: ACCEPTED / CLOSED

The v1 Teams Channel list capability returns at most one root post.
It does not list replies or all channel messages.

## Canonical naming

```text
capability ID: vendor.ms365_graph.teams_channel.list
request schema: schema://vendor-knowledge/live/ms365_graph/teams_channel/list/request/v1
result schema: schema://vendor-knowledge/live/ms365_graph/teams_channel/list/result/v1
shared identity changed: no
shared schema validator changed: no
exact semantic limit: one root post, one binding-fixed team/channel scope,
one adapter call, cursor=None, limit=1
```

The request is strict, immutable and zero-field:
`MsGraphTeamsChannelListLiveRequestV1`.

The handler invokes the existing adapter boundary exactly once:

```python
await adapter.read_page(
    integration=integration,
    source=source,
    cursor=None,
    limit=1,
)
```

The descriptor limits are all hard-bounded to one where they govern provider
or result cardinality:

```text
max_result_items = 1
max_upstream_items = 1
max_provider_page_size = 1
max_provider_pages = 1
max_provider_requests = 1
```

## Operation matrix

```text
bounded search:                  UNSUPPORTED_BY_PROVIDER
bounded list:                    SUPPORTED, at most one root post
bounded replies list:            DEFERRED
full channel traversal:         UNSUPPORTED_BY_CURRENT_SURFACE
exact message read:              UNSUPPORTED_BY_PROVIDER
thread read:                     DEFERRED
content read:                    DEFERRED
attachment / hosted content read: DEFERRED
```

Only `vendor.ms365_graph.teams_channel.list` is registered. The Microsoft
Graph family publication order is deterministic:

```text
vendor.ms365_graph.drive.list
vendor.ms365_graph.mail.list
vendor.ms365_graph.teams_channel.list
vendor.ms365_graph.teams_chat.list
vendor.ms365_graph.calendar.list
```

The live result may contain root-post metadata and explicit deletion evidence;
it does not contain reply traversal, full-channel traversal, message bodies,
or attachment/hosted-content bytes. The complete Microsoft Graph live family
now publishes five deterministic list bundles through one shared registry and
executor; Teams Chat and Calendar use their own binding-scoped opaque scopes.
