from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_hosted_content import (
    _extract_hosted_contents_path,
    validate_msgraph_teams_channel_hosted_contents_continuation,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphTeamsChannelMessageKind,
)
from urllib.parse import urlparse

_TEAM = "team-abc-123"
_CHANNEL = "channel-abc-123"
_MSG = "root-msg-001"
url = (
    f"https://graph.microsoft.com/v1.0/teams('{_TEAM}')/channels('{_CHANNEL}')"
    f"/messages('{_MSG}')/hostedContents?$skiptoken=x"
)
p = urlparse(url).path
print("path", p)
print("extract", _extract_hosted_contents_path(p, graph_base_path="/v1.0"))
cont = MsGraphKnowledgeContinuation(
    kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE, url=url
)
validate_msgraph_teams_channel_hosted_contents_continuation(
    cont,
    team_id=_TEAM,
    channel_id=_CHANNEL,
    thread_root_id=_MSG,
    message_id=_MSG,
    message_kind=MsGraphTeamsChannelMessageKind.ROOT,
    graph_base_url="https://graph.microsoft.com/v1.0",
)
print("ok")
