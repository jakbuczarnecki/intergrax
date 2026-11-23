# © Artur Czarnecki. All rights reserved.

from tests.qualification.trace_x._trace_x_p6_discovery import _restart_resume_marker_present


def test_redelivery_comment_does_not_match_redeliver_marker() -> None:
    text = "# policy: no automatic redelivery of stale events"
    assert _restart_resume_marker_present(text, "redeliver") is False


def test_redeliver_token_matches_word_boundary() -> None:
    text = "await broker.redeliver(message_id)"
    assert _restart_resume_marker_present(text, "redeliver") is True


def test_other_markers_remain_literal_substring() -> None:
    assert _restart_resume_marker_present("resume_token = x", "resume_token") is True
