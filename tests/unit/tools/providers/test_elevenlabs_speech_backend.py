from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from intergrax.tools.providers.speech.backends import ElevenLabsSpeechBackend
from intergrax.tools.providers.speech.contracts import SpeechSynthesizeInput


def test_elevenlabs_synthesize_posts_and_returns_uri() -> None:
    backend = ElevenLabsSpeechBackend(api_key="test-key")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_response

    with patch("intergrax.tools.providers.speech.backends.httpx.Client", return_value=mock_client):
        output = backend.synthesize(SpeechSynthesizeInput(text="hello", voice_id="default"))

    assert output.audio_uri.startswith("elevenlabs://audio/")
    assert output.character_count == len("hello")
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert "21m00Tcm4TlvDq8ikWAM" in call_kwargs[0][0]
    assert call_kwargs[1]["headers"]["xi-api-key"] == "test-key"
