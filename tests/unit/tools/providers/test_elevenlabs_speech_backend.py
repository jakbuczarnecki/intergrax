from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from intergrax.speech_adapters.contracts.io import SpeechSynthesizeInput
from intergrax.speech_adapters.providers.elevenlabs_speech import ElevenLabsSpeechAdapter


def test_elevenlabs_synthesize_posts_and_returns_uri() -> None:
    backend = ElevenLabsSpeechAdapter(api_key="test-key")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_response

    with patch("intergrax.speech_adapters.providers.elevenlabs_speech.httpx.Client", return_value=mock_client):
        output = backend.synthesize(SpeechSynthesizeInput(text="hello", voice_id="default"))

    assert output.audio_uri.startswith("elevenlabs://audio/")
    assert output.character_count == len("hello")
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert "21m00Tcm4TlvDq8ikWAM" in call_kwargs[0][0]
    assert call_kwargs[1]["headers"]["xi-api-key"] == "test-key"
