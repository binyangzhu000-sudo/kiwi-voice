"""Tests for the optional Atlas Cloud TTS provider."""

import io
import wave
from unittest.mock import MagicMock, call, patch

import numpy as np
import requests

from kiwi.tts.atlas_cloud import AtlasCloudTTSClient, AtlasCloudTTSConfig


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(np.array([0, 1000, -1000], dtype="<i2").tobytes())
    return output.getvalue()


def _response(payload=None, content=b""):
    response = MagicMock()
    response.json.return_value = payload
    response.content = content
    response.raise_for_status.return_value = None
    return response


@patch("kiwi.tts.atlas_cloud.time.sleep")
@patch("kiwi.tts.atlas_cloud.requests.get")
@patch("kiwi.tts.atlas_cloud.requests.post")
def test_submit_once_poll_and_decode_wav(mock_post, mock_get, mock_sleep):
    mock_post.return_value = _response(
        {
            "code": 200,
            "data": {"id": "prediction-1", "status": "processing"},
        }
    )
    mock_get.side_effect = [
        _response(
            {
                "code": 200,
                "data": {
                    "id": "prediction-1",
                    "status": "completed",
                    "outputs": ["https://cdn.example/audio.wav"],
                },
            }
        ),
        _response(content=_wav_bytes()),
    ]
    client = AtlasCloudTTSClient(AtlasCloudTTSConfig(api_key="atlas-key", voice_id="leo", poll_interval=0, max_polls=2))

    audio, sample_rate = client.synthesize("Hello from Kiwi", style="whisper", language="English", use_cache=False)

    assert mock_post.call_count == 1
    request = mock_post.call_args.kwargs
    assert request["json"] == {
        "model": "xai/tts-v1",
        "text": "<whisper>Hello from Kiwi</whisper>",
        "language": "en",
        "voice_id": "leo",
        "codec": "wav",
        "sample_rate": 24000,
        "speed": 1.0,
        "text_normalization": True,
    }
    assert request["headers"]["Authorization"] == "Bearer atlas-key"
    assert request["headers"]["User-Agent"].startswith("kiwi-voice/")
    assert mock_get.call_args_list[0] == call(
        "https://api.atlascloud.ai/api/v1/model/prediction/prediction-1",
        headers={
            "Authorization": "Bearer atlas-key",
            "User-Agent": "kiwi-voice/0.5 (+https://github.com/ekleziast/kiwi-voice)",
        },
        timeout=60,
    )
    assert sample_rate == 24000
    assert audio is not None and audio.dtype == np.float32 and len(audio) == 3


@patch("kiwi.tts.atlas_cloud.requests.post")
def test_paid_submit_is_not_retried(mock_post):
    mock_post.side_effect = requests.ConnectionError("network down")
    client = AtlasCloudTTSClient(AtlasCloudTTSConfig(api_key="atlas-key"))

    audio, sample_rate = client.synthesize("hello", use_cache=False)

    assert (audio, sample_rate) == (None, 0)
    assert mock_post.call_count == 1


@patch("kiwi.tts.atlas_cloud.time.monotonic")
@patch("kiwi.tts.atlas_cloud.time.sleep")
@patch("kiwi.tts.atlas_cloud.requests.get")
def test_polling_stops_at_overall_deadline(mock_get, mock_sleep, mock_monotonic):
    # start=0, first poll at 1s (within budget), second check at 100s (past 90s budget)
    mock_monotonic.side_effect = [0.0, 1.0, 1.0, 100.0]
    mock_get.return_value = _response({"code": 200, "data": {"id": "p", "status": "processing"}})
    client = AtlasCloudTTSClient(AtlasCloudTTSConfig(api_key="atlas-key", poll_interval=0.5))

    assert client._poll_prediction("p") is None
    assert mock_get.call_count == 1
    assert mock_get.call_args.kwargs["timeout"] == 60
