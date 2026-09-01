"""Atlas Cloud TTS provider for Kiwi Voice."""

import io
import time
import wave
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

import numpy as np
import requests

from kiwi.tts.base import TTSCacheMixin
from kiwi.utils import kiwi_log


ATLAS_API_BASE = "https://api.atlascloud.ai/api/v1"
ATLAS_TTS_MODEL = "xai/tts-v1"
DEFAULT_ATLAS_VOICE_ID = "eve"
_USER_AGENT = "kiwi-voice/0.5 (+https://github.com/ekleziast/kiwi-voice)"

_LANGUAGE_CODES = {
    "Arabic": "ar-EG",
    "Chinese": "zh",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt-PT",
    "Russian": "ru",
    "Spanish": "es-ES",
    "Turkish": "tr",
}
_SUPPORTED_LANGUAGE_CODES = {
    "auto",
    "en",
    "zh",
    "ar-EG",
    "ar-SA",
    "ar-AE",
    "bn",
    "fr",
    "de",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "pt-BR",
    "pt-PT",
    "ru",
    "es-MX",
    "es-ES",
    "tr",
    "vi",
}

_STYLE_TAGS = {
    "calm": "soft",
    "excited": "build-intensity",
    "whisper": "whisper",
    "angry": "loud",
    "serious": "emphasis",
}


@dataclass
class AtlasCloudTTSConfig:
    api_key: str = ""
    voice_id: str = DEFAULT_ATLAS_VOICE_ID
    timeout: int = 60
    poll_interval: float = 0.5
    max_polls: int = 120
    cache_size: int = 100
    cache_ttl: int = 3600


class AtlasCloudTTSClient(TTSCacheMixin):
    """One-shot Atlas submission with bounded prediction polling."""

    def __init__(self, config: Optional[AtlasCloudTTSConfig] = None):
        self.config = config or AtlasCloudTTSConfig()
        self._cache_size = self.config.cache_size
        self._cache_ttl = self.config.cache_ttl
        self._init_cache()

    @staticmethod
    def _response_data(payload: Dict[str, Any]) -> Dict[str, Any]:
        code = payload.get("code")
        if code not in (None, 0, 200, "200"):
            message = payload.get("message") or payload.get("msg") or "unknown error"
            raise RuntimeError(f"Atlas Cloud API error ({code}): {message}")
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise RuntimeError("Atlas Cloud returned an invalid response payload")
        return data

    @staticmethod
    def _decode_wav(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")
        if channels > 1:
            audio = audio.reshape((-1, channels)).mean(axis=1)
        return audio.astype(np.float32), int(sample_rate)

    @staticmethod
    def _styled_text(text: str, style: str) -> str:
        tag = _STYLE_TAGS.get(style)
        return f"<{tag}>{text}</{tag}>" if tag else text

    def _poll_prediction(self, prediction_id: str) -> Optional[str]:
        url = f"{ATLAS_API_BASE}/model/prediction/{quote(prediction_id, safe='')}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "User-Agent": _USER_AGENT,
        }
        delay = max(0.0, self.config.poll_interval)
        for attempt in range(self.config.max_polls):
            try:
                response = requests.get(url, headers=headers, timeout=self.config.timeout)
                response.raise_for_status()
                prediction = self._response_data(response.json())
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                if attempt + 1 >= self.config.max_polls:
                    kiwi_log("ATLAS", f"Prediction polling failed: {exc}", level="ERROR")
                    return None
            else:
                status = str(prediction.get("status", "")).lower()
                if status in {"completed", "succeeded"}:
                    outputs = prediction.get("outputs") or []
                    if outputs and isinstance(outputs[0], str):
                        return outputs[0]
                    kiwi_log("ATLAS", "Prediction completed without an audio URL", level="ERROR")
                    return None
                if status in {"failed", "timeout", "canceled", "cancelled"}:
                    error = prediction.get("error") or prediction.get("message") or status
                    kiwi_log("ATLAS", f"TTS prediction failed: {error}", level="ERROR")
                    return None
            if attempt + 1 < self.config.max_polls:
                time.sleep(delay)
                delay = min(max(delay * 1.5, 0.5), 5.0)
        return None

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        style: str = "neutral",
        language: str = "Russian",
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Synthesize text through Atlas Cloud and return mono float32 WAV audio."""
        del kwargs
        if not text or not text.strip():
            return None, 0
        if not self.config.api_key:
            kiwi_log("ATLAS", "ATLASCLOUD_API_KEY is empty", level="ERROR")
            return None, 0

        voice_id = str(voice or self.config.voice_id).strip() or DEFAULT_ATLAS_VOICE_ID
        language_code = _LANGUAGE_CODES.get(
            language,
            language if language in _SUPPORTED_LANGUAGE_CODES else "auto",
        )
        cache_key = self._make_cache_key(text, voice_id, style, language_code, ATLAS_TTS_MODEL)
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        payload: Dict[str, Any] = {
            "model": ATLAS_TTS_MODEL,
            "text": self._styled_text(text.strip(), style),
            "language": language_code,
            "voice_id": voice_id,
            "codec": "wav",
            "sample_rate": 24000,
            "speed": 1.0,
            "text_normalization": True,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }

        # Submission is deliberately attempted once because it creates a paid task.
        try:
            response = requests.post(
                f"{ATLAS_API_BASE}/model/generateAudio",
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            prediction = self._response_data(response.json())
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            kiwi_log("ATLAS", f"TTS submit failed without retry: {exc}", level="ERROR")
            return None, 0

        status = str(prediction.get("status", "")).lower()
        outputs = prediction.get("outputs") or []
        audio_url = (
            outputs[0] if status in {"completed", "succeeded"} and outputs and isinstance(outputs[0], str) else None
        )
        if not audio_url:
            prediction_id = prediction.get("id")
            if not isinstance(prediction_id, str) or not prediction_id:
                kiwi_log("ATLAS", "TTS submission returned no prediction ID", level="ERROR")
                return None, 0
            audio_url = self._poll_prediction(prediction_id)
        if not audio_url:
            return None, 0

        try:
            audio_response = requests.get(
                audio_url,
                headers={"User-Agent": _USER_AGENT},
                timeout=self.config.timeout,
            )
            audio_response.raise_for_status()
            audio, sample_rate = self._decode_wav(audio_response.content)
        except (requests.RequestException, ValueError, wave.Error) as exc:
            kiwi_log("ATLAS", f"Audio download or decode failed: {exc}", level="ERROR")
            return None, 0

        audio = self._coerce_audio(audio)
        if use_cache:
            self._set_cached(cache_key, audio, sample_rate)
        return audio, sample_rate
