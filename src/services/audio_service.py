"""
AudioService — Google Cloud Text-to-Speech integration.

Wraps the google-cloud-texttospeech API with the existing resilience patterns
(circuit breaker + retry + rate limiting) consistent with AIService.

Why Google TTS?
- Native GCP integration (same project credentials as Firestore/Storage)
- Supports Tamil (ta-IN), Hindi (hi-IN), English (en-US) and many other languages
- Cost-effective for children's story narration (pay-per-character)
- Returns audio bytes (MP3) directly, no streaming required for background processing
"""

import asyncio
import io
import struct
import wave
from google.cloud import texttospeech

from .database.firestore_service import FirestoreService  # for credential pattern reference
from ..utils.config import get_settings
from ..utils.logger import setup_logger
from ..utils.resilience import circuit_breaker, retry_with_backoff, CircuitBreakerError

logger = setup_logger(__name__)
settings = get_settings()


class AudioService:
    def __init__(self):
        self._client = None

    @property
    def client(self) -> texttospeech.TextToSpeechClient:
        """Lazy-initialize TTS client using Application Default Credentials."""
        if self._client is None:
            self._client = texttospeech.TextToSpeechClient()
        return self._client

    @circuit_breaker(
        name="google_tts",
        failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS,
    )
    @retry_with_backoff(
        max_retries=settings.MAX_RETRIES,
        base_delay=settings.RETRY_DELAY_SECONDS,
    )
    async def synthesize_speech(
        self,
        text: str,
        language_code: str = None,
        voice_name: str = None,
    ) -> bytes:
        """
        Synthesizes speech from text using Google Cloud TTS.

        Args:
            text: Story text to narrate.
            language_code: BCP-47 code (e.g. "en-US", "ta-IN"). Defaults to config.
            voice_name: TTS voice name (e.g. "en-US-Standard-A"). Defaults to config.

        Returns:
            MP3 audio bytes.

        Raises:
            CircuitBreakerError: If TTS circuit is open.
            Exception: On synthesis failure after retries.
        """
        lang = language_code or settings.TTS_LANGUAGE_CODE
        voice = voice_name or settings.TTS_VOICE_NAME

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang,
            name=voice,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding[settings.TTS_AUDIO_ENCODING],
        )

        logger.info(f"[AudioService] Synthesizing speech: lang={lang} voice={voice} chars={len(text)}")

        # TTS client is synchronous; run in executor to keep async loop free
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            ),
        )
        logger.info("[AudioService] Speech synthesis completed")
        return response.audio_content

    async def synthesize_with_fallback(
        self,
        text: str,
        language_code: str = None,
        voice_name: str = None,
    ) -> bytes | None:
        """
        Synthesizes speech with graceful fallback (returns None instead of raising).
        Used by AudioGeneratorAgent where a missing audio is recoverable via retry.
        """
        try:
            return await self.synthesize_speech(text, language_code, voice_name)
        except CircuitBreakerError:
            logger.error("[AudioService] TTS circuit breaker OPEN — audio generation unavailable")
            return None
        except Exception as e:
            logger.error(f"[AudioService] TTS synthesis failed: {e}")
            return None

    @staticmethod
    def _wav_duration(wav_bytes: bytes) -> float:
        """Returns the duration in seconds of a WAV byte string."""
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            return wf.getnframes() / wf.getframerate()

    @staticmethod
    def _combine_wav(wav_chunks: list[bytes]) -> bytes:
        """Concatenates multiple WAV byte strings into a single WAV."""
        if not wav_chunks:
            return b""
        # Read params from first chunk
        with wave.open(io.BytesIO(wav_chunks[0]), "rb") as first:
            params = first.getparams()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setparams(params)
            for chunk in wav_chunks:
                with wave.open(io.BytesIO(chunk), "rb") as wf:
                    out.writeframes(wf.readframes(wf.getnframes()))
        return buf.getvalue()

    async def synthesize_paragraphs(
        self,
        paragraphs: list[str],
        language_code: str = None,
        voice_name: str = None,
    ) -> tuple[bytes, list[dict]] | tuple[None, None]:
        """
        Synthesizes each paragraph separately, then combines into one WAV.

        Returns:
            (combined_wav_bytes, audio_timepoints) on success
            (None, None) on any failure

        audio_timepoints format:
            [{"ParagraphNumber": 1, "StartTimestamp": 0.0, "EndTimestamp": 4.58, "Duration": 4.58}, ...]
        """
        # Ensure WAV encoding for duration calculation
        lang = language_code or settings.TTS_LANGUAGE_CODE
        voice = voice_name or settings.TTS_VOICE_NAME

        try:
            wav_chunks: list[bytes] = []
            timepoints: list[dict] = []
            cursor = 0.0

            for idx, paragraph in enumerate(paragraphs, start=1):
                if not paragraph.strip():
                    continue

                synthesis_input = texttospeech.SynthesisInput(text=paragraph)
                voice_params = texttospeech.VoiceSelectionParams(
                    language_code=lang,
                    name=voice,
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,  # WAV
                )

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda ap=audio_config, vi=voice_params, si=synthesis_input: self.client.synthesize_speech(
                        input=si,
                        voice=vi,
                        audio_config=ap,
                    ),
                )

                chunk_bytes = response.audio_content
                duration = self._wav_duration(chunk_bytes)
                end = round(cursor + duration, 4)

                wav_chunks.append(chunk_bytes)
                timepoints.append({
                    "ParagraphNumber": idx,
                    "StartTimestamp": round(cursor, 4),
                    "EndTimestamp": end,
                    "Duration": round(duration, 4),
                })
                cursor = end
                logger.info(f"[AudioService] Paragraph {idx}: {duration:.4f}s")

            combined = self._combine_wav(wav_chunks)
            logger.info(f"[AudioService] Combined WAV: {len(combined)} bytes, {cursor:.4f}s total")
            return combined, timepoints

        except CircuitBreakerError:
            logger.error("[AudioService] TTS circuit breaker OPEN")
            return None, None
        except Exception as e:
            logger.error(f"[AudioService] synthesize_paragraphs failed: {e}")
            return None, None
