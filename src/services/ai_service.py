from ..utils.config import get_settings
from ..utils.logger import setup_logger
from ..utils.resilience import (
    circuit_breaker,
    retry_with_backoff,
    CircuitBreakerError,
    RateLimiter,
)
from google import genai
from google.genai import types
import hashlib
from functools import lru_cache
import io
from huggingface_hub import InferenceClient


# settings = get_settings()
# logger = setup_logger(__name__)

settings = get_settings()
logger = setup_logger(__name__)

class AIService:
    def __init__(self):
        # Initialize the new Client from google-genai
        self._client = None
        # Ensure we use a model that supports image generation if requested
        # e.g., "gemini-2.0-flash-exp" or "gemini-2.5-flash-image"
        self.model_name = settings.GEMINI_MODEL 
        self.fallback_model_name = settings.GEMINI_FALLBACK_MODEL
        self.multimodal_model_name = settings.MULTIMODAL_MODEL
        self.rate_limiter = RateLimiter(
            rate=settings.RATE_LIMIT_TOKENS_PER_SECOND,
            capacity=settings.RATE_LIMIT_BURST_CAPACITY,
        )

    def _build_generate_content_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4000,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_LOW_AND_ABOVE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_LOW_AND_ABOVE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_LOW_AND_ABOVE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_LOW_AND_ABOVE",
                ),
            ],
            response_mime_type="application/json",
        )

    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        return self._client

    @lru_cache(maxsize=100)
    def _generate_cached(self, prompt_hash: str, prompt: str, model_name: str):
        """
        Internal method to cache AI text responses.
        """
        logger.info(f"Generating new content for hash: {prompt_hash[:8]} using {model_name}...")
        
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=self._build_generate_content_config(),
        )
        return response.text

    @circuit_breaker(
        name="gemini_primary",
        failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS,
    )
    @retry_with_backoff(
        max_retries=settings.MAX_RETRIES,
        base_delay=settings.RETRY_DELAY_SECONDS,
    )
    async def _generate_with_primary(self, prompt: str) -> str:
        await self.rate_limiter.acquire()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return self._generate_cached(prompt_hash, prompt, self.model_name)

    @circuit_breaker(
        name="gemini_fallback",
        failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS,
    )
    @retry_with_backoff(
        max_retries=settings.MAX_RETRIES,
        base_delay=settings.RETRY_DELAY_SECONDS,
    )
    async def _generate_with_fallback(self, prompt: str) -> str:
        await self.rate_limiter.acquire()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return self._generate_cached(prompt_hash, prompt, self.fallback_model_name)

    async def generate_content(self, prompt: str, model_override: str = None, fallback_override: str = None, use_cache: bool = True) -> str:
        """
        Public method to generate text content with primary model and fallback model.

        Args:
            prompt: The prompt to send to the model.
            model_override: If provided, use this model instead of self.model_name.
                            Used by per-workflow agents (e.g. STORY_CREATOR_MODEL).
                            Existing callers that pass no override continue to use
                            the default gemini-2.0-flash-lite.
            fallback_override: If provided, use this as the fallback model.
                               Defaults to self.fallback_model_name if not set.
        """
        # Default path: use decorated _generate_with_primary/_generate_with_fallback
        # which have @circuit_breaker + @retry_with_backoff protection.
        if model_override is None and fallback_override is None:
            try:
                return await self._generate_with_primary(prompt)
            except CircuitBreakerError as primary_cb_error:
                logger.warning(f"Primary model circuit breaker open: {primary_cb_error}")
            except Exception as primary_error:
                logger.warning(f"Primary model failed, trying fallback: {primary_error}")

            if self.fallback_model_name == self.model_name:
                logger.error("Fallback model is same as primary model and primary failed")
                raise RuntimeError("No distinct fallback model available")

            return await self._generate_with_fallback(prompt)

        # Override path: per-workflow agents explicitly select a model.
        primary = model_override or self.model_name
        fallback = fallback_override or self.fallback_model_name

        def _call(model: str) -> str:
            if use_cache:
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
                return self._generate_cached(prompt_hash, prompt, model)
            # Bypass lru_cache — call the underlying API directly
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=self._build_generate_content_config(),
            )
            return response.text

        try:
            await self.rate_limiter.acquire()
            return _call(primary)
        except CircuitBreakerError as primary_cb_error:
            logger.warning(f"Primary model ({primary}) unavailable due to circuit breaker: {primary_cb_error}")
        except Exception as primary_error:
            logger.warning(f"Primary model ({primary}) failed, trying fallback: {primary_error}")

        if fallback == primary:
            logger.error("Fallback model is same as primary model and primary failed")
            raise RuntimeError("No distinct fallback model available")

        try:
            await self.rate_limiter.acquire()
            return _call(fallback)
        except CircuitBreakerError as fallback_cb_error:
            logger.error(f"Fallback model ({fallback}) unavailable due to circuit breaker: {fallback_cb_error}")
            raise
        except Exception as fallback_error:
            logger.error(f"Fallback model ({fallback}) failed: {fallback_error}")
            raise

    async def generate_multimodal_content(self, prompt: str) -> dict:
        """
        Generates both TEXT and IMAGES from a single prompt using the new SDK.
        Returns a dict with 'text' and 'images' (list of dictionaries with 'mime_type' and 'data').
        """
        logger.info(f"Generating multimodal content for: {prompt[:30]}...")
        
        try:
            await self.rate_limiter.acquire()
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1000, 
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_LOW_AND_ABOVE",  # Block few
                    ),
                ],
                response_mime_type="application/json",
                response_modalities=["IMAGE", "TEXT"],
            )

            text_parts = []
            images = []
            
            # Using streaming to handle mixed content
            # Note: This is synchronous in the SDK currently, but wrapped in async method
            for chunk in self.client.models.generate_content_stream(
                model=self.multimodal_model_name,
                contents=contents,
                config=generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue

                part = chunk.candidates[0].content.parts[0]
                
                # Handle Image
                if part.inline_data and part.inline_data.data:
                    images.append({
                        "mime_type": part.inline_data.mime_type,
                        "data": part.inline_data.data # Raw binary data
                    })
                
                # Handle Text
                if part.text:
                    text_parts.append(part.text)

            return {
                "text": "".join(text_parts).strip(),
                "images": images
            }

        except Exception as e:
            logger.error(f"Multimodal Generation failed: {str(e)}")
            raise e

    @circuit_breaker(name="flux_image", failure_threshold=3, recovery_timeout=120)
    @retry_with_backoff(max_retries=2, base_delay=2.0)
    async def generate_image(self, prompt: str, fallback_on_failure: bool = True):
        """
        Generates an image from a prompt using the Together API.
        Wrapped with circuit breaker and retry with exponential backoff.
        
        Args:
            prompt: Image generation prompt
            fallback_on_failure: If True, return None instead of raising on failure
        """
        try:
            logger.info(f"Generating image for: {prompt[:30]}...")
            await self.rate_limiter.acquire()
            client = InferenceClient(api_key=settings.HF_TOKEN)

            # output is a PIL.Image object
            # Model from config: defaults to FLUX.1-schnell (10x cheaper, same quality for children's art)
            image = client.text_to_image(
                prompt,
                model=settings.FLUX_IMAGE_MODEL,
            )
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')   
            img_byte_arr = img_byte_arr.getvalue()
            return img_byte_arr 
        except CircuitBreakerError:
            logger.error("FLUX circuit breaker is OPEN - image generation unavailable")
            if fallback_on_failure:
                logger.warning("Returning None as fallback for image generation")
                return None
            raise
        except Exception as e:
            logger.error(f"Image Generation failed: {str(e)}")
            if fallback_on_failure:
                logger.warning("Returning None as fallback for image generation")
                return None
            raise e
