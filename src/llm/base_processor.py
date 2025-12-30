"""
Base LLM Processor - Shared functionality for all LLM-based processors.

Provides common patterns for:
- Async client initialization
- LLM calls with error handling
- Concurrency control via semaphore
"""
import asyncio
import json
from typing import Any

from openai import AsyncOpenAI

from utils.logger import logger


class BaseLLMProcessor:
    """Base class for LLM processors with shared async functionality."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "deepseek-chat",
        max_concurrent: int = 100
    ):
        """
        Initialize the LLM processor.

        Args:
            api_key: OpenAI API key
            base_url: Custom API base URL (optional)
            model: Model name to use
            max_concurrent: Maximum concurrent requests
        """
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _call_llm_async(
        self,
        prompt: str,
        temperature: float = 0.3,
        response_format: dict[str, str] | None = None
    ) -> str:
        """
        Make an async LLM call with error handling.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            The response content as string
        """
        try:
            async with self.semaphore:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = await self.async_client.chat.completions.create(**kwargs)

                # Warn if response was truncated
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    logger.warning(
                        "Response truncated (finish_reason=length). "
                        "Model hit max output limit. Consider using a model with higher output capacity."
                    )
                    # Log the input prompt for analysis
                    prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
                    logger.warning(f"Input text preview:\n{prompt_preview}")
                    logger.warning(f"Input text length: {len(prompt)} characters")
                    # Save full input to debug file for analysis
                    self._save_truncation_debug(prompt, choice.message.content)

                return choice.message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    async def _call_llm_json_async(
        self,
        prompt: str,
        temperature: float = 0.1,
        error_context: str = "LLM call"
    ) -> dict[str, Any] | None:
        """
        Make an async LLM call expecting JSON response.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (default 0.1 for more deterministic output)
            error_context: Context string for error logging

        Returns:
            Parsed JSON dict, or None if parsing failed
        """
        try:
            async with self.semaphore:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": temperature,
                }

                response = await self.async_client.chat.completions.create(**kwargs)
                raw_content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # Warn if response was truncated
                if finish_reason == "length":
                    logger.warning(
                        f"{error_context}: Response truncated (finish_reason=length). "
                        f"Model hit max output limit. JSON may be incomplete."
                    )
                    # Log the input prompt for analysis (truncated to avoid huge logs)
                    prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
                    logger.warning(f"{error_context}: Input text preview:\n{prompt_preview}")
                    logger.warning(f"{error_context}: Input text length: {len(prompt)} characters")

                return json.loads(raw_content)
        except json.JSONDecodeError as je:
            logger.error(f"JSON parse error in {error_context}: {je}")
            logger.error(f"Raw response (first 300 chars): {raw_content[:300]}...")

            # Try to recover partial JSON from truncated response
            recovered = self._try_parse_partial_json(raw_content)
            if recovered:
                logger.info(f"Recovered partial data from truncated response: {len(recovered.get('nodes', []))} nodes, {len(recovered.get('edges', []))} edges")
                return recovered

            return None
        except Exception as e:
            logger.error(f"{error_context} failed: {e}")
            return None

    def _save_truncation_debug(self, prompt: str, response: str) -> None:
        """
        Save full prompt and response for debugging truncation issues.

        Creates a timestamped debug file with the input that caused truncation.
        """
        try:
            import time
            from pathlib import Path

            debug_dir = Path("output/truncation_debug")
            debug_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            debug_file = debug_dir / f"truncated_{timestamp}.txt"

            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("TRUNCATED RESPONSE DEBUG\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Input length: {len(prompt)} characters\n")
                f.write(f"Output length: {len(response)} characters\n\n")
                f.write("-" * 60 + "\n")
                f.write("INPUT PROMPT:\n")
                f.write("-" * 60 + "\n")
                f.write(prompt)
                f.write("\n\n")
                f.write("-" * 60 + "\n")
                f.write("OUTPUT RESPONSE:\n")
                f.write("-" * 60 + "\n")
                f.write(response)

            logger.info(f"Saved truncation debug info to: {debug_file}")
        except Exception as e:
            logger.debug(f"Could not save truncation debug: {e}")

    async def close(self):
        """
        Properly close the async client.

        Call this before the event loop closes to avoid "Event loop is closed" errors.
        """
        try:
            await self.async_client.close()
        except Exception as e:
            # Suppress errors during cleanup (event loop may already be closed)
            logger.debug(f"Client cleanup note: {e}")

    def __del__(self):
        """Destructor to ensure client is cleaned up."""
        # Note: __del__ can't be async, so we can't properly close here
        # Users should call await processor.close() explicitly if needed
        pass
