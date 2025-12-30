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
        top_p: float | None = None,
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool = False
    ) -> str:
        """
        Make an async LLM call with error handling.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter (0.8 recommended for non-thinking mode)
            response_format: Optional response format (e.g., {"type": "json_object"})
            max_tokens: Maximum tokens to generate (prevents infinite loops)
            enable_thinking: Whether to enable Qwen3 thinking mode (default: False)

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
                if top_p is not None:
                    kwargs["top_p"] = top_p
                if response_format:
                    kwargs["response_format"] = response_format
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens

                # For vLLM with Qwen3, pass enable_thinking via extra_body
                extra_body = {}
                if not enable_thinking:
                    extra_body["enable_thinking"] = False
                if extra_body:
                    kwargs["extra_body"] = extra_body

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

                # Parse Qwen3 thinking blocks if present
                content = self._parse_qwen3_thinking(choice.message.content)
                return content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    async def _call_llm_json_async(
        self,
        prompt: str,
        temperature: float = 0.1,
        top_p: float | None = None,
        error_context: str = "LLM call",
        max_tokens: int | None = None,
        enable_thinking: bool = False
    ) -> dict[str, Any] | None:
        """
        Make an async LLM call expecting JSON response.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (default 0.1 for more deterministic output)
            top_p: Nucleus sampling parameter (0.8 recommended for non-thinking mode)
            error_context: Context string for error logging
            max_tokens: Maximum tokens to generate (prevents infinite loops)
            enable_thinking: Whether to enable Qwen3 thinking mode (default: False)

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
                if top_p is not None:
                    kwargs["top_p"] = top_p
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens

                # For vLLM with Qwen3, pass enable_thinking via extra_body
                extra_body = {}
                if not enable_thinking:
                    extra_body["enable_thinking"] = False
                if extra_body:
                    kwargs["extra_body"] = extra_body

                response = await self.async_client.chat.completions.create(**kwargs)
                raw_content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # Parse Qwen3 thinking blocks if present
                parsed_content = self._parse_qwen3_thinking(raw_content)

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

                return json.loads(parsed_content)
        except json.JSONDecodeError as je:
            logger.error(f"JSON parse error in {error_context}: {je}")
            logger.error(f"Raw response (first 300 chars): {raw_content[:300]}...")

            # For natural language mode, we can't recover from truncation
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

    def _parse_qwen3_thinking(self, content: str) -> str:
        """
        Parse Qwen3 thinking blocks and extract the final response.

        Qwen3 with thinking mode outputs content in this format:
        <think>
        ...thinking content...
        </think>
        final response

        This method extracts and returns only the final response.
        """
        # Handle <think>...</think> blocks
        think_start = content.find("<think>")
        if think_start == -1:
            # No thinking block, return content as-is
            return content

        think_end = content.find("</think>", think_start)
        if think_end == -1:
            # Malformed thinking block, log warning and return as-is
            logger.warning("Malformed Qwen3 thinking block: <think> without </think>")
            return content

        # Extract content after </think>
        final_content = content[think_end + len("</think>"):].strip()
        thinking_content = content[think_start + len("<think>"):think_end].strip()

        # Log thinking size for debugging
        if thinking_content:
            logger.debug(f"Qwen3 thinking block: {len(thinking_content)} chars, final: {len(final_content)} chars")

        return final_content if final_content else content

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
