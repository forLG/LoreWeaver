"""
Base LLM Processor - Shared functionality for all LLM-based processors.

Provides common patterns for:
- Async client initialization
- LLM calls with error handling (natural language mode)
- Concurrency control via semaphore
"""
import asyncio

# Use project root logger
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.logger import logger


class BaseLLMProcessor:
    """Base class for LLM processors with shared async functionality."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "deepseek-chat",
        max_concurrent: int = 100,
        repetition_penalty: float | None = None
    ):
        """
        Initialize the LLM processor.

        Args:
            api_key: OpenAI API key
            base_url: Custom API base URL (optional)
            model: Model name to use
            max_concurrent: Maximum concurrent requests
            repetition_penalty: Repetition penalty for vLLM (1.0 = no penalty, 1.1-1.5 recommended)
        """
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.repetition_penalty = repetition_penalty

    async def _call_llm_async(
        self,
        prompt: str,
        temperature: float = 0.3,
        top_p: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool = False,
        stop: list[str] | None = None
    ) -> str:
        """
        Make an async LLM call with error handling.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            enable_thinking: Whether to enable Qwen3 thinking mode (default: False)
            stop: Stop sequences

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
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if stop:
                    kwargs["stop"] = stop

                # For vLLM with Qwen3, pass enable_thinking and repetition_penalty via extra_body
                extra_body = {}
                if not enable_thinking:
                    extra_body["enable_thinking"] = False
                if self.repetition_penalty is not None:
                    extra_body["repetition_penalty"] = self.repetition_penalty
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
                # Parse Qwen3 thinking blocks if present
                content = self._parse_qwen3_thinking(choice.message.content)
                return content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

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

