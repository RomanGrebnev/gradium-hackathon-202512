import os
from typing import AsyncIterator

from mistralai import Mistral
from mistralai.models import UserMessage, SystemMessage, AssistantMessage


class MistralClient:
    def __init__(self):
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        self.client = Mistral(api_key=api_key)

    async def chat_stream(self, messages: list[dict], model: str = "mistral-large-latest", response_format: dict = None) -> AsyncIterator[str]:
        """
        Streams chat completions from Mistral.
        
        Args:
            messages: List of dicts with {"role": str, "content": str}
            model: Model identifier
            response_format: Optional dict to specify output format (e.g. {"type": "json_object"})
            
        Yields:
            str: Content chunks
        """
        # Convert dict messages to Mistral SDK objects if needed, 
        # or rely on the SDK's ability to handle dicts (newer SDKs usually do).
        # We'll trust the SDK handles dicts for now to keep it simple, 
        # as seen in the previous vLLM wrapper usage.
        
        stream_response = await self.client.chat.stream_async(
            model=model,
            messages=messages,
            temperature=0.7,
            response_format=response_format,
        )

        async for chunk in stream_response:
            delta = chunk.data.choices[0].delta.content
            if delta:
                yield delta
