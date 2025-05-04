import os
import abc

from loguru import logger
from abc import ABC
from typing import Any, Optional

LLMResponse = Any

class LLM(ABC):
    def __init__(self, model: str):
        self._client: Any = None

    @abc.abstractmethod
    def get_text(self, response: Any) -> str:
        pass

    @abc.abstractmethod
    def query(self, system_prompt: str, user_prompt: str) -> str:
        pass

class OpenAILLM(ABC):
    def __init__(self, model: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def get_text(self, response: LLMResponse) -> str:
        return response.output_text

    def query(self, messages: list[dict], available_tools: list = []) -> LLMResponse:
        logger.info(f"Sent the following to {self.__class__.__name__}: {str(messages)}")
        response = self._client.responses.create(
            model=self.model, input=messages, tools=available_tools
        )
        logger.info(f"Received the following from {self.__class__.__name__}: {response.output_text}")
        return response

class AnthropicLLM(ABC):
    def __init__(self, model: str):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def get_text(self, response: LLMResponse) -> str:
        received_texts = []
        for content in response.content:
            if content.type == "text":
                logger.info(f"Content type is text: {content.text}")
                received_texts.append(content.text)
            elif content.type == "tool_use":
                logger.info(f"Content type is tool_use: {content.name} / {content.input}")
                tool_name = content.name
                tool_args = content.input
            else:
                raise Exception("Unknown content type for Anthropic answer")
        return "\n".join(received_texts)

    def query(self, messages: list[dict], available_tools: list = []) -> LLMResponse:
        system_prompt = None
        new_messages_list = []
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            else:
                new_messages_list.append(message)
        response = self._client.messages.create(
            model=self.model, messages=new_messages_list, tools=available_tools, max_tokens=2048, system=system_prompt
        )
        return response