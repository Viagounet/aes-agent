import os
import abc

from loguru import logger
from abc import ABC
from typing import Any

class LLM(ABC):
    def __init__(self, model: str):
        self._client: Any = None

    @abc.abstractmethod
    def query(self, system_prompt: str, user_prompt: str) -> str:
        pass

class OpenAILLM(ABC):
    def __init__(self, model: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def query(self, system_prompt: str, user_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        logger.info(f"Sent the following to {self.__class__.__name__}: {str(messages)}")
        response = self._client.responses.create(
            model=self.model, input=messages
        )
        logger.info(f"Received the following from {self.__class__.__name__}: {response.output_text}")
        return response.output_text

class AnthropicLLM(ABC):
    def __init__(self, model: str):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def query(self, system_prompt: str, user_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self._client.responses.create(
            model=self.model, input=messages
        )
        return response.output_text