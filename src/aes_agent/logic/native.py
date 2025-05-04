from aes_agent.utils import ToolCallingResults, parse_function_call
from aes.llm import AnthropicLLM, OpenAILLM
from loguru import logger


async def native(
    session, llm, available_tools, task, history: list[ToolCallingResults]
) -> ToolCallingResults:
    
    if isinstance(llm, OpenAILLM):
        pass # To implement
    elif isinstance(llm, AnthropicLLM):
        pass # To implement
    else:
        raise Exception(f"No 'native' tool calling for LLM of type {llm}")

    return {
            "reasoning": "<No reasoning with native tool use>",
            "tool_called_name": "Tool error",
            "tool_called_arguments": {},
            "tool_called_result": None,
        }