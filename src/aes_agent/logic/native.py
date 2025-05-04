from aes_agent.utils import ToolCallingResults, parse_function_call
from aes_agent.llm import AnthropicLLM, OpenAILLM
from loguru import logger


async def native(
    session, llm, available_tools, task, history: list[ToolCallingResults]
) -> ToolCallingResults:
    system_prompt = "Your role is to complete the user's task by using tools that are provided to you. You will make sure to explain your reasoning before using a particular tool."
    user_prompt = task
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if isinstance(llm, OpenAILLM):
        pass  # To implement
    elif isinstance(llm, AnthropicLLM):
        response = llm.query(messages, available_tools=available_tools)
        logger.info(f"Response length: {len(response.content)}")
        reasoning = "<no reasoning>"
        for content in response.content:
            if content.type == "text":
                reasoning = content.text
            if content.type == "tool_use":
                logger.info(
                    f"Content type is tool_use: {content.name} / {content.input}"
                )
                tool_name = content.name
                tool_args = content.input
                result = await session.call_tool(tool_name, tool_args)
                return {
                    "reasoning": reasoning,
                    "tool_called_name": tool_name,
                    "tool_called_arguments": tool_args,
                    "tool_called_result": result,
                }
    else:
        raise Exception(f"No 'native' tool calling for LLM of type {llm}")

    return {
        "reasoning": "<No reasoning with native tool use>",
        "tool_called_name": "No tool called",
        "tool_called_arguments": {},
        "tool_called_result": None,
    }
