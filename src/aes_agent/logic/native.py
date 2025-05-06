import json

from aes_agent.utils import ToolCallingResults, parse_function_call
from aes_agent.llm import AnthropicLLM, OpenAILLM
from loguru import logger


async def native(
    session, environment, llm, available_tools, task, history: list[ToolCallingResults]
) -> ToolCallingResults:
    system_prompt = f"{environment.state}\nYour role is to complete the user's task by using tools that are provided to you. You will make sure to explain your reasoning before using a particular tool."
    user_prompt = task
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if isinstance(llm, OpenAILLM):
        tools_openai_format = []
        for tool in available_tools:
            tool_openai_format = {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "additionalProperties": False,
            }
            properties_openai_format = {}
            for argument_name, argument_properties in tool["input_schema"][
                "properties"
            ].items():
                properties_openai_format[argument_name] = {
                    "type": argument_properties["type"]
                    if "type" in argument_properties
                    else "string"
                }

            tool_openai_format["parameters"] = {
                "type": "object",
                "properties": properties_openai_format,
                "required": list(properties_openai_format.keys()),
                "required": tool["input_schema"]["required"],
            }
            tools_openai_format.append(tool_openai_format)

        response = llm.query(messages, available_tools=tools_openai_format)
        tool_call = response.output[0]
        tool_name = tool_call.name
        tool_args = json.loads(tool_call.arguments)
        toolcall_result = await session.call_tool(tool_name, tool_args)
        return {
                    "reasoning": "<no reasoning with OpenAI's native tool calling>",
                    "tool_called_name": tool_name,
                    "tool_called_arguments": tool_args,
                    "tool_called_result": toolcall_result.content[0].text,
                    "metadata": {}
                }

    elif isinstance(llm, AnthropicLLM):
        for tool_result in history:
            if "metadata" not in tool_result:
                raise Exception("No tool was called")
            if tool_result["metadata"]:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            tool_result["metadata"]["assistant_content"],
                            tool_result["metadata"]["tool_content"],
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_result["metadata"]["tool_use_id"],
                                "content": tool_result["tool_called_result"],
                            }
                        ],
                    }
                )
        response = llm.query(messages, available_tools=available_tools)
        logger.info(f"Response length: {len(response.content)}")
        reasoning = "<no reasoning>"
        for content in response.content:
            if content.type == "text":
                reasoning = content.text
                assistant_content = content
            if content.type == "tool_use":
                logger.info(
                    f"Content type is tool_use: {content.name} / {content.input}"
                )
                tool_name = content.name
                tool_args = content.input
                tool_content = content
                toolcall_result = await session.call_tool(tool_name, tool_args)
                logger.info(
                    f"Called tool {tool_name} with the following arguments: {tool_args} --> result is {toolcall_result}"
                )
                return {
                    "reasoning": assistant_content.text,
                    "tool_called_name": tool_name,
                    "tool_called_arguments": tool_args,
                    "tool_called_result": toolcall_result.content[0].text,
                    "metadata": {
                        "tool_use_id": content.id,
                        "assistant_content": assistant_content,
                        "tool_content": tool_content,
                    },
                }
    else:
        raise Exception(f"No 'native' tool calling for LLM of type {llm}")

    return {
        "reasoning": "<No reasoning with native tool use>",
        "tool_called_name": "No tool called",
        "tool_called_arguments": {},
        "tool_called_result": None,
    }
