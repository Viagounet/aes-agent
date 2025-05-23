import json

from aes_agent.utils import ToolCallingResults, Turn, parse_function_call, format_args
from aes_agent.llm import AnthropicLLM, OpenAILLM
from loguru import logger


async def native(
    session, environment, llm, available_tools, task, history: list[Turn]
) -> Turn:
    system_prompt = f"{environment.state}\nYour role is to complete the user's task by using tools that are provided to you. You will make sure to explain your reasoning before using a particular tool."
    user_prompt = task
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if isinstance(llm, OpenAILLM):
        if history:
            for i, turn in enumerate(history):
                # Add argument: with or without reasoning
                for tool_call in turn["tools_called"]:
                    tool_result_string = f"<Tool execution (turn {i + 1})>{tool_call['name']}({format_args(tool_call['arguments'])}) = {tool_call['result']}</Tool execution (turn {i + 1})>"
                    messages.append({"role": "assistant", "content": tool_result_string})

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
        tools_called: list[ToolCallingResults] = []
        reasoning = "<no reasoning>"
        for output in response.output:
            if output.type == "message":
                reasoning = output.content
            elif output.type == "function_call":
                arguments = json.loads(output.arguments)
                logger.info(
                    f"Calling tool {output.name} with the following arguments: {arguments}"
                )
                toolcall_result = await session.call_tool(output.name, arguments)
                logger.info(f"Result: {toolcall_result}")
                tools_called.append({"name": output.name, "arguments": arguments, "result": toolcall_result.content[0].text, "id": output.id, "metadata": {}})

        return {
            "reasoning": reasoning,
            "tools_called": tools_called
        }

    elif isinstance(llm, AnthropicLLM):
        for tool_result in history:
            for tool_call in tool_result["tools_called"]:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                tool_call["metadata"]["assistant_full_content"],
                                tool_call["metadata"]["tool_full_content"],
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_call["id"],
                                    "content": tool_call["result"],
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
                tool_name = content.name
                tool_args = content.input
                tool_content = content
                logger.info(
                    f"Calling tool {tool_name} with the following arguments: {tool_args}"
                )
                toolcall_result = await session.call_tool(tool_name, tool_args)
                if toolcall_result.isError:
                    logger.error(f"Error: {toolcall_result.content[0].text}")
                else:
                    logger.success(f"Result: {toolcall_result.content[0].text}")

                return {
                    "reasoning": reasoning,
                    "tools_called": [
                        {
                            "name": tool_name,
                            "arguments": tool_args,
                            "result": toolcall_result.content[0].text,
                            "id": content.id,
                            "metadata": {
                                "assistant_full_content": assistant_content,
                                "tool_full_content": tool_content,
                            },
                        }
                    ],
                }
    else:
        raise Exception(f"No 'native' tool calling for LLM of type {llm}")

    return {"reasoning": "<No reasoning with native tool use>", "tools_called": []}
