from aes_agent.utils import ToolCallingResults, parse_function_call, Turn, format_args
from loguru import logger


def tool_to_docllm_format(tool: dict) -> str:
    TYPES_MAPPING = {"integer": "int", "number": "float", "array": "list"}

    args_strings: list[str] = []
    for arg_name, arg_properties in tool["input_schema"]["properties"].items():
        arg_type = "Unknown"
        if "type" not in arg_properties:
            arg_type = "Any"
        else:
            if arg_properties["type"] in TYPES_MAPPING:
                arg_type = TYPES_MAPPING[arg_properties["type"]]
        args_strings.append(f"{arg_name}: {arg_type}")
    return f"{tool['name']}({', '.join(args_strings)}) => {tool['description']}"


async def custom_parser(
    session, environment, llm, available_tools, task, history: list[Turn]
) -> Turn:
    tools_strings: list[str] = []
    for tool in available_tools:
        tools_strings.append(tool_to_docllm_format(tool))
    system_prompt = f"{environment.state}<tools>{'\n'.join(tools_strings)}</tools>\n<answer template>\nReasoning: {{your_reasoning (string)}}\nAction: func(arg1=value1, ...)</answer template>\nUsing the tools at your disposal, complete the user's request by answering following exactly the template."
    user_prompt = task

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if history:
        for i, turn in enumerate(history):
            # Add argument: with or without reasoning
            for tool_call in turn["tools_called"]:
                tool_result_string = f"<Tool execution (turn {i + 1})>{tool_call['name']}({format_args(tool_call['arguments'])}) = {tool_call['result']}</Tool execution (turn {i + 1})>"
                messages.append({"role": "assistant", "content": tool_result_string})

    answer = llm.get_text(llm.query(messages))
    reasoning = answer.split("Action:")[0].replace("Reasoning: ", "").strip()
    action = answer.split("Action: ")[1].strip()
    parsed_function = parse_function_call(action)
    if not parsed_function:
        logger.debug(f"Couldn't parse action: {action}")
        return {
            "reasoning": "<Tool error>",
            "tools_called": [],
        }

    function_name = ""
    arguments = {}
    for available_tool in available_tools:
        if "_" + available_tool["name"] == parsed_function["function_name"]:
            function_name = available_tool["name"]
            argument_names = list(available_tool["input_schema"]["properties"].keys())
            for argument_name, positional_argument_value in zip(
                argument_names, parsed_function["positional_args"]
            ):
                arguments[argument_name] = positional_argument_value
            for argument_name, argument_value in parsed_function[
                "keyword_args"
            ].items():
                arguments[argument_name] = argument_value
    logger.info(
        f"Calling the function '{function_name}' with the following arguments: {arguments}"
    )
    toolcall_result = await session.call_tool(function_name, arguments)
    logger.info(f"Results of '{function_name}': {toolcall_result.content[0].text}")
    result: Turn = {
        "reasoning": reasoning,
        "tools_called": [{"name": function_name, "arguments": arguments, "result": toolcall_result.content[0].text, "id": None, "metadata": {}}]
    }
    return result
