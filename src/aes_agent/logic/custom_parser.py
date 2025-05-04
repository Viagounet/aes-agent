from aes_agent.utils import ToolCallingResults, parse_function_call
from loguru import logger

def tool_to_docllm_format(tool: dict) -> str:
    TYPES_MAPPING = {"integer": "int", "number": "float", "array": "list"}

    args_strings: list[str] = []
    for arg_name, arg_properties in tool["input_schema"]["properties"].items():
        arg_type = "Unknown"
        if arg_properties["type"] in TYPES_MAPPING:
            arg_type = TYPES_MAPPING[arg_properties["type"]]
        args_strings.append(f"{arg_name}: {arg_type}")
    return f"{tool['name']}({', '.join(args_strings)}) => {tool['description']}"


async def custom_parser(session, llm, available_tools, task) -> ToolCallingResults:
    tools_strings: list[str] = []
    for tool in available_tools:
        tools_strings.append(tool_to_docllm_format(tool))
    system_prompt = f"<tools>{'\n'.join(tools_strings)}</tools>\n<answer template>\nReasoning: {{your_reasoning (string)}}\nAction: func(arg1=value1, ...)</answer template>\nUsing the tools at your disposal, complete the user's request by answering following exactly the template."
    user_prompt = task
    answer = llm.query(
        system_prompt=system_prompt, user_prompt=user_prompt
    )
    reasoning = (
        answer.split("Action:")[0].replace("Reasoning: ", "").strip()
    )
    action = answer.split("Action: ")[1].strip()
    parsed_function = parse_function_call(action)
    if not parsed_function:
        logger.debug(f"Couldn't parse action: {action}")
        return {
        "reasoning": "Tool error",
        "tool_called_name": "Tool error",
        "tool_called_arguments": [],
        "tool_called_result": None,
    }

    function_name = ""
    arguments = {}
    for available_tool in available_tools:
        if (
            "_" + available_tool["name"]
            == parsed_function["function_name"]
        ):
            function_name = available_tool["name"]
            argument_names = list(
                available_tool["input_schema"]["properties"].keys()
            )
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
    toolcall_result = await session.call_tool(
        function_name, arguments
    )
    logger.info(
        f"Results of '{function_name}': {toolcall_result.content[0].text}"
    )
    result: ToolCallingResults = {
        "reasoning": reasoning,
        "tool_called_name": function_name,
        "tool_called_arguments": arguments,
        "tool_called_result": toolcall_result.content[0].text,
    }
    return result
