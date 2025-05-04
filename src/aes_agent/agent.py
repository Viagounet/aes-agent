import asyncio

from aes_agent.llm import LLM
from aes_agent.environment import Environment
from aes_agent.mcp.client import MCPClient
from aes_agent.logic.custom_parser import custom_parser
from aes_agent.utils import ToolCallingResults

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


TOOL_FORMATING_MAPPING = {"custom-parser": tool_to_docllm_format}


class Agent:
    def __init__(self, llm: LLM, mode="doc_llm"):
        self.llm = llm
        self._mcp_client = MCPClient()
        self.mode = mode
        self.history: list[ToolCallingResults] = []
    @property
    def _tool_formating_function(self):
        if self.mode not in TOOL_FORMATING_MAPPING:
            raise Exception(f"{self.mode} not in available tool formats")
        return TOOL_FORMATING_MAPPING[self.mode]

    async def _run(self, environment: Environment, task: str):
        logger.info(
            f"Setting up environment's MCP server: {environment._mcp_server_script}"
        )
        await self._mcp_client.connect_to_server(environment._mcp_server_script)
        logger.info(f"Running agent in environment {environment}")
        while environment.is_running:
            environment.turn += 1
            logger.info(f"Entering turn {environment.turn}")
            response = await self._mcp_client.session.list_tools()
            available_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in response.tools
            ]

            match self.mode:
                case "custom-parser":
                    result = await custom_parser(self._mcp_client.session, self.llm, available_tools, task, self.history)
                    print(result)
                    input("=====")
                case "native":
                    system_prompt = "Your role is to complete the user's task by using tools that are provided to you. You will make sur e to explain your reasoning before using a particular tool."
                    user_prompt = task
                    messages = [
                        {"role": "assistant", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    response = self.llm._client.messages.create(
                        model=self.llm.model,
                        max_tokens=1000,
                        messages=messages,
                        tools=available_tools,
                    )

                    final_text = []
                    assistant_message_content = []
                    for content in response.content:
                        if content.type == "text":
                            logger.info(f"Content type is text: {content.text}")
                            final_text.append(content.text)
                            assistant_message_content.append(content)
                        elif content.type == "tool_use":
                            logger.info(f"Content type is tool_use: {content.name} / {content.input}")
                            tool_name = content.name
                            tool_args = content.input

                            # Execute tool call
                            result = await self._mcp_client.session.call_tool(tool_name, tool_args)
                            final_text.append(
                                f"[Calling tool {tool_name} with args {tool_args}]"
                            )

                            assistant_message_content.append(content)
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": assistant_message_content,
                                }
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": content.id,
                                            "content": result.content,
                                        }
                                    ],
                                }
                            )

                            # Get next response from Claude
                            response = self.llm._client.messages.create(
                                model="claude-3-5-sonnet-20241022",
                                max_tokens=1000,
                                messages=messages,
                                tools=available_tools,
                            )
                            logger.info(f"Getting new answer: {response.content[0].text}")
                            final_text.append(response.content[0].text)
                    final_answer_string = "\n".join(final_text)
                    logger.info(f"Final answer: {final_answer_string}")

                case _:
                    raise Exception(f"{self.mode} is not a correct mode.")
            self.history.append(result)
        logger.info(f"Exiting {environment}")
        await self._mcp_client.cleanup()

    def run(self, environment: Environment, task: str):
        asyncio.run(self._run(environment, task))
