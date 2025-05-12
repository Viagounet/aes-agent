import asyncio

from aes_agent.llm import LLM
from aes_agent.environment import Environment
from aes_agent.mcp.client import MCPClient
from aes_agent.logic.custom_parser import custom_parser
from aes_agent.logic.native import native
from aes_agent.utils import ToolCallingResults, Turn

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
        self.history: list[Turn] = []

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
                    result = await custom_parser(
                        self._mcp_client.session,
                        environment,
                        self.llm,
                        available_tools,
                        task,
                        self.history,
                    )
                case "native":
                    result = await native(
                        self._mcp_client.session,
                        environment,
                        self.llm,
                        available_tools,
                        task,
                        self.history,
                    )
                case _:
                    raise Exception(f"{self.mode} is not a correct mode.")

            self.history.append(result)
            for tool_call in result["tools_called"]:
                if tool_call["name"] == "final_answer":
                    logger.success(f"Final answer: {tool_call['result']}")
                    logger.info(f"Exiting {environment}")
                    await self._mcp_client.cleanup()
                    return self.history

        logger.info(f"Exiting {environment}")
        await self._mcp_client.cleanup()

    def run(self, environment: Environment, task: str):
        asyncio.run(self._run(environment, task))
