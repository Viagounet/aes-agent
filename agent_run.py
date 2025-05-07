import sys
import yaml

from argparse import ArgumentParser
from loguru import logger
from datetime import datetime

from aes_agent.environment import (
    OfflineSearchEnvironment,
    OnlineSearchEnvironment,
    Environment,
)
from aes_agent.llm import LLM, AnthropicLLM, OpenAILLM
from aes_agent.agent import Agent

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

log_filename = "logs/my_app_log_{time:YYYY-MM-DD-hh-mm-ss}.log"
logger.add(
    log_filename,
    level="INFO",  # Log messages of level INFO and above (WARNING, ERROR, CRITICAL)
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",  # Example format
)


def load_from_cgf(path: str) -> tuple[Environment, Agent]:
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    model_name = config["agent"]["llm"]["model"]
    match config["agent"]["llm"]["type"]:
        case "openai":
            llm = OpenAILLM(model=model_name)
        case "anthropic":
            llm = AnthropicLLM(model=model_name)
        case _:
            raise Exception(f"{config['agent']['llm']['type']} is not a supported LLM.")

    match config["environment"]["type"]:
        case "OfflineSearchEnvironment":
            env = OfflineSearchEnvironment(**config["environment"]["args"])
        case "OnlineSearchEnvironment":
            env = OnlineSearchEnvironment(**config["environment"]["args"])
        case _:
            raise Exception(
                f"{config['environment']['type']} is not a supported environment."
            )
    agent = Agent(llm=llm, mode=config["agent"]["output_mode"])
    return env, agent


env, agent = load_from_cgf(args.config)
agent.run(env, "Quelle est la différence entre le RLVR et le RLHF?")
