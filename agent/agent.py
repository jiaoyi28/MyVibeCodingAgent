from openai import APIConnectionError, APITimeoutError, OpenAI
from dotenv import load_dotenv
import httpx
import json
import logging
import os
from typing import Any, List
from pathlib import Path
from agent.tool import ToolManager, run_bash, run_edit, run_read, run_write
from agent.schema import Tool, ToolParameters, ToolProperty
from agent.prompts import assemble_system_prompt

PROJECT_ROOT = Path(__file__).parent.parent
LOG_PATH = PROJECT_ROOT / "log.txt"

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("myvibecodingagent")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.propagate = False



api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL") or None
model = os.getenv("MODEL_NAME")
request_timeout = 180.0
connect_timeout = 30.0
max_retries = 2

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
    max_retries=max_retries,
)

class MyAgent:
    def __init__(self, client: OpenAI, tool_manager: ToolManager):
        self.client = client
        self.tool_manager = tool_manager
        self._register_tools()
        self.system_prompt = assemble_system_prompt()
        self.messages = []

    def _register_tools(self):
        self.tool_manager.register(Tool(name="run_bash", description="Run a bash command", parameters=ToolParameters(type="object", properties={"command": ToolProperty(type="string", description="The command to run")}, required=["command"])), run_bash)
        self.tool_manager.register(Tool(name="run_edit", description="Edit a file", parameters=ToolParameters(type="object", properties={"path": ToolProperty(type="string", description="The path to the file"), "old_text": ToolProperty(type="string", description="The old text"), "new_text": ToolProperty(type="string", description="The new text")}, required=["path", "old_text", "new_text"])), run_edit)
        self.tool_manager.register(Tool(name="run_read", description="Read a file", parameters=ToolParameters(type="object", properties={"path": ToolProperty(type="string", description="The path to the file")}, required=["path"])), run_read)
        self.tool_manager.register(Tool(name="run_write", description="Write to a file", parameters=ToolParameters(type="object", properties={"path": ToolProperty(type="string", description="The path to the file"), "content": ToolProperty(type="string", description="The content to write")}, required=["path", "content"])), run_write)

    def _parse_tool_arguments(self, arguments: Any) -> dict:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Tool arguments JSON must decode to an object")
        raise TypeError(f"Unsupported tool arguments type: {type(arguments).__name__}")

    def agent_loop(self, messages: List):
        while True:
            try:
                response = self.client.responses.create(
                    model=model,
                    input=messages,
                    instructions=self.system_prompt,
                    tools=self.tool_manager.list_all(),
                )
            except APITimeoutError:
                logger.error(
                    "OpenAI request timed out. Check your network, proxy, and OPENAI_BASE_URL, "
                    "or increase OPENAI_TIMEOUT_SECONDS / OPENAI_CONNECT_TIMEOUT_SECONDS."
                )
                return
            except APIConnectionError as exc:
                logger.error(
                    f"OpenAI connection failed: {exc}. Check your network, proxy, and OPENAI_BASE_URL."
                )
                return
            logger.info(response.output)
            # messages.append({"role": "assistant", "content": response.output_text})
            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                logger.info("No function calls found. Current agent loop is finished.")
                logger.info(f"response.output_text: {response.output_text}")
                messages.append({"role": "assistant", "content": response.output_text})
                return
            
            tool_use_results = []
            for function_call in function_calls:
                tool_name = function_call.name
                if not self.tool_manager.has_tool(tool_name):
                    current_result = f"Tool {tool_name} not found. Skipping function call."
                else:
                    try:
                        tool_args = self._parse_tool_arguments(function_call.arguments)
                        current_result = self.tool_manager.execute(tool_name, **tool_args)
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        current_result = f"Error parsing arguments for {tool_name}: {exc}"
                logger.info(f"Execute tool {tool_name}: {current_result}")
                tool_use_results.append({"type": "function_call_output", "call_id": function_call.id, "output": current_result})
            logger.info(f"tool use results: {tool_use_results}")
            messages += tool_use_results



if __name__ == "__main__":
    tool_manager = ToolManager()
    agent = MyAgent(client, tool_manager)
    history_messages = []
    while True:
        try:
            query = input("User>> ")
        except (KeyboardInterrupt, EOFError):
            logger.info("Exiting...")
            break

        if query.lower() in ["exit", "quit", "bye", "q"]:
            logger.info("Exiting...")
            break

        history_messages.append({"role": "user", "content": query})
        agent.agent_loop(history_messages)
        response_content = history_messages[-1]["content"]
        if isinstance(response_content, list):
            for result in response_content:
                logger.info(result)
        else:
            logger.info(response_content)

