from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from dotenv import load_dotenv
import httpx
import json
import logging
import os
from typing import Any, List
from pathlib import Path
from agent.tool import ToolManager, register_builtin_tools
from agent.prompts import assemble_system_prompt

PROJECT_ROOT = Path(__file__).parent.parent
LOG_PATH = PROJECT_ROOT / "log.txt"

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("myvibecodingagent")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
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
        self.client_call_count = 0

    def _register_tools(self):
        register_builtin_tools(self.tool_manager)

    def _parse_tool_arguments(self, arguments: Any) -> dict:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Tool arguments JSON must decode to an object")
        raise TypeError(f"Unsupported tool arguments type: {type(arguments).__name__}")

    def _chat_tools(self) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in self.tool_manager.list_all()
        ]

    def _next_client_call_count(self) -> int:
        self.client_call_count += 1
        return self.client_call_count

    def _chat_loop(self, messages: List):
        # Compatible providers usually support chat.completions better than responses,
        # so this loop keeps the full chat history and appends tool results back to it.
        while True:
            try:
                call_count = self._next_client_call_count()
                logger.debug(f"Calling model service via chat.completions, attempt #{call_count}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        *messages,
                    ],
                    tools=self._chat_tools(),
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
            except APIStatusError as exc:
                logger.error(
                    f"OpenAI request failed with status {exc.status_code}: {exc}"
                )
                return

            message = response.choices[0].message
            logger.debug(message)
            tool_calls = message.tool_calls or []
            if not tool_calls:
                content = message.content or ""
                logger.debug("No function calls found. Current agent loop is finished.")
                logger.debug(f"message.content: {content}")
                messages.append({"role": "assistant", "content": content})
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in tool_calls
                    ],
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                if not self.tool_manager.has_tool(tool_name):
                    current_result = f"Tool {tool_name} not found. Skipping function call."
                else:
                    try:
                        tool_args = self._parse_tool_arguments(tool_call.function.arguments)
                        current_result = self.tool_manager.execute(tool_name, **tool_args)
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        current_result = f"Error parsing arguments for {tool_name}: {exc}"

                logger.debug(f"Execute tool {tool_name}: {current_result}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": current_result,
                    }
                )

    def _responses_loop(self, messages: List):
        # OpenAI native responses API uses previous_response_id plus
        # function_call_output items to continue the tool-calling chain.
        response_input = messages
        previous_response_id = None
        while True:
            try:
                call_count = self._next_client_call_count()
                logger.debug(f"Calling model service via responses, attempt #{call_count}")
                response = self.client.responses.create(
                    model=model,
                    input=response_input,
                    instructions=self.system_prompt,
                    tools=self.tool_manager.list_all(),
                    previous_response_id=previous_response_id,
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
            except APIStatusError as exc:
                logger.error(
                    f"OpenAI request failed with status {exc.status_code}: {exc}"
                )
                return
            logger.debug(response.output)
            # messages.append({"role": "assistant", "content": response.output_text})
            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                logger.debug("No function calls found. Current agent loop is finished.")
                logger.debug(f"response.output_text: {response.output_text}")
                messages.append({"role": "assistant", "content": response.output_text or ""})
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
                logger.debug(f"Execute tool {tool_name}: {current_result}")
                tool_use_results.append({"type": "function_call_output", "call_id": function_call.id, "output": current_result})
            logger.debug(f"tool use results: {tool_use_results}")
            response_input = tool_use_results
            previous_response_id = response.id

    def agent_loop(self, messages: List):
        if base_url:
            # 三方兼容的base_url，通常不会兼容openai sdk最新的responses，但是会兼容原有的chat.completions
            # 这两个用法稍有不同
            self._chat_loop(messages)
            return
        self._responses_loop(messages)



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

