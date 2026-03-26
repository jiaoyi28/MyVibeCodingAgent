from httpx._transports import base
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
agent_loop_backend = "chat" if not base_url else "response"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
    max_retries=max_retries,
)

class LoopAgent:
    """
    提供统一的对外入口：`agent_loop(messages)`。
    内部通过子类实现不同的调用形态：
    - `chat.completions`：完整保留 chat history，并把 tool 结果追加回 messages
    - `responses`：基于 previous_response_id 串联 function_call_output
    """

    def __init__(self, client: OpenAI, tool_manager: ToolManager):
        self.client = client
        self.tool_manager = tool_manager
        self.model = model
        self.system_prompt = assemble_system_prompt()
        self.client_call_count = 0
        self._register_tools()

    @classmethod
    def create(cls, client: OpenAI, tool_manager: ToolManager, agent_loop_backend="chat") -> "LoopAgent":
        # 三方兼容的 base_url 通常更容易兼容 chat.completions，而不一定兼容 responses。
        if agent_loop_backend == "chat":
            return ChatCompletionLoopAgent(client, tool_manager)
        return ResponsesOutputLoopAgent(client, tool_manager)

    def _register_tools(self):
        register_builtin_tools(self.tool_manager)

    def _next_client_call_count(self) -> int:
        self.client_call_count += 1
        return self.client_call_count

    def _parse_tool_arguments(self, arguments: Any) -> dict:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Tool arguments JSON must decode to an object")
        raise TypeError(f"Unsupported tool arguments type: {type(arguments).__name__}")

    def _execute_tool(self, tool_name: str, arguments: Any) -> str:
        if not self.tool_manager.has_tool(tool_name):
            return f"Tool {tool_name} not found. Skipping function call."

        try:
            tool_args = self._parse_tool_arguments(arguments)
            return self.tool_manager.execute(tool_name, **tool_args)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return f"Error parsing arguments for {tool_name}: {exc}"

    def agent_loop(self, messages: List):
        self._agent_loop(messages)

    def _agent_loop(self, messages: List):
        raise NotImplementedError


class ChatCompletionLoopAgent(LoopAgent):
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

    def _agent_loop(self, messages: List):
        # Compatible providers usually support chat.completions better than responses,
        # so this loop keeps the full chat history and appends tool results back to it.
        while True:
            try:
                call_count = self._next_client_call_count()
                logger.debug(f"Calling model service via chat.completions, attempt #{call_count}")
                response = self.client.chat.completions.create(
                    model=self.model,
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
                logger.error(f"OpenAI request failed with status {exc.status_code}: {exc}")
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
                current_result = self._execute_tool(tool_name, tool_call.function.arguments)
                logger.debug(f"Execute tool {tool_name}: {current_result}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": current_result,
                    }
                )


class ResponsesOutputLoopAgent(LoopAgent):
    def _agent_loop(self, messages: List):
        # OpenAI native responses API uses previous_response_id plus
        # function_call_output items to continue the tool-calling chain.
        response_input = messages
        previous_response_id = None
        while True:
            try:
                call_count = self._next_client_call_count()
                logger.debug(f"Calling model service via responses, attempt #{call_count}")
                response = self.client.responses.create(
                    model=self.model,
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
                logger.error(f"OpenAI request failed with status {exc.status_code}: {exc}")
                return

            logger.debug(response.output)
            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                logger.debug("No function calls found. Current agent loop is finished.")
                logger.debug(f"response.output_text: {response.output_text}")
                messages.append({"role": "assistant", "content": response.output_text or ""})
                return

            tool_use_results = []
            for function_call in function_calls:
                current_result = self._execute_tool(function_call.name, function_call.arguments)
                logger.debug(f"Execute tool {function_call.name}: {current_result}")
                tool_use_results.append(
                    {
                        "type": "function_call_output",
                        "call_id": function_call.id,
                        "output": current_result,
                    }
                )
            logger.debug(f"tool use results: {tool_use_results}")
            response_input = tool_use_results
            previous_response_id = response.id


if __name__ == "__main__":
    tool_manager = ToolManager()
    agent = LoopAgent.create(client, tool_manager)
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

