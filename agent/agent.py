from httpx._transports import base
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from dotenv import load_dotenv
import httpx
import json
import logging
import os
import time
from typing import Any, List
from pathlib import Path
from agent.tool import ToolManager, register_builtin_tools
from agent.prompts import assemble_exploration_subagent_system_prompt, assemble_system_prompt

PROJECT_ROOT = Path(__file__).parent.parent
LOG_PATH = PROJECT_ROOT / "log.txt"

load_dotenv(PROJECT_ROOT / ".env")

class TerminalLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "send_to_terminal", True)

logger = logging.getLogger("myvibecodingagent")
terminal_logger = logging.getLogger("myvibecodingagent.terminal")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.addFilter(TerminalLogFilter())
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    terminal_logger.setLevel(logging.INFO)
    terminal_logger.addHandler(stream_handler)

logger.propagate = False
terminal_logger.propagate = False



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
    role = "main_agent"
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
        register_builtin_tools(
            self.tool_manager,
            spawn_exploration_subagent_handler=spawn_exploration_subagent,
        )

    def _next_client_call_count(self) -> int:
        self.client_call_count += 1
        return self.client_call_count

    def _serialize_for_log(self, value: Any, max_length: int = 12000) -> str:
        try:
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            serialized = json.dumps(value, ensure_ascii=False, default=str, indent=2)
        except TypeError:
            serialized = str(value)

        if len(serialized) <= max_length:
            return serialized
        return f"{serialized[:max_length]}... (truncated {len(serialized) - max_length} chars)"

    def _log_model_request(self, api_name: str, call_count: int, payload: dict) -> None:
        logger.debug(
            "Model request | api=%s | attempt=%s | payload=%s",
            api_name,
            call_count,
            self._serialize_for_log(payload),
        )

    def _log_model_response(
        self, api_name: str, call_count: int, started_at: float, response: Any
    ) -> None:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(
            "Model response | api=%s | attempt=%s | elapsed_ms=%.2f | payload=%s",
            api_name,
            call_count,
            elapsed_ms,
            self._serialize_for_log(response),
        )

    def _log_tool_call(self, tool_name: str, tool_args: Any, result: str) -> None:
        logger.info(
            "Tool call | name=%s | input=%s | output=%s",
            tool_name,
            self._serialize_for_log(tool_args),
            self._serialize_for_log(result),
            extra={"send_to_terminal": False},
        )
        terminal_logger.info(">%s: %s", tool_name, result)

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
            result = f"Tool {tool_name} not found. Skipping function call."
            self._log_tool_call(tool_name, arguments, result)
            return result

        try:
            tool_args = self._parse_tool_arguments(arguments)
            result = self.tool_manager.execute(tool_name, **tool_args)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            tool_args = arguments
            result = f"Error parsing arguments for {tool_name}: {exc}"

        self._log_tool_call(tool_name, tool_args, result)
        return result

    def agent_loop(self, messages: List):
        return self._agent_loop(messages)

    def _agent_loop(self, messages: List, max_loop: int | None = None):
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
            for tool in self.tool_manager.list_all(role=self.role)
        ]

    def _agent_loop(self, messages: List, max_loop: int | None = None):
        # Compatible providers usually support chat.completions better than responses,
        # so this loop keeps the full chat history and appends tool results back to it.
        loop_count = 0
        while True:
            if max_loop is not None and loop_count >= max_loop:
                logger.warning("Max loop count reached before assistant produced a final summary.")
                return "Stopped after reaching max_loop before producing a final summary."
            loop_count += 1
            call_count = self._next_client_call_count()
            request_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    *messages,
                ],
                "tools": self._chat_tools(),
            }
            started_at = time.perf_counter()
            self._log_model_request("chat.completions", call_count, request_payload)
            try:
                response = self.client.chat.completions.create(
                    **request_payload,
                )
                self._log_model_response("chat.completions", call_count, started_at, response)
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
                return content

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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": current_result,
                    }
                )


class ResponsesOutputLoopAgent(LoopAgent):
    def _agent_loop(self, messages: List, max_loop: int | None = None):
        # OpenAI native responses API uses previous_response_id plus
        # function_call_output items to continue the tool-calling chain.
        response_input = messages
        previous_response_id = None
        loop_count = 0
        while True:
            if max_loop is not None and loop_count >= max_loop:
                logger.warning("Max loop count reached before assistant produced a final summary.")
                return "Stopped after reaching max_loop before producing a final summary."
            loop_count += 1
            call_count = self._next_client_call_count()
            request_payload = {
                "model": self.model,
                "input": response_input,
                "instructions": self.system_prompt,
                "tools": self.tool_manager.list_all(role=self.role),
                "previous_response_id": previous_response_id,
            }
            started_at = time.perf_counter()
            self._log_model_request("responses", call_count, request_payload)
            try:
                response = self.client.responses.create(
                    **request_payload,
                )
                self._log_model_response("responses", call_count, started_at, response)
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
                return response.output_text or ""

            tool_use_results = []
            for function_call in function_calls:
                current_result = self._execute_tool(function_call.name, function_call.arguments)
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

class ExplorationSubAgent(ChatCompletionLoopAgent):
    role = "sub_agent"

    def __init__(self, client: OpenAI, tool_manager: ToolManager):
        super().__init__(client, tool_manager)
        self.system_prompt = assemble_exploration_subagent_system_prompt()

    def invoke(self, messages: List, max_loop = 10) -> str:
        return self._agent_loop(messages, max_loop=max_loop)


def spawn_exploration_subagent(task: str, max_loop: int = 10) -> str:
    if not str(task).strip():
        return "Error: task is required"
    if max_loop < 1:
        return "Error: max_loop must be at least 1"

    try:
        tool_manager = ToolManager()
        subagent = ExplorationSubAgent(client, tool_manager)
        return subagent.invoke(
            messages=[{"role": "user", "content": task}],
            max_loop=max_loop,
        )
    except Exception as e:
        return f"Error: {e}"


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

