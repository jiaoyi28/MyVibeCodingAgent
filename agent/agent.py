from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Any, List
from pathlib import Path
from agent.tool import ToolManager, run_bash, run_edit, run_read, run_write
from agent.schema import Tool, ToolParameters, ToolProperty, Message
from agent.prompts import assemble_system_prompt

load_dotenv(Path(__file__).parent / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
model = os.getenv("MODEL")

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

    def agent_loop(self, messages: List[Message]):
        while True:
            response = self.client.responses.create(model=model, input=messages, instructions=self.system_prompt, tools=self.tool_manager.list_all())
            print(response.output_text)
            messages.append(Message(role="assistant", content=response.output_text))
            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                print("No function calls found. Current agent loop is finished.")
            
            tool_use_results = []
            for function_call in function_calls:
                tool_name = function_call.name
                if not self.tool_manager.has_tool(tool_name):
                    current_result = f"Tool {tool_name} not found. Skipping function call."
                else:
                    current_result = self.tool_manager.execute(tool_name, **function_call.arguments)
                print(f"Execute tool {tool_name}: {current_result}")
                tool_use_results.append(current_result)
            messages.append(Message(role="user", content=tool_use_results))



if __name__ == "__main__":
    tool_manager = ToolManager()
    agent = MyAgent(client, tool_manager)
    history_messages = []
    while True:
        try:
            query = input("User>> ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if query.lower() in ["exit", "quit", "bye", "q"]:
            print("Exiting...")
            break

        history_messages.append(Message(role="user", content=query))
        agent.agent_loop(history_messages)
        response_content = history_messages[-1][content]
        if isinstance(response_content, list):
            for result in response_content:
                print(result)
        else:
            print(response_content)



