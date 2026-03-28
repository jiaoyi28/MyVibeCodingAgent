import subprocess
from typing import List

from agent.utils import WORKSPACE, safe_path
from agent.schema import Tool, ToolParameters, ToolProperty

class ToolManager:
    """tool管理"""

    def __init__(self) -> None:
        self.tools = {}
        self.tool_handlers = {}

    def register(self, tool: Tool, tool_handler):
        self.tools[tool.name] = tool
        self.tool_handlers[tool.name] = tool_handler


    def list_all(self) -> List:
        return [tool.to_dict() for tool in self.tools.values()]
    
    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tool_handlers
    
    def execute(self, tool_name: str, **kwargs):
        if tool_name not in self.tool_handlers:
            raise ValueError(f"Tool {tool_name} not found")
        return self.tool_handlers[tool_name](**kwargs)

class ToDoManager:
    def __init__(self, capacity: int = 10) -> None:
        self.todo_list = []
        self.capacity = capacity
    
    def update(self, todos: List) -> None:
        if len(todos) > self.capacity:
            raise ValueError("Exceed todo capacity")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(todos):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.todo_list = validated
        return self.render()

    def render(self) -> str:
        if not self.todo_list:
            return "No todos."
        lines = []
        for item in self.todo_list:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.todo_list if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.todo_list)} completed)")
        return "\n".join(lines)

to_do_manager = ToDoManager()


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKSPACE,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_SPECS = [
    {
        "name": "todo",
        "description": "Update todo task list. Track progress on multi-step tasks.",
        "properties": {
            "todos": ToolProperty(
                type="array",
                description="The todos to manage",
                items=ToolProperty(
                    type="object",
                    description="A todo item",
                    properties={
                        "id": ToolProperty(
                            type="string",
                            description="The todo item id",
                        ),
                        "text": ToolProperty(
                            type="string",
                            description="The todo item text",
                        ),
                        "status": ToolProperty(
                            type="string",
                            description="The todo item status",
                            enum=["pending", "in_progress", "completed"],
                        ),
                    },
                    required=["text"],
                ),
                max_items=10,
            )
        },
        "required": ["todos"],
        "handler": to_do_manager.update,
    },
    {
        "name": "run_bash",
        "description": "Run a bash command",
        "properties": {
            "command": ToolProperty(
                type="string",
                description="The command to run",
            )
        },
        "required": ["command"],
        "handler": run_bash,
    },
    {
        "name": "run_edit",
        "description": "Edit a file",
        "properties": {
            "path": ToolProperty(
                type="string",
                description="The path to the file",
            ),
            "old_text": ToolProperty(
                type="string",
                description="The old text",
            ),
            "new_text": ToolProperty(
                type="string",
                description="The new text",
            ),
        },
        "required": ["path", "old_text", "new_text"],
        "handler": run_edit,
    },
    {
        "name": "run_read",
        "description": "Read a file",
        "properties": {
            "path": ToolProperty(
                type="string",
                description="The path to the file",
            )
        },
        "required": ["path"],
        "handler": run_read,
    },
    {
        "name": "run_write",
        "description": "Write to a file",
        "properties": {
            "path": ToolProperty(
                type="string",
                description="The path to the file",
            ),
            "content": ToolProperty(
                type="string",
                description="The content to write",
            ),
        },
        "required": ["path", "content"],
        "handler": run_write,
    },
]


def register_builtin_tools(tool_manager: ToolManager):
    for spec in TOOL_SPECS:
        tool_manager.register(
            Tool(
                name=spec["name"],
                description=spec["description"],
                parameters=ToolParameters(
                    type="object",
                    properties=spec["properties"],
                    required=spec["required"],
                ),
            ),
            spec["handler"],
        )