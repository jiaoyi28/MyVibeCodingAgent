import subprocess
from enum import Enum
from typing import List

from agent.utils import WORKSPACE, safe_path
from agent.schema import Tool

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