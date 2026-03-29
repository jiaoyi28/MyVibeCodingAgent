import fnmatch
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping
from logging import getLogger

from agent.utils import WORKSPACE, safe_path
from agent.schema import JsonSchema, Tool, ToolParameters
from agent.skill import SkillManager

logger = getLogger("myvibecodingagent")

ROLE_VISIBLE_TOOL_NAMES = {
    "sub_agent": {"glob", "grep", "read_file"},
}

class ToolManager:
    _instance = None

    """tool管理"""

    def __init__(self) -> None:
        self.tools = {}
        self.tool_handlers = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, tool: Tool, tool_handler):
        self.tools[tool.name] = tool
        self.tool_handlers[tool.name] = tool_handler


    def _is_tool_visible(self, tool_name: str, role: str | None = None) -> bool:
        if not role or role == "main_agent":
            return True
        visible_tool_names = ROLE_VISIBLE_TOOL_NAMES.get(role)
        if visible_tool_names is None:
            return True
        return tool_name in visible_tool_names

    def list_all(self, role: str | None = None) -> List:
        return [
            tool.to_dict()
            for tool in self.tools.values()
            if self._is_tool_visible(tool.name, role)
        ]
    
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
    
    def update(self, items: List) -> None:
        if len(items) > self.capacity:
            raise ValueError("Exceed todo capacity")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
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

TO_DO_MANAGER = ToDoManager()


ToolHandler = Callable[..., str]

SCHEMA_KEY_MAP = {
    "minItems": "min_items",
    "maxItems": "max_items",
    "minLength": "min_length",
    "maxLength": "max_length",
    "additionalProperties": "additional_properties",
    "oneOf": "one_of",
    "anyOf": "any_of",
    "allOf": "all_of",
}


def schema_from_dict(data: Mapping[str, Any], *, root: bool = False) -> JsonSchema:
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        normalized_key = SCHEMA_KEY_MAP.get(key, key)
        if key == "properties":
            kwargs["properties"] = {
                name: schema_from_dict(schema_data) for name, schema_data in value.items()
            }
        elif key == "items":
            if isinstance(value, list):
                kwargs["items"] = [schema_from_dict(item) for item in value]
            else:
                kwargs["items"] = schema_from_dict(value)
        elif key in {"oneOf", "anyOf", "allOf"}:
            kwargs[normalized_key] = [schema_from_dict(item) for item in value]
        elif key == "additionalProperties" and isinstance(value, Mapping):
            kwargs["additional_properties"] = schema_from_dict(value)
        else:
            kwargs[normalized_key] = value
    schema_cls = ToolParameters if root else JsonSchema
    return schema_cls(**kwargs)


@dataclass(frozen=True)
class BuiltinToolSpec:
    name: str
    description: str
    input_schema: JsonSchema | Mapping[str, Any]
    handler: ToolHandler

    def schema(self) -> JsonSchema:
        if isinstance(self.input_schema, JsonSchema):
            return self.input_schema
        return schema_from_dict(self.input_schema, root=True)

    def build_tool(self, *, name: str | None = None) -> Tool:
        return Tool(
            name=name or self.name,
            description=self.description,
            parameters=self.schema(),
        )


def bash(command: str) -> str:
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


def read_file(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _workspace_relative_path(path) -> str:
    return str(path.relative_to(WORKSPACE)).replace("\\", "/")


def glob(pattern: str, base_path: str = ".") -> str:
    try:
        if not str(pattern).strip():
            return "Error: pattern is required"

        root = safe_path(base_path)
        matches = sorted(
            [
                _workspace_relative_path(path)
                for path in root.rglob("*")
                if fnmatch.fnmatch(path.name, pattern)
                or fnmatch.fnmatch(str(path.relative_to(root)).replace("\\", "/"), pattern)
            ]
        )
        if not matches:
            return "No matches found."
        if len(matches) > 200:
            matches = matches[:200] + [f"... ({len(matches) - 200} more matches)"]
        return "\n".join(matches)
    except Exception as e:
        return f"Error: {e}"


def grep(
    pattern: str,
    base_path: str = ".",
    file_pattern: str = "*",
    case_sensitive: bool = False,
) -> str:
    try:
        if not str(pattern).strip():
            return "Error: pattern is required"

        root = safe_path(base_path)
        regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        results = []

        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative_to_root = str(path.relative_to(root)).replace("\\", "/")
            if not fnmatch.fnmatch(path.name, file_pattern) and not fnmatch.fnmatch(
                relative_to_root, file_pattern
            ):
                continue

            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                lines = path.read_text(errors="ignore").splitlines()

            for line_no, line in enumerate(lines, start=1):
                if regex.search(line):
                    relative_path = _workspace_relative_path(path)
                    results.append(f"{relative_path}:{line_no}:{line}")
                    if len(results) >= 200:
                        results.append("... (more matches)")
                        return "\n".join(results)[:50000]

        return "\n".join(results)[:50000] if results else "No matches found."
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def spawn_exploration_subagent(task: str, max_loop: int = 10) -> str:
    if not str(task).strip():
        return "Error: task is required"
    if max_loop < 1:
        return "Error: max_loop must be at least 1"

    try:
        from agent.agent import ExplorationSubAgent, client

        tool_manager = ToolManager()
        subagent = ExplorationSubAgent(client, tool_manager)
        return subagent.invoke(
            messages=[{"role": "user", "content": task}],
            max_loop=max_loop,
        )
    except Exception as e:
        return f"Error: {e}"


def load_skill(name: str) -> str:
    if not str(name).strip():
        err = "load_skill: name is required"
        logger.error(err)
        return err
    return SkillManager().get_content(name)


TOOL_SPECS = [
    BuiltinToolSpec(
        name="bash",
        description="Run a shell command.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to run.",
                }
            },
            "required": ["command"],
        },
        handler=bash,
    ),
    BuiltinToolSpec(
        name="read_file",
        description="Read file contents.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the file to read.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional max number of lines to read.",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        },
        handler=read_file,
    ),
    BuiltinToolSpec(
        name="glob",
        description="Find files by glob pattern.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files, such as '*.py' or 'agent/*.py'.",
                },
                "base_path": {
                    "type": "string",
                    "description": "Optional base directory to search from.",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
        handler=glob,
    ),
    BuiltinToolSpec(
        name="grep",
        description="Search file contents with a regex pattern.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "base_path": {
                    "type": "string",
                    "description": "Optional base directory to search from.",
                    "default": ".",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob for files to include, such as '*.py'.",
                    "default": "*",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the regex search is case-sensitive.",
                    "default": False,
                },
            },
            "required": ["pattern"],
        },
        handler=grep,
    ),
    BuiltinToolSpec(
        name="write_file",
        description="Write content to file.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write.",
                },
            },
            "required": ["path", "content"],
        },
        handler=write_file,
    ),
    BuiltinToolSpec(
        name="edit_file",
        description="Replace exact text in file.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the file to edit.",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to replace.",
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
        handler=edit_file,
    ),
    BuiltinToolSpec(
        name="todo",
        description="Update task list. Track progress on multi-step tasks.",
        input_schema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "The todo items to manage.",
                    "items": {
                        "type": "object",
                        "description": "A todo item.",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "The todo item id.",
                            },
                            "text": {
                                "type": "string",
                                "description": "The todo item text.",
                            },
                            "status": {
                                "type": "string",
                                "description": "The todo item status.",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["text"],
                    },
                    "maxItems": 10,
                }
            },
            "required": ["items"],
        },
        handler=TO_DO_MANAGER.update,
    ),
    BuiltinToolSpec(
        name="spawn_exploration_subagent",
        description="Spawn an exploration subagent for read-only research and return its summary.",
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The exploration task to delegate.",
                },
                "max_loop": {
                    "type": "integer",
                    "description": "Optional max model-call loops for the subagent.",
                    "minimum": 1,
                    "default": 10,
                },
            },
            "required": ["task"],
        },
        handler=spawn_exploration_subagent,
    ),
    BuiltinToolSpec(
        name="load_skill",
        description="Load specialized skill and knowledge by name.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the skill to load.",
                }
            },
            "required": ["name"],
        },
        handler=load_skill
    )
]


def register_builtin_tools(tool_manager: ToolManager):
    for spec in TOOL_SPECS:
        tool_manager.register(spec.build_tool(), spec.handler)