import fnmatch
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
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
    """tool管理"""

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}
        self.tool_handlers: dict[str, ToolHandler] = {}

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
        tool = self.tools.get(tool_name)
        if tool and tool.parameters.properties:
            allowed = set(tool.parameters.properties)
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
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


IS_WINDOWS = sys.platform == "win32"

_DANGEROUS_UNIX = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
_DANGEROUS_WIN = [
    "format c:", "format d:",
    "rd /s /q c:\\", "del /f /s /q c:\\",
    "reg delete", "bcdedit",
]


def _is_wsl_bash(path: str) -> bool:
    """Return True if *path* points to the WSL bash.exe wrapper."""
    normalized = os.path.normcase(os.path.realpath(path))
    return "system32" in normalized or "sysnative" in normalized


def _get_shell() -> tuple[str | None, list[str]]:
    """Return (shell_path, argv_prefix) for the best available shell."""
    if not IS_WINDOWS:
        for sh in ("/bin/bash", "/bin/sh"):
            if os.path.isfile(sh):
                return sh, [sh, "-c"]
        return None, []

    # Prefer Git Bash at well-known locations before shutil.which,
    # because which("bash") often returns the WSL wrapper at System32.
    _GIT_BASH_CANDIDATES = [
        os.path.expandvars(r"%ProgramFiles%\Git\bin\bash.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Git\bin\bash.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\bin\bash.exe"),
    ]
    for candidate in _GIT_BASH_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate, [candidate, "-c"]

    found_bash = shutil.which("bash")
    if found_bash and not _is_wsl_bash(found_bash):
        return found_bash, [found_bash, "-c"]

    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh:
        return pwsh, [pwsh, "-NoProfile", "-Command"]

    return None, []


def detect_shell_name() -> str:
    """Return a human-readable name for the shell selected by _get_shell()."""
    shell_exe, _ = _get_shell()
    if shell_exe is None:
        return "cmd.exe" if IS_WINDOWS else "sh"
    lower = shell_exe.lower()
    if "bash" in lower:
        return "bash (Git Bash)" if IS_WINDOWS else "bash"
    if "pwsh" in lower or "powershell" in lower:
        return "powershell"
    return shell_exe


def _is_dangerous(command: str) -> bool:
    cmd_lower = command.lower()
    for d in _DANGEROUS_UNIX:
        if d in cmd_lower:
            return True
    if IS_WINDOWS:
        for d in _DANGEROUS_WIN:
            if d.lower() in cmd_lower:
                return True
    return False


def run_command(command: str) -> str:
    if _is_dangerous(command):
        return "Error: Dangerous command blocked"

    shell_exe, prefix_args = _get_shell()

    try:
        run_kwargs: dict[str, Any] = dict(
            cwd=WORKSPACE,
            capture_output=True,
            timeout=120,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if prefix_args:
            r = subprocess.run(prefix_args + [command], **run_kwargs)
        else:
            r = subprocess.run(command, shell=True, **run_kwargs)

        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except FileNotFoundError:
        return f"Error: Shell not found (platform={platform.system()})"
    except Exception as e:
        return f"Error: {e}"


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


def _matches_glob_ignore_case(value: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(value.casefold(), pattern.casefold())


_IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", ".eggs",
    "dist", "build", "*.egg-info",
}


def _walk(root: Path, *, dirs: bool = True, files: bool = True):
    """Yield paths under *root*, skipping common noise directories."""
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            if entry.name in _IGNORED_DIRS or entry.name.endswith(".egg-info"):
                continue
            if dirs:
                yield entry
            yield from _walk(entry, dirs=dirs, files=files)
        elif files:
            yield entry


def glob(pattern: str, base_path: str = ".") -> str:
    try:
        if not str(pattern).strip():
            return "Error: pattern is required"

        root = safe_path(base_path)
        matches = sorted(
            [
                _workspace_relative_path(path)
                for path in _walk(root)
                if _matches_glob_ignore_case(path.name, pattern)
                or _matches_glob_ignore_case(
                    str(path.relative_to(root)).replace("\\", "/"), pattern
                )
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

        for path in _walk(root, dirs=False):
            relative_to_root = str(path.relative_to(root)).replace("\\", "/")
            if not _matches_glob_ignore_case(
                path.name, file_pattern
            ) and not _matches_glob_ignore_case(
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


def load_skill(name: str, skill_manager: SkillManager | None = None) -> str:
    if not str(name).strip():
        err = "load_skill: name is required"
        logger.error(err)
        return err
    skill_manager = skill_manager or SkillManager()
    return skill_manager.get_content(name)


def build_builtin_tool_specs(
    *,
    todo_manager: ToDoManager | None = None,
    skill_manager: SkillManager | None = None,
    spawn_exploration_subagent_handler: ToolHandler | None = None,
) -> list[BuiltinToolSpec]:
    todo_manager = todo_manager or ToDoManager()
    skill_manager = skill_manager or SkillManager()
    tool_specs = [
        BuiltinToolSpec(
            name="run_command",
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
            handler=run_command,
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
                    },
                },
                "required": ["items"],
            },
            handler=todo_manager.update,
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
            handler=lambda name: load_skill(name, skill_manager=skill_manager),
        ),
    ]

    if spawn_exploration_subagent_handler is not None:
        tool_specs.append(
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
                handler=spawn_exploration_subagent_handler,
            )
        )

    return tool_specs


def register_builtin_tools(
    tool_manager: ToolManager,
    *,
    todo_manager: ToDoManager | None = None,
    skill_manager: SkillManager | None = None,
    spawn_exploration_subagent_handler: ToolHandler | None = None,
):
    for spec in build_builtin_tool_specs(
        todo_manager=todo_manager,
        skill_manager=skill_manager,
        spawn_exploration_subagent_handler=spawn_exploration_subagent_handler
    ):
        tool_manager.register(spec.build_tool(), spec.handler)