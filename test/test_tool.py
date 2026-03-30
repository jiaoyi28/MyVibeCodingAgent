import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent.tool as tool_module
import agent.utils as utils_module


class ToolSearchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)

        self.workspace_patcher = patch.object(tool_module, "WORKSPACE", self.workspace)
        self.safe_path_workspace_patcher = patch.object(
            utils_module, "WORKSPACE", self.workspace
        )
        self.workspace_patcher.start()
        self.safe_path_workspace_patcher.start()

    def tearDown(self) -> None:
        self.workspace_patcher.stop()
        self.safe_path_workspace_patcher.stop()
        self.temp_dir.cleanup()

    def write_file(self, relative_path: str, content: str) -> None:
        file_path = self.workspace / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    def test_glob_matches_files_from_base_path(self) -> None:
        self.write_file("agent/tool.py", "print('tool')\n")
        self.write_file("agent/agent.py", "print('agent')\n")
        self.write_file("docs/readme.md", "# docs\n")

        result = tool_module.glob("*.py", base_path="agent")

        self.assertEqual(result.splitlines(), ["agent/agent.py", "agent/tool.py"])

    def test_glob_supports_relative_patterns(self) -> None:
        self.write_file("src/core/main.py", "print('main')\n")
        self.write_file("src/utils/helper.py", "print('helper')\n")

        result = tool_module.glob("core/*.py", base_path="src")

        self.assertEqual(result.splitlines(), ["src/core/main.py"])

    def test_grep_filters_by_file_pattern_and_reports_line_numbers(self) -> None:
        self.write_file("agent/tool.py", "alpha\nbeta\nAlpha tool\n")
        self.write_file("notes.txt", "alpha outside python\n")

        result = tool_module.grep("alpha", file_pattern="*.py")

        self.assertEqual(
            result.splitlines(),
            ["agent/tool.py:1:alpha", "agent/tool.py:3:Alpha tool"],
        )

    def test_grep_case_sensitive(self) -> None:
        self.write_file("agent/tool.py", "alpha\nAlpha\n")

        insensitive = tool_module.grep("alpha", file_pattern="*.py")
        sensitive = tool_module.grep("alpha", file_pattern="*.py", case_sensitive=True)

        self.assertEqual(
            insensitive.splitlines(),
            ["agent/tool.py:1:alpha", "agent/tool.py:2:Alpha"],
        )
        self.assertEqual(sensitive.splitlines(), ["agent/tool.py:1:alpha"])

    def test_grep_returns_regex_error(self) -> None:
        result = tool_module.grep("(")

        self.assertIn("Error: invalid regex pattern:", result)

    def test_tool_manager_instances_are_isolated(self) -> None:
        first = tool_module.ToolManager()
        second = tool_module.ToolManager()

        first.register(
            tool_module.BuiltinToolSpec(
                name="only_first",
                description="first-only tool",
                input_schema={"type": "object", "properties": {}, "required": []},
                handler=lambda: "ok",
            ).build_tool(),
            lambda: "ok",
        )

        self.assertIsNot(first, second)
        self.assertIn("only_first", first.tools)
        self.assertNotIn("only_first", second.tools)

    def test_todo_state_is_isolated_per_registered_tool_manager(self) -> None:
        first = tool_module.ToolManager()
        second = tool_module.ToolManager()

        tool_module.register_builtin_tools(first)
        tool_module.register_builtin_tools(second)

        first_result = first.execute(
            "todo",
            items=[{"id": "1", "text": "first task", "status": "in_progress"}],
        )
        second_result = second.execute(
            "todo",
            items=[{"id": "2", "text": "second task", "status": "pending"}],
        )

        self.assertIn("first task", first_result)
        self.assertNotIn("second task", first_result)
        self.assertIn("second task", second_result)
        self.assertNotIn("first task", second_result)

    def test_register_builtin_tools_skips_subagent_without_handler(self) -> None:
        tool_manager = tool_module.ToolManager()

        tool_module.register_builtin_tools(tool_manager)

        self.assertNotIn("spawn_exploration_subagent", tool_manager.tools)

    def test_register_builtin_tools_includes_subagent_with_handler(self) -> None:
        tool_manager = tool_module.ToolManager()

        def fake_subagent_handler(task: str, max_loop: int = 10) -> str:
            return f"{task}:{max_loop}"

        tool_module.register_builtin_tools(
            tool_manager,
            spawn_exploration_subagent_handler=fake_subagent_handler,
        )

        self.assertIn("spawn_exploration_subagent", tool_manager.tools)
        self.assertEqual(
            tool_manager.execute("spawn_exploration_subagent", task="inspect", max_loop=3),
            "inspect:3",
        )


if __name__ == "__main__":
    unittest.main()
