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


if __name__ == "__main__":
    unittest.main()
