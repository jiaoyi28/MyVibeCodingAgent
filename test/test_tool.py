import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock
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

    def test_glob_excludes_git_directory(self) -> None:
        self.write_file(".git/objects/ab/cdef1234", "blob")
        self.write_file(".git/HEAD", "ref: refs/heads/main\n")
        self.write_file("src/main.py", "print('hi')\n")

        result = tool_module.glob("*")

        self.assertNotIn(".git", result)
        self.assertIn("src/main.py", result)

    def test_glob_excludes_node_modules(self) -> None:
        self.write_file("node_modules/lodash/index.js", "module.exports = {};\n")
        self.write_file("src/app.js", "console.log('app');\n")

        result = tool_module.glob("*.js")

        self.assertNotIn("node_modules", result)
        self.assertIn("src/app.js", result)

    def test_glob_excludes_pycache(self) -> None:
        self.write_file("__pycache__/mod.cpython-312.pyc", "\x00")
        self.write_file("mod.py", "pass\n")

        result = tool_module.glob("*")

        self.assertNotIn("__pycache__", result)
        self.assertIn("mod.py", result)

    def test_grep_excludes_git_directory(self) -> None:
        self.write_file(".git/config", "alpha in git config\n")
        self.write_file("src/main.py", "alpha in source\n")

        result = tool_module.grep("alpha")

        self.assertNotIn(".git", result)
        self.assertIn("src/main.py:1:alpha in source", result)

    def test_grep_excludes_node_modules(self) -> None:
        self.write_file("node_modules/pkg/index.js", "alpha\n")
        self.write_file("app.js", "alpha\n")

        result = tool_module.grep("alpha")

        self.assertNotIn("node_modules", result)
        self.assertIn("app.js:1:alpha", result)

    def test_grep_returns_regex_error(self) -> None:
        result = tool_module.grep("(")

        self.assertIn("Error: invalid regex pattern:", result)

    def test_execute_strips_unknown_kwargs(self) -> None:
        """LLM may send extra parameters not in the schema; they should be silently dropped."""
        tm = tool_module.ToolManager()
        tool_module.register_builtin_tools(tm)

        result = tm.execute("run_command", command="echo hello", description="run echo")

        self.assertNotIn("unexpected keyword argument", result)

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

    def test_register_builtin_tools_load_skill_uses_injected_skill_manager(self) -> None:
        tool_manager = tool_module.ToolManager()
        skill_manager = Mock()
        skill_manager.get_content.return_value = "loaded skill"

        tool_module.register_builtin_tools(tool_manager, skill_manager=skill_manager)

        result = tool_manager.execute("load_skill", name="python")

        self.assertEqual(result, "loaded skill")
        skill_manager.get_content.assert_called_once_with("python")


class RunCommandToolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.workspace_patcher = patch.object(tool_module, "WORKSPACE", self.workspace)
        self.workspace_patcher.start()

    def tearDown(self) -> None:
        self.workspace_patcher.stop()
        self.temp_dir.cleanup()

    # -- dangerous command blocking --

    def test_blocks_unix_dangerous_commands(self) -> None:
        for cmd in ["rm -rf /", "sudo apt install", "shutdown now", "reboot", "> /dev/null"]:
            result = tool_module.run_command(cmd)
            self.assertEqual(result, "Error: Dangerous command blocked", f"should block: {cmd}")

    @patch.object(tool_module, "IS_WINDOWS", True)
    def test_blocks_windows_dangerous_commands(self) -> None:
        for cmd in ["format C:", "RD /S /Q C:\\", "DEL /F /S /Q C:\\", "reg delete HKLM", "bcdedit /set"]:
            result = tool_module.run_command(cmd)
            self.assertEqual(result, "Error: Dangerous command blocked", f"should block: {cmd}")

    def test_safe_command_not_blocked(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="hello\n", stderr="")
            result = tool_module.run_command("echo hello")
            self.assertNotIn("Dangerous command blocked", result)

    # -- shell selection --

    @patch.object(tool_module, "IS_WINDOWS", False)
    @patch("os.path.isfile", side_effect=lambda p: p == "/bin/bash")
    @patch("subprocess.run")
    def test_unix_prefers_bash(self, mock_run, mock_isfile) -> None:
        mock_run.return_value = MagicMock(stdout="ok", stderr="")
        tool_module.run_command("echo ok")
        args = mock_run.call_args[0][0]
        self.assertEqual(args, ["/bin/bash", "-c", "echo ok"])

    @patch.object(tool_module, "IS_WINDOWS", False)
    @patch("os.path.isfile", side_effect=lambda p: p == "/bin/sh")
    @patch("subprocess.run")
    def test_unix_falls_back_to_sh(self, mock_run, mock_isfile) -> None:
        mock_run.return_value = MagicMock(stdout="ok", stderr="")
        tool_module.run_command("echo ok")
        args = mock_run.call_args[0][0]
        self.assertEqual(args, ["/bin/sh", "-c", "echo ok"])

    @patch.object(tool_module, "IS_WINDOWS", True)
    @patch("os.path.isfile", return_value=False)
    @patch("shutil.which", side_effect=lambda name: "C:\\Git\\bin\\bash.exe" if name == "bash" else None)
    @patch.object(tool_module, "_is_wsl_bash", return_value=False)
    @patch("subprocess.run")
    def test_windows_uses_git_bash_from_which(self, mock_run, mock_wsl, mock_which, mock_isfile) -> None:
        mock_run.return_value = MagicMock(stdout="ok", stderr="")
        tool_module.run_command("echo ok")
        args = mock_run.call_args[0][0]
        self.assertEqual(args, ["C:\\Git\\bin\\bash.exe", "-c", "echo ok"])

    @patch.object(tool_module, "IS_WINDOWS", True)
    @patch("os.path.isfile", return_value=False)
    @patch("shutil.which", side_effect=lambda name: "C:\\Windows\\System32\\bash.exe" if name == "bash" else ("C:\\pwsh.exe" if name == "pwsh" else None))
    @patch.object(tool_module, "_is_wsl_bash", return_value=True)
    @patch("subprocess.run")
    def test_windows_skips_wsl_bash_and_falls_back_to_pwsh(self, mock_run, mock_wsl, mock_which, mock_isfile) -> None:
        mock_run.return_value = MagicMock(stdout="ok", stderr="")
        tool_module.run_command("echo ok")
        args = mock_run.call_args[0][0]
        self.assertEqual(args, ["C:\\pwsh.exe", "-NoProfile", "-Command", "echo ok"])

    @patch.object(tool_module, "IS_WINDOWS", True)
    @patch("os.path.isfile", return_value=False)
    @patch("shutil.which", side_effect=lambda name: "C:\\pwsh.exe" if name == "pwsh" else None)
    @patch("subprocess.run")
    def test_windows_falls_back_to_pwsh(self, mock_run, mock_which, mock_isfile) -> None:
        mock_run.return_value = MagicMock(stdout="ok", stderr="")
        tool_module.run_command("echo ok")
        args = mock_run.call_args[0][0]
        self.assertEqual(args, ["C:\\pwsh.exe", "-NoProfile", "-Command", "echo ok"])

    @patch.object(tool_module, "IS_WINDOWS", True)
    @patch("os.path.isfile", return_value=False)
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_windows_falls_back_to_shell_true(self, mock_run, mock_which, mock_isfile) -> None:
        mock_run.return_value = MagicMock(stdout="ok", stderr="")
        tool_module.run_command("echo ok")
        self.assertTrue(mock_run.call_args[1].get("shell", False))

    # -- encoding --

    @patch("subprocess.run")
    def test_uses_utf8_encoding(self, mock_run) -> None:
        mock_run.return_value = MagicMock(stdout="你好", stderr="")
        tool_module.run_command("echo 你好")
        kwargs = mock_run.call_args[1]
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")

    # -- output handling --

    @patch("subprocess.run")
    def test_returns_combined_stdout_stderr(self, mock_run) -> None:
        mock_run.return_value = MagicMock(stdout="out\n", stderr="err\n")
        result = tool_module.run_command("some_cmd")
        self.assertIn("out", result)
        self.assertIn("err", result)

    @patch("subprocess.run")
    def test_returns_no_output_placeholder(self, mock_run) -> None:
        mock_run.return_value = MagicMock(stdout="", stderr="")
        result = tool_module.run_command("true")
        self.assertEqual(result, "(no output)")

    @patch("subprocess.run")
    def test_truncates_long_output(self, mock_run) -> None:
        mock_run.return_value = MagicMock(stdout="x" * 60000, stderr="")
        result = tool_module.run_command("big_output")
        self.assertEqual(len(result), 50000)

    # -- error handling --

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="sleep", timeout=120))
    def test_timeout_returns_error(self, mock_run) -> None:
        result = tool_module.run_command("sleep 999")
        self.assertEqual(result, "Error: Timeout (120s)")

    @patch("subprocess.run", side_effect=FileNotFoundError("not found"))
    def test_shell_not_found_returns_error(self, mock_run) -> None:
        result = tool_module.run_command("echo hi")
        self.assertIn("Error: Shell not found", result)
        self.assertIn("platform=", result)

    @patch("subprocess.run", side_effect=OSError("weird error"))
    def test_generic_exception_returns_error(self, mock_run) -> None:
        result = tool_module.run_command("echo hi")
        self.assertIn("Error:", result)
        self.assertIn("weird error", result)


class IsDangerousTestCase(unittest.TestCase):
    """Direct tests for _is_dangerous helper."""

    def test_normal_commands_are_safe(self) -> None:
        for cmd in ["echo hello", "ls -la", "python --version", "git status"]:
            self.assertFalse(tool_module._is_dangerous(cmd), f"should be safe: {cmd}")

    def test_case_insensitive_unix(self) -> None:
        self.assertTrue(tool_module._is_dangerous("RM -RF /"))
        self.assertTrue(tool_module._is_dangerous("Sudo apt install"))

    @patch.object(tool_module, "IS_WINDOWS", True)
    def test_case_insensitive_windows(self) -> None:
        self.assertTrue(tool_module._is_dangerous("FORMAT C:"))
        self.assertTrue(tool_module._is_dangerous("Reg Delete HKLM"))

    @patch.object(tool_module, "IS_WINDOWS", False)
    def test_windows_commands_not_checked_on_unix(self) -> None:
        self.assertFalse(tool_module._is_dangerous("format c:"))


class IsWslBashTestCase(unittest.TestCase):
    """Direct tests for _is_wsl_bash helper."""

    def test_detects_system32_bash(self) -> None:
        self.assertTrue(tool_module._is_wsl_bash(r"C:\Windows\System32\bash.exe"))

    def test_detects_sysnative_bash(self) -> None:
        self.assertTrue(tool_module._is_wsl_bash(r"C:\Windows\Sysnative\bash.exe"))

    def test_git_bash_is_not_wsl(self) -> None:
        self.assertFalse(tool_module._is_wsl_bash(r"C:\Program Files\Git\bin\bash.exe"))


class GetShellTestCase(unittest.TestCase):
    """Direct tests for _get_shell helper."""

    @patch.object(tool_module, "IS_WINDOWS", False)
    @patch("os.path.isfile", return_value=False)
    def test_returns_none_when_no_shell_on_unix(self, mock_isfile) -> None:
        shell_exe, prefix = tool_module._get_shell()
        self.assertIsNone(shell_exe)
        self.assertEqual(prefix, [])

    @patch.object(tool_module, "IS_WINDOWS", True)
    @patch("os.path.isfile", return_value=False)
    @patch("shutil.which", return_value=None)
    def test_returns_none_when_no_shell_on_windows(self, mock_which, mock_isfile) -> None:
        shell_exe, prefix = tool_module._get_shell()
        self.assertIsNone(shell_exe)
        self.assertEqual(prefix, [])

    @patch.object(tool_module, "IS_WINDOWS", True)
    @patch("os.path.isfile", side_effect=lambda p: "Git" in p and "bash" in p)
    def test_prefers_well_known_git_bash_path(self, mock_isfile) -> None:
        shell_exe, prefix = tool_module._get_shell()
        self.assertIsNotNone(shell_exe)
        self.assertIn("Git", shell_exe)
        self.assertEqual(prefix[-1], "-c")

    @patch.object(tool_module, "IS_WINDOWS", True)
    @patch("os.path.isfile", return_value=False)
    @patch("shutil.which", side_effect=lambda name: r"C:\Windows\System32\bash.exe" if name == "bash" else ("C:\\pwsh.exe" if name == "pwsh" else None))
    def test_skips_wsl_bash_from_which(self, mock_which, mock_isfile) -> None:
        shell_exe, prefix = tool_module._get_shell()
        self.assertEqual(shell_exe, "C:\\pwsh.exe")


if __name__ == "__main__":
    unittest.main()
