# MyVibeCodingAgent

一个还在搭建中的 Python Agent 原型项目，目标是基于 OpenAI Responses API 实现一个可调用本地工具的命令行代理。

## 当前状态

项目结构已经初步搭好，但**当前版本还不能稳定运行**。我检查后确认，仓库目前更接近“骨架代码”，适合继续开发，不适合直接作为可用成品使用。

## 项目结构

```text
agent/
  agent.py      # 主循环，负责调用 OpenAI API 和执行工具
  tool.py       # ToolManager 与本地文件/命令工具
  schema.py     # tools / messages 的 schema 定义
  prompts.py    # system prompt 组装
  utils.py      # 工作区路径与安全路径处理
  mcp.py        # 预留，当前为空
  skill.py      # 预留，当前为空
```

## 环境要求

- Python 3.13+
- OpenAI Python SDK
- `python-dotenv`

当前 `pyproject.toml` 只声明了 `openai`，但代码中还依赖了 `python-dotenv`，因此安装依赖时需要额外补上。

## 环境变量

项目代码中使用了以下环境变量：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `MODEL`

建议在项目根目录放置 `.env` 文件进行管理。

## 已实现能力

- 定义基础消息结构 `Message`
- 定义更通用的 `JsonSchema` / `Tool` 数据结构
- 保留 `ToolParameters` / `ToolProperty` 作为兼容别名
- 支持输出 OpenAI `parameters` 风格，以及 `input_schema` 风格的工具 schema
- 通过声明式 `BuiltinToolSpec` 注册内置工具，新增工具时只需补充 schema 和 handler
- 注册本地工具：
  - `bash`
  - `read_file`
  - `write_file`
  - `edit_file`
  - `todo`
- 对工作区文件路径进行简单越界保护

## 快速安装

如果你只是想先把依赖装起来，可以先执行：

```bash
pip install -e .
pip install python-dotenv
```

## 后续建议

建议按下面顺序继续完善：

1. 先修复 `agent.py` 与 `prompts.py` 中会导致主流程无法运行的问题。
2. 继续统一 `Message` 和 `Tool` 的序列化边界，减少不同 API 分支中的重复适配代码。
3. 为工具参数校验、tool call arguments 的 JSON 解析与 tool result 回传补充更完整的错误处理。
4. 再补充 `mcp.py`、`skill.py` 与更完整的 README 示例。