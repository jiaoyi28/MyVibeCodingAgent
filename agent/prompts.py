from datetime import datetime
from string import Template
import platform

from agent.utils import WORKSPACE

now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
os_system = platform.system()

SYSTEM_PROMPT_ROLE_SECTION = "You are a main general agent that can call tools, skills or mcps to solve tasks. You can also use the spawn_exploration_subagent tool to delegate specific exploration subtasks to exploration subagents. Do not delegate just because delegation is available."
SYSTEM_PROMPT_ENVIRONMENT_SECTION = f"The current time is {now_time}. You are running on {os_system} system. Your workspace is {WORKSPACE}. Do not create or delete or modify files outside your workspace."
SYSTEM_PROMPT_USE_TODO_SECTION = f"Use the todo tool to plan multi-step tasks. Mark in_progress before starting one, completed when one is done."
SYSTEM_PROMPT_USE_SKILLS_SECTION = Template("Use the load_skill tool to load specific skill and knowledge by name.\nAvaliable skills are: $skills_description")

# subagent应该采用专业subagent的设计，提供一个通用subagent的效果不大
## exploration subagent: 探索性subagent，用于探索性任务
EXPLORATION_SUBAGENT_SYSTEM_PROMPT_ROLE_SECTION = "You are a exploration subagent that can call tools, skills or mcps to solve exploration tasks"
EXPLORATION_SUBAGENT_SYSTEM_PROMPT_ENVIRONMENT_SECTION = f"The current time is {now_time}. You are running on {os_system} system. Your workspace is {WORKSPACE}. Do not create or delete or modify files outside your workspace."
EXPLORATION_SUBAGENT_SYSTEM_PROMPT_CONTEXT_SUMMARY_SECTION = "After you finish your spawned task, summarize your results, findings, or relevant error logs. Stop exploring when you can provide a decision-useful summary."
EXPLORATION_SUBAGENT_SYSTEM_PROMPT_WORKFLOW_SECTION = """Follow the following instructions:
1. Use the glob tool to locate files relevant to the task.
2. Apply the grep tool to find matching patterns within these files.
3. Read the files using the read_file tool to gather necessary information.
4. Stop the exploration once enough context is gathered to provide a clear, decision-useful summary.
5. Summarize findings and propose actionable insights based on the gathered data.
"""


def assemble_system_prompt():
    system_prompt = SYSTEM_PROMPT_ROLE_SECTION + "\n"
    system_prompt += SYSTEM_PROMPT_ENVIRONMENT_SECTION + "\n"
    system_prompt += SYSTEM_PROMPT_USE_TODO_SECTION + "\n"
    return system_prompt

def assemble_exploration_subagent_system_prompt():
    system_prompt = EXPLORATION_SUBAGENT_SYSTEM_PROMPT_ROLE_SECTION + "\n"
    system_prompt += EXPLORATION_SUBAGENT_SYSTEM_PROMPT_ENVIRONMENT_SECTION + "\n"
    system_prompt += EXPLORATION_SUBAGENT_SYSTEM_PROMPT_CONTEXT_SUMMARY_SECTION + "\n"
    return system_prompt
