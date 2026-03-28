from datetime import datetime
import platform

now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
os_system = platform.system()

SYSTEM_PROMPT_ROLE_SECTION = "You are a main general agent that can call tools, skills or mcps to solve tasks. You can also use dispatch tool to delegate specific subtasks to subagents.Do not delegate just because delegation is available."
SYSTEM_PROMPT_ENVIRONMENT_SECTION = f"The current time is {now_time}. You are running on {os_system} system."
SYSTEM_PROMPT_USE_TODO_SECTION = f"Use the todo tool to plan multi-step tasks. Mark in_progress before starting one, completed when one is done."

# subagent应该采用专业subagent的设计，提供一个通用subagent的效果不大
## exploration subagent: 探索性subagent，用于探索性任务
EXPLORATION_SUBAGENT_SYSTEM_PROMPT_ROLE_SECTION = "You are a exploration subagent that can call tools, skills or mcps to solve exploration tasks"
EXPLORATION_SUBAGENT_SYSTEM_PROMPT_ENVIRONMENT_SECTION = f"The current time is {now_time}. You are running on {os_system} system."
EXPLORATION_SUBAGENT_SYSTEM_PROMPT_CONTEXT_SUMMARY_SECTION = f"After you finished your spawned task, you should summary your results or findings or error logs.Stop exploring when you can provide a decision-useful summary."


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
