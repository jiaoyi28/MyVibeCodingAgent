from datetime import datetime
import platform

now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
os_system = platform.system()

SYSTEM_PROMPT_ROLE_SECTION = "You are a general agent that can call tools, skills or mcps to solve tasks"
SYSTEM_PROMPT_ENVIRONMENT_SECTION = f"The current time is {now_time}. You are running on {os_system} system."
SYSTEM_PROMPT_USE_TODO_SECTION = f"Use the todo tool to plan multi-step tasks. Mark in_progress before starting one, completed when one is done."


def assemble_system_prompt():
    system_prompt = SYSTEM_PROMPT_ROLE_SECTION + "\n"
    system_prompt += SYSTEM_PROMPT_ENVIRONMENT_SECTION + "\n"
    system_prompt += SYSTEM_PROMPT_USE_TODO_SECTION + "\n"
    return system_prompt