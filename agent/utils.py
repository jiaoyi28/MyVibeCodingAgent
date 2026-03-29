from pathlib import Path

USER_HOME = Path.home()
WORKSPACE = Path(__file__).parent.parent

def safe_path(p: str) -> Path:
    path = (WORKSPACE / p).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError(f"Path escapes workspace: {p}")
    return path