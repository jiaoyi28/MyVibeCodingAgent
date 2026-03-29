from pathlib import Path
import re
import logging

from agent.utils import WORKSPACE

logger = logging.getLogger("myvibecodingagent")

DEFAULT_SKILLS_DIR = WORKSPACE / "skills"

class SkillManager:
    def __init__(self, skills_dir: Path = DEFAULT_SKILLS_DIR):
        self.skills = {}
        self.skills_dir = skills_dir
        self._load_skills()

    def _parse_frontmatter(self, text: str) -> tuple:
        """Parse YAML frontmatter between --- delimiters."""
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return meta, match.group(2).strip()
    
    def _load_skills(self) -> None:
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory {self.skills_dir} not exists. Please check.")
            return
        logger.debug(f"Loading skills from {self.skills_dir}...")
        for file in self.skills_dir.glob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                meta, body = self._parse_frontmatter(text)
                if not meta:
                    continue
                logger.debug(f"Loading skill {meta['name']} from {file}...")
                name = meta.get("name", file.stem)
                self.skills[name] = {
                    "name": name,
                    "meta": meta,
                    "body": body,
                    "path": str(file)}
    
    def get_description(self) -> str:
        # 渐进式披露第一层
        if not self.skills:
            return "No skills available."
        
        lines = []
        for name, skill in self.skills.items():
            description = skill.get("meta", {}).get("description", "No description.")
            line = f"  - {name}: {description}"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, skill_name: str) -> str:
        # 渐进式披露第二层
        if not self.skills:
            return "No skills available."
        if skill_name not in self.skills:
            return f"Skill {skill_name} not found. Available skills: {", ".join(self.skills.keys())}."
        return f"<skill name=\"{skill_name}\">\n{self.skills[skill_name]['body']}\n</skill>"
