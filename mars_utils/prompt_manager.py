import os
import re
from typing import Dict


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_prompt_sections(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    sections: Dict[str, str] = {}
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections[normalize_key(title)] = content
    return sections


class PromptManager:
    def __init__(self, prompt_dir: str):
        self.prompt_dir = prompt_dir
        self.user_sections = parse_prompt_sections(
            os.path.join(prompt_dir, "ALL_userproxy_task_input.md")
        )
        self.planner_sections = parse_prompt_sections(
            os.path.join(prompt_dir, "ALL_prompt_planner_template.md")
        )

    def get_user_prompt(self, key: str) -> str:
        normalized = normalize_key(key)
        if normalized not in self.user_sections:
            raise KeyError(f"Missing user prompt for key={key}")
        return self.user_sections[normalized]

    def get_planner_prompt(self, key: str) -> str:
        normalized = normalize_key(key)
        if normalized not in self.planner_sections:
            raise KeyError(f"Missing planner prompt for key={key}")
        return self.planner_sections[normalized]
