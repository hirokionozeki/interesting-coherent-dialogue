"""Dialogue breakdown detection using LLM."""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from openai import OpenAI


DEFAULT_ROLE_PROMPT = "You are an expert detector of dialogue breakdown."
DEFAULT_TASK_PROMPT_FILE = "configs/prompts/dbd_task.txt"


class BreakdownDetector:

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        cache_file: Optional[str] = None,
        role_prompt: str = DEFAULT_ROLE_PROMPT,
        task_prompt_file: str = DEFAULT_TASK_PROMPT_FILE,
    ):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.role_prompt = role_prompt

        task_prompt_path = Path(task_prompt_file)
        if not task_prompt_path.exists():
            raise FileNotFoundError(f"Task prompt file not found: {task_prompt_file}")
        with open(task_prompt_path, "r") as f:
            self.task_prompt = f.read()

        self.cache: Dict[str, Dict[str, bool]] = {}
        self.cache_file = cache_file
        if cache_file:
            self.load_cache(cache_file)

    def load_cache(self, cache_file: str) -> None:
        cache_path = Path(cache_file)
        if cache_path.exists():
            with open(cache_path, "r") as f:
                self.cache = json.load(f)

    def save_cache(self) -> None:
        if self.cache_file:
            cache_path = Path(self.cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(self.cache, f, indent=2)

    def check_cache(self, data_idx: int, knowledge_idx: int) -> Optional[bool]:
        data_key = str(data_idx)
        kn_key = str(knowledge_idx)
        if data_key in self.cache and kn_key in self.cache[data_key]:
            return self.cache[data_key][kn_key]
        return None

    def detect(
        self,
        response: str,
        dialogue_topic: str,
        dialogue_history: list[Tuple[str, str]],
        data_idx: Optional[int] = None,
        knowledge_idx: Optional[int] = None,
    ) -> Tuple[bool, str]:
        if data_idx is not None and knowledge_idx is not None:
            cached_result = self.check_cache(data_idx, knowledge_idx)
            if cached_result is not None:
                return cached_result, "from cache"

        if len(dialogue_history) == 0:
            return False, "No dialogue history to evaluate"

        prompt = self._build_prompt(response, dialogue_topic, dialogue_history)
        is_breakdown, gpt_response = self._call_gpt(prompt)

        if data_idx is not None and knowledge_idx is not None:
            data_key = str(data_idx)
            kn_key = str(knowledge_idx)
            if data_key not in self.cache:
                self.cache[data_key] = {}
            self.cache[data_key][kn_key] = is_breakdown

        return is_breakdown, gpt_response

    def _build_prompt(
        self,
        response: str,
        dialogue_topic: str,
        dialogue_history: list[Tuple[str, str]],
    ) -> str:
        prompt = self.task_prompt + "\n"
        prompt += "### Dialogue Topic ###\n"
        prompt += dialogue_topic + "\n"
        prompt += "### Dialogue History ###\n"
        for speaker, utterance in dialogue_history:
            if speaker == "Apprentice":
                prompt += "SpeakerA: " + utterance + "\n"
            elif speaker == "Wizard":
                prompt += "SpeakerB: " + utterance + "\n"
            else:
                prompt += f"{speaker}: {utterance}\n"
        prompt += "\n### Response ###\n"
        prompt += "SpeakerB: " + response + "\n"
        prompt += "\n### Reason ###\n"
        return prompt

    def _call_gpt(self, prompt: str, max_retries: int = 2) -> Tuple[bool, str]:
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": self.role_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )

                gpt_response = completion.choices[0].message.content
                pred_text = gpt_response.split("\n")[-1].strip()
                is_breakdown = self._parse_label(pred_text)

                if is_breakdown is not None:
                    return is_breakdown, gpt_response

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    return True, f"API call failed: {str(e)}"

        return True, "Failed to parse GPT response"

    @staticmethod
    def _parse_label(label_text: str) -> Optional[bool]:
        if label_text == "Dialogue Breakdown":
            return True
        elif label_text == "No Dialogue Breakdown":
            return False
        else:
            return None
