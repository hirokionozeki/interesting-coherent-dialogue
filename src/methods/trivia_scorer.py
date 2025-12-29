"""On-demand trivia score calculation using GPT-4o.

This module provides functionality to calculate trivia scores for knowledge
candidates in real-time using the OpenAI API, as a supplement when pre-computed
scores are not available.
"""

import os
import re
import time
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Default prompt for trivia score prediction
DEFAULT_PROMPT = """### Task ###
You will be given a topic and one piece of knowledge related to that topic.
Your task is to classify this piece of knowledge into one of three labels based on its level of interestingness: Good Trivia, Trivia, or Not Trivia.
The definitions for each label are as follows:
 - Good Trivia: The knowledge is an interesting fact that is unusual, unexpected, or unique.
 - Trivia: The knowledge is not interesting fact that is unusual, unexpected, or unique.
 - Not Trivia: The knowledge is a common, expected, or irrelevant fact.
Provide the reasoning for your classification, and then output the label at the end.
Start with ### Reason ### and provide a clear explanation for your classification decision, detailing why the knowledge falls into the selected category.
End with ### Label ### followed by the chosen label (Good Trivia, Trivia, or Not Trivia).

"""


class TriviaScorer:
    """Calculate trivia scores on-demand using OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-2024-08-06",
        num_samples: int = 5,
        temperature: float = 1.0,
        prompt: Optional[str] = None,
        cache_scores: bool = True
    ):
        """Initialize the trivia scorer.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY environment variable
            model: OpenAI model to use for scoring
            num_samples: Number of times to sample for each knowledge (default: 5)
            temperature: Temperature for sampling (default: 1.0)
            prompt: Custom prompt template. If None, uses default prompt
            cache_scores: Whether to cache computed scores (default: True)

        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not provided
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for on-demand trivia scoring. "
                "Install it with: pip install openai"
            )

        # Get API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Provide it via api_key parameter "
                "or set OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.num_samples = num_samples
        self.temperature = temperature
        self.prompt_template = prompt or DEFAULT_PROMPT
        self.cache_scores = cache_scores
        self.score_cache: Dict[Tuple[str, str], float] = {}

    def _label_to_score(self, label_text: str) -> int:
        """Convert label text to numeric score.

        Args:
            label_text: Label text from model output

        Returns:
            Numeric score: 2 (Good Trivia), 1 (Trivia), 0 (Not Trivia), -1 (error)
        """
        # Try direct match first
        label_text_lower = label_text.lower().strip()
        if "good trivia" in label_text_lower:
            return 2
        elif "not trivia" in label_text_lower:
            return 0
        elif "trivia" in label_text_lower:
            return 1

        # Try to extract from ### Label ### section
        if "### label ###" in label_text_lower:
            label_section = label_text.split("### Label ###")[-1].strip()
            if "good trivia" in label_section.lower():
                return 2
            elif "not trivia" in label_section.lower():
                return 0
            elif "trivia" in label_section.lower():
                return 1

        # Try to find numeric value
        if re.search(r'\b2\b', label_text) and not re.search(r'\b[01]\b', label_text):
            return 2
        elif re.search(r'\b1\b', label_text) and not re.search(r'\b[02]\b', label_text):
            return 1
        elif re.search(r'\b0\b', label_text) and not re.search(r'\b[12]\b', label_text):
            return 0

        return -1

    def calculate_score(
        self,
        topic: str,
        knowledge: str,
        retry_on_error: bool = True,
        max_retries: int = 3
    ) -> float:
        """Calculate trivia score for a single piece of knowledge.

        Args:
            topic: Topic (e.g., Wikipedia article title)
            knowledge: Knowledge text to score
            retry_on_error: Whether to retry on API errors
            max_retries: Maximum number of retries for API calls

        Returns:
            Average trivia score (0.0 to 2.0)

        Raises:
            RuntimeError: If all attempts fail
        """
        # Check cache
        cache_key = (topic, knowledge)
        if self.cache_scores and cache_key in self.score_cache:
            return self.score_cache[cache_key]

        # Construct prompt
        prompt = self.prompt_template
        prompt += f"### Topic ###\n{topic}\n\n"
        prompt += f"### Knowledge ###\n{knowledge}\n\n"
        prompt += "### Reason ###\n"

        # Call API multiple times
        scores = []
        last_error = None

        for attempt in range(max_retries if retry_on_error else 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert evaluator of knowledge."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    n=self.num_samples,
                )

                # Extract scores from responses
                for choice in response.choices:
                    content = choice.message.content
                    # Extract label from last line
                    label_text = content.strip().split('\n')[-1]
                    score = self._label_to_score(label_text)

                    if score != -1:
                        scores.append(score)

                # If we got valid scores, break retry loop
                if scores:
                    break

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1 and retry_on_error:
                    # Wait before retry
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise RuntimeError(
                        f"Failed to calculate trivia score after {max_retries} attempts: {e}"
                    ) from e

        # Calculate average score
        if not scores:
            raise RuntimeError(
                f"Could not extract valid scores from model outputs. Last error: {last_error}"
            )

        avg_score = sum(scores) / len(scores)

        # Cache result
        if self.cache_scores:
            self.score_cache[cache_key] = avg_score

        return avg_score

    def calculate_batch_scores(
        self,
        topic: str,
        knowledge_list: List[str],
        show_progress: bool = False
    ) -> List[float]:
        """Calculate trivia scores for a batch of knowledge candidates.

        Args:
            topic: Topic for all knowledge candidates
            knowledge_list: List of knowledge texts to score
            show_progress: Whether to print progress (default: False)

        Returns:
            List of average trivia scores
        """
        scores = []
        for i, knowledge in enumerate(knowledge_list):
            if show_progress:
                print(f"Calculating trivia score {i+1}/{len(knowledge_list)}...")

            try:
                score = self.calculate_score(topic, knowledge)
                scores.append(score)
            except Exception as e:
                print(f"Warning: Failed to calculate score for knowledge {i}: {e}")
                # Use neutral score on error
                scores.append(1.0)

        return scores

    def clear_cache(self) -> None:
        """Clear the score cache."""
        self.score_cache.clear()

    def get_cache_size(self) -> int:
        """Get the number of cached scores.

        Returns:
            Number of cached (topic, knowledge) pairs
        """
        return len(self.score_cache)
