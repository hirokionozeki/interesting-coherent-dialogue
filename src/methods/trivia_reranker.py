"""Trivia score-based knowledge reranking.

This module reranks knowledge candidates using pre-computed trivia scores.
Trivia scores measure the interestingness of knowledge for dialogue.
Optionally supports on-demand score calculation when pre-computed scores are unavailable.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .trivia_scorer import TriviaScorer
    TRIVIA_SCORER_AVAILABLE = True
except ImportError:
    TRIVIA_SCORER_AVAILABLE = False


class TriviaReranker:
    """Reranker for knowledge candidates using trivia scores."""

    def __init__(
        self,
        scores_file: Optional[str] = None,
        enable_on_demand: bool = False,
        on_demand_config: Optional[Dict] = None
    ):
        """Initialize the trivia reranker.

        Args:
            scores_file: Path to JSON file containing pre-computed trivia scores
            enable_on_demand: Whether to enable on-demand score calculation
                when pre-computed scores are not available (default: False)
            on_demand_config: Configuration dict for TriviaScorer.
                Supported keys: api_key, model, num_samples, temperature, cache_scores

        Raises:
            ImportError: If enable_on_demand=True but TriviaScorer is not available
        """
        self.trivia_scores: Optional[List[List[float]]] = None
        self.enable_on_demand = enable_on_demand
        self.trivia_scorer: Optional['TriviaScorer'] = None

        # Load pre-computed scores if provided
        if scores_file:
            self.load_scores(scores_file)

        # Initialize on-demand scorer if enabled
        if enable_on_demand:
            if not TRIVIA_SCORER_AVAILABLE:
                raise ImportError(
                    "TriviaScorer is not available. Cannot enable on-demand scoring."
                )
            config = on_demand_config or {}
            self.trivia_scorer = TriviaScorer(**config)

    def load_scores(self, scores_file: str) -> None:
        """Load trivia scores from file.

        Args:
            scores_file: Path to JSON file containing trivia scores
                Expected format: List[List[float]] where each inner list contains
                trivia scores for knowledge candidates of one data instance
        """
        scores_path = Path(scores_file)
        if not scores_path.exists():
            raise FileNotFoundError(f"Trivia scores file not found: {scores_file}")

        with open(scores_path, "r") as f:
            self.trivia_scores = json.load(f)

    def rerank(
        self,
        knowledge_indices: List[int],
        data_idx: int,
        include_no_passage: bool = True,
        topic: Optional[str] = None,
        knowledge_texts: Optional[List[str]] = None
    ) -> List[int]:
        """Rerank knowledge indices by trivia scores.

        Args:
            knowledge_indices: List of knowledge indices to rerank
            data_idx: Index of the current data instance
            include_no_passage: Whether to include no_passages_used (index 0) in scores
            topic: Topic for on-demand scoring (required if using on-demand scoring)
            knowledge_texts: List of knowledge texts for on-demand scoring
                (required if using on-demand scoring)

        Returns:
            List of knowledge indices reranked by trivia score (highest first)

        Raises:
            RuntimeError: If trivia scores are not loaded and on-demand is disabled
            ValueError: If data_idx is out of range or trivia scores are invalid,
                or if required parameters for on-demand scoring are missing
        """
        # Try to use pre-computed scores first
        if self.trivia_scores is not None:
            if data_idx >= len(self.trivia_scores):
                raise ValueError(
                    f"data_idx {data_idx} out of range. "
                    f"Max index: {len(self.trivia_scores) - 1}"
                )

            # Get trivia scores for this data instance
            instance_scores = self.trivia_scores[data_idx]

            # Add no_passages_used score (always 0) if needed
            if include_no_passage:
                full_scores = [0.0] + instance_scores
            else:
                full_scores = instance_scores

        # Fall back to on-demand scoring
        elif self.enable_on_demand and self.trivia_scorer is not None:
            if topic is None or knowledge_texts is None:
                raise ValueError(
                    "topic and knowledge_texts are required for on-demand scoring"
                )

            # Calculate scores on-demand
            print(f"Computing trivia scores on-demand for data_idx {data_idx}...")
            instance_scores = self.trivia_scorer.calculate_batch_scores(
                topic=topic,
                knowledge_list=knowledge_texts,
                show_progress=False
            )

            # Add no_passages_used score (always 0) if needed
            if include_no_passage:
                full_scores = [0.0] + instance_scores
            else:
                full_scores = instance_scores

        else:
            raise RuntimeError(
                "Trivia scores not loaded and on-demand scoring is disabled. "
                "Either call load_scores() or enable on-demand scoring."
            )

        # Get scores for selected indices
        selected_scores = []
        for idx in knowledge_indices:
            if idx >= len(full_scores):
                raise ValueError(
                    f"Knowledge index {idx} out of range. "
                    f"Max index: {len(full_scores) - 1}"
                )
            score = full_scores[idx]
            if score == -1:
                raise ValueError(
                    f"Invalid trivia score (-1) at index {idx}. "
                    "This indicates a data preparation error."
                )
            selected_scores.append(score)

        # Rerank by trivia score (highest first)
        ranked_indices = [
            idx
            for _, idx in sorted(
                zip(selected_scores, knowledge_indices),
                key=lambda x: -x[0]
            )
        ]

        return ranked_indices

    def get_score(
        self,
        data_idx: int,
        knowledge_idx: int,
        topic: Optional[str] = None,
        knowledge_text: Optional[str] = None
    ) -> float:
        """Get trivia score for a specific knowledge candidate.

        Args:
            data_idx: Index of the data instance
            knowledge_idx: Index of the knowledge candidate
            topic: Topic for on-demand scoring (required if using on-demand scoring)
            knowledge_text: Knowledge text for on-demand scoring
                (required if using on-demand scoring and knowledge_idx > 0)

        Returns:
            Trivia score for the specified knowledge

        Raises:
            RuntimeError: If trivia scores are not loaded and on-demand is disabled
            ValueError: If indices are out of range or required parameters are missing
        """
        # Handle no_passages_used (always score 0)
        if knowledge_idx == 0:
            return 0.0

        # Try to use pre-computed scores first
        if self.trivia_scores is not None:
            if data_idx >= len(self.trivia_scores):
                raise ValueError(f"data_idx {data_idx} out of range")

            instance_scores = self.trivia_scores[data_idx]
            adjusted_idx = knowledge_idx - 1  # Subtract 1 for no_passages_used

            if adjusted_idx >= len(instance_scores):
                raise ValueError(f"knowledge_idx {knowledge_idx} out of range")

            return instance_scores[adjusted_idx]

        # Fall back to on-demand scoring
        elif self.enable_on_demand and self.trivia_scorer is not None:
            if topic is None or knowledge_text is None:
                raise ValueError(
                    "topic and knowledge_text are required for on-demand scoring"
                )

            # Calculate score on-demand
            score = self.trivia_scorer.calculate_score(
                topic=topic,
                knowledge=knowledge_text
            )
            return score

        else:
            raise RuntimeError(
                "Trivia scores not loaded and on-demand scoring is disabled. "
                "Either call load_scores() or enable on-demand scoring."
            )
