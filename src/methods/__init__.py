"""Proposed method components for enhancing dialogue quality.

This package contains four main components:
1. ConfidenceClassifier: Classifies knowledge selection confidence
2. TriviaReranker: Reranks knowledge by trivia scores
3. TriviaScorer: Calculates trivia scores on-demand using GPT
4. BreakdownDetector: Detects dialogue breakdown in responses
"""

from .breakdown_detector import BreakdownDetector
from .confidence_classifier import ConfidenceClassifier
from .trivia_reranker import TriviaReranker

try:
    from .trivia_scorer import TriviaScorer
    __all__ = ["ConfidenceClassifier", "TriviaReranker", "TriviaScorer", "BreakdownDetector"]
except ImportError:
    # TriviaScorer requires openai package
    __all__ = ["ConfidenceClassifier", "TriviaReranker", "BreakdownDetector"]
