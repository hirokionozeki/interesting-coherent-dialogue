"""
Evaluation utilities for knowledge-grounded dialogue.

Provides F1 score calculation and other evaluation metrics.
"""

from collections import Counter
from typing import List, Union

import nltk


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for evaluation.

    Args:
        text: Input text string

    Returns:
        List of lowercased tokens
    """
    text = text.strip().lower()
    return nltk.word_tokenize(text)


def f1_score(prediction: str, references: List[str]) -> float:
    """
    Calculate token-level F1 score.

    Args:
        prediction: Predicted text
        references: List of reference texts

    Returns:
        F1 score (0-1 range)
    """
    pred_tokens = tokenize(prediction)
    ref_tokens_list = [tokenize(ref) for ref in references]

    # Calculate F1 against all references, take maximum
    f1_scores = []
    for ref_tokens in ref_tokens_list:
        f1 = _calculate_f1(pred_tokens, ref_tokens)
        f1_scores.append(f1)

    return max(f1_scores) if f1_scores else 0.0


def _calculate_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """
    Calculate F1 score between two token lists.

    Args:
        pred_tokens: Predicted tokens
        ref_tokens: Reference tokens

    Returns:
        F1 score
    """
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # Calculate overlap
    overlap = 0
    for token in pred_counter:
        overlap += min(pred_counter[token], ref_counter.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def exact_match(prediction: str, references: List[str]) -> bool:
    """
    Check if prediction exactly matches any reference.

    Args:
        prediction: Predicted text
        references: List of reference texts

    Returns:
        True if exact match found, False otherwise
    """
    pred_normalized = " ".join(tokenize(prediction))
    ref_normalized = [" ".join(tokenize(ref)) for ref in references]
    return pred_normalized in ref_normalized


def eval_f1(predictions: List[str], references: List[str]) -> float:
    """
    Calculate average F1 score over a dataset.

    Args:
        predictions: List of predicted texts
        references: List of reference texts (one per prediction)

    Returns:
        Average F1 score (0-100 range)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")

    f1_scores = []
    for pred, ref in zip(predictions, references):
        f1_scores.append(f1_score(pred, [ref]))

    return sum(f1_scores) / len(f1_scores) * 100


def eval_all(predictions: List[str], references: List[str]) -> dict:
    """
    Calculate multiple evaluation metrics.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Dictionary with evaluation metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")

    f1_scores = []
    em_scores = []

    for pred, ref in zip(predictions, references):
        f1_scores.append(f1_score(pred, [ref]))
        em_scores.append(int(exact_match(pred, [ref])))

    return {
        "f1": sum(f1_scores) / len(f1_scores) * 100,
        "em": sum(em_scores) / len(em_scores) * 100,
        "count": len(predictions),
    }
