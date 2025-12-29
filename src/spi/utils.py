"""
SPI Utilities

Helper functions for SPI inference and evaluation.
"""

import json
from typing import List

import numpy as np
from tqdm import tqdm


def get_sorted_indices(data_list: List[float]) -> List[int]:
    """
    Get indices that would sort the list in descending order.

    Args:
        data_list: List of numeric values

    Returns:
        List of indices sorted by descending values

    Examples:
        >>> get_sorted_indices([0.1, 0.5, 0.3])
        [1, 2, 0]
    """
    return sorted(range(len(data_list)), key=lambda x: data_list[x], reverse=True)


def softmax(scores: List[float]) -> List[float]:
    """
    Apply softmax to a list of scores.

    Args:
        scores: List of raw scores

    Returns:
        List of probabilities
    """
    exp_scores = np.exp(np.array(scores) - np.max(scores))
    return (exp_scores / exp_scores.sum()).tolist()


def load_jsons(path: str) -> List[dict]:
    """
    Load JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries

    Examples:
        >>> data = load_jsons("data.jsonl")
        >>> len(data)
        100
    """
    data = []
    with open(path, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, total=len(lines), desc="Read JSONS data"):
        data.append(json.loads(line))
    return data
