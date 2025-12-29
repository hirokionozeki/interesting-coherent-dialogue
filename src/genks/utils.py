"""
GenKS Utilities

Helper functions for GenKS inference and evaluation.
"""

from typing import List, Tuple, Union

import nltk


def split_id(collect: List[str]) -> Tuple[List[str], List[str]]:
    """
    Split knowledge ID from generated text.

    The GenKS model generates responses in the format: "<s{id}>response text"
    This function separates the knowledge ID from the response text.

    Args:
        collect: List of generated text strings

    Returns:
        Tuple of (id_collect, text_collect) where:
            - id_collect: List of knowledge IDs (e.g., "<s5>")
            - text_collect: List of response texts without IDs

    Examples:
        >>> split_id(["<s5>Hello world", "<s3>How are you"])
        (["<s5>", "<s3>"], ["Hello world", "How are you"])
        >>> split_id(["Hello world"])
        ([""], ["Hello world"])
    """
    id_collect = []
    text_collect = []

    for line in collect:
        if ">" in line:
            id_collect.append(line[: line.index(">") + 1].strip())
            text_collect.append(line[line.index(">") + 1 :].strip())
        else:
            id_collect.append("")
            text_collect.append(line)

    return id_collect, text_collect


def lower(text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Lowercase and tokenize text for evaluation.

    Args:
        text: Single string or list of strings

    Returns:
        Lowercased and tokenized text in the same format as input

    Examples:
        >>> lower("Hello World!")
        "hello world !"
        >>> lower(["Hello", "World"])
        ["hello", "world"]
    """
    if isinstance(text, str):
        text = text.strip().lower()
        text = " ".join(nltk.word_tokenize(text))
        return text.strip()
    return [lower(item) for item in text]


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


def filter_data(
    data: List, psg_filter: List, turn: int = 0
) -> Tuple[List, List]:
    """
    Filter data by turn ID.

    Args:
        data: List of data examples
        psg_filter: List of passage filters
        turn: Turn ID to filter by

    Returns:
        Tuple of (filtered_data, filtered_psg_filter)
    """
    new_data = []
    new_psg_filter = []

    for example, psg in zip(data, psg_filter):
        if example["turn_id"] == turn:
            new_data.append(example)
            new_psg_filter.append(psg)

    return new_data, new_psg_filter
