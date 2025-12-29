"""
SPI Data Preparation

Prepares two types of datasets for SPI based on wow_less.py:
1. Origin: Without no_passages_used in knowledge candidates
2. Ours: With no_passages_used added to all knowledge candidates

This script is adapted from /Users/zeki/Documents/program/SPI/src/data_utils/wow_less.py
Key difference is in lines 165-173 of the original code.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

TOKEN_NOCHOSEN = 'no_passages_used'
TOKEN_KNOWLEDGE = '__knowledge__'


def _first_val(dictionary: Dict) -> str:
    """Get first value from dictionary."""
    vals = list(dictionary.values())
    return vals[0] if len(vals) > 0 else ''


def _first_key(dictionary: Dict) -> str:
    """Get first key from dictionary."""
    keys = list(dictionary.keys())
    return keys[0] if len(keys) > 0 else ''


def _get_chosen_title_and_sent(wizard_entry: Dict, k_dict: Dict) -> tuple:
    """
    Extract title and chosen sentence.

    Args:
        wizard_entry: Wizard turn entry
        k_dict: Knowledge dictionary

    Returns:
        Tuple of (title, sentence)
    """
    title_dict = wizard_entry.get('checked_passage', 'none')
    sentence_dict = wizard_entry.get('checked_sentence', {})
    title = None
    sentence = None

    if sentence_dict == {}:
        title = sentence = TOKEN_NOCHOSEN
    else:
        sentence = _first_val(sentence_dict)
        if sentence == TOKEN_NOCHOSEN:
            title = TOKEN_NOCHOSEN
        else:
            title = ''
            cand_title1 = _first_val(title_dict) if title_dict else ''
            cand_title2 = ' '.join(_first_key(sentence_dict).split('_')[1:-1])

            if cand_title1 and cand_title1 in k_dict and sentence in k_dict[cand_title1]:
                title = cand_title1
            elif cand_title2 in k_dict and sentence in k_dict[cand_title2]:
                title = cand_title2
            else:
                for t, passage in k_dict.items():
                    if sentence in passage:
                        title = t
                        break

    return title, sentence


def len_episode(dialog: List[Dict]) -> tuple:
    """Calculate episode length."""
    wizard_first = 'Wizard' in dialog[0]['speaker']
    dialog_len = (len(dialog) - 1) // 2 if wizard_first else len(dialog) // 2
    return dialog_len * 2 - 1 if wizard_first else dialog_len * 2, dialog_len


def process_dataset(
    input_path: str,
    output_path: str,
    history_length: int = 1,
    max_knowledge: int = -1,
    add_no_passages: bool = False
) -> List[Dict]:
    """
    Process WoW dataset for SPI.

    Args:
        input_path: Path to raw WoW JSON file
        output_path: Path to save processed JSONL file
        history_length: Number of history turns to keep
        max_knowledge: Maximum number of knowledge candidates (-1 for all)
        add_no_passages: Whether to add no_passages_used to knowledge candidates

    Returns:
        List of processed samples
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    samples = []
    for dialog_idx, element in enumerate(data):
        chosen_topic = element.get("chosen_topic", "")
        chosen_topic_passage = element["chosen_topic_passage"]
        dialog = element["dialog"]

        max_len_per_d, dd = len_episode(dialog)
        history = []
        dialog = dialog[:max_len_per_d]

        for turn_idx, turn in enumerate(dialog):
            speaker = turn["speaker"]
            utterance = turn["text"].strip()

            if turn_idx == 0 and "wizard" in speaker.lower():
                history.append(chosen_topic)
            elif turn_idx == 0:
                utterance = chosen_topic + "\n" + utterance

            if "wizard" in speaker.lower():
                # Build knowledge dictionary
                apprentice_ret_passages = wizard_ret_passages = {}
                if turn_idx != 0:
                    apprentice_entry = dialog[turn_idx - 1]
                    apprentice_ret_passages = apprentice_entry["retrieved_passages"]
                if turn_idx - 2 >= 0:
                    wizard_prev_entry = dialog[turn_idx - 2]
                    wizard_ret_passages = wizard_prev_entry["retrieved_passages"]

                knowledge_dict = {chosen_topic: chosen_topic_passage}
                for ret_passes in [apprentice_ret_passages, wizard_ret_passages]:
                    for passage in ret_passes:
                        for k, v in passage.items():
                            if k not in knowledge_dict.keys():
                                knowledge_dict[k] = v
                            else:
                                tmp_knowledge_list = knowledge_dict[k]
                                for knowledge_elem in v:
                                    if knowledge_elem not in tmp_knowledge_list:
                                        tmp_knowledge_list.append(knowledge_elem)
                                knowledge_dict[k] = tmp_knowledge_list

                wizard_entry = turn
                title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
                selected_knowledge = f'{title} {TOKEN_KNOWLEDGE} {sentence}'

                # Build knowledge list
                knowledge_list = []
                knowledge_label = None
                for title, passage in knowledge_dict.items():
                    for p in passage:
                        cand = f'{title} {TOKEN_KNOWLEDGE} {p}'
                        knowledge_list.append(cand)
                        if cand == selected_knowledge:
                            knowledge_label = len(knowledge_list) - 1

                # Handle no knowledge selected case
                if knowledge_label is None:
                    if selected_knowledge == f"{TOKEN_NOCHOSEN} {TOKEN_KNOWLEDGE} {TOKEN_NOCHOSEN}":
                        knowledge_list = [selected_knowledge] + knowledge_list
                    else:
                        # Skip noisy samples
                        pass
                else:
                    # IMPORTANT: Add no_passages_used only for "ours" variant
                    if add_no_passages:
                        non_select_knowledge = f"{TOKEN_NOCHOSEN} {TOKEN_KNOWLEDGE} {TOKEN_NOCHOSEN}"
                        if non_select_knowledge not in knowledge_list:
                            knowledge_list.append(non_select_knowledge)

                    # Move gold knowledge to first position
                    knowledge_list[0], knowledge_list[knowledge_label] = \
                        knowledge_list[knowledge_label], knowledge_list[0]

                # Limit knowledge if specified
                if max_knowledge > 0 and len(knowledge_list) > max_knowledge:
                    # Keep first (gold) and sample rest
                    import numpy as np
                    keepers = 1 + np.random.choice(
                        len(knowledge_list) - 1,
                        max_knowledge - 1,
                        replace=False
                    )
                    knowledge_list = [knowledge_list[0]] + [
                        knowledge_list[idx] for idx in sorted(keepers)
                    ]

                # Truncate history
                truncated_history = history[-(2 * history_length + 1):].copy()

                sample = {
                    "idx": f"d{dialog_idx}_t{turn_idx}",
                    "history": truncated_history,
                    "knowledges": knowledge_list,
                    "knowledge_label": 0,  # Gold is always at index 0
                    "knowledge_text": selected_knowledge,
                    "response": utterance,
                }
                samples.append(sample)

            history.append(utterance)

    print(f"Processed {len(samples)} samples")

    # Save to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return samples


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SPI datasets")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["origin", "ours", "both"],
        default="both",
        help="Which variant to prepare"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["seen", "unseen", "both"],
        default="both",
        help="Which split to process"
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=1,
        help="Number of history turns to keep"
    )
    parser.add_argument(
        "--max-knowledge",
        type=int,
        default=-1,
        help="Maximum number of knowledge candidates (-1 for no limit)"
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    spi_dir = project_root / "data" / "spi"

    # Define splits
    splits = []
    if args.split in ["seen", "both"]:
        splits.append(("test_random_split.json", "seen_full.json"))
    if args.split in ["unseen", "both"]:
        splits.append(("test_topic_split.json", "unseen_full.json"))

    # Process each variant and split
    variants = []
    if args.variant in ["origin", "both"]:
        variants.append(("origin", False))
    if args.variant in ["ours", "both"]:
        variants.append(("ours", True))

    for variant_name, add_no_passages in variants:
        print(f"\n{'='*60}")
        print(f"Processing variant: {variant_name}")
        print(f"{'='*60}")

        for raw_file, output_file in splits:
            split_name = "seen" if "random" in raw_file else "unseen"
            print(f"\n{split_name.upper()} split:")

            input_path = raw_dir / raw_file
            output_path = spi_dir / "wizard" / variant_name / output_file

            print(f"  Input:  {input_path}")
            print(f"  Output: {output_path}")
            print(f"  Add no_passages_used: {add_no_passages}")

            samples = process_dataset(
                input_path=str(input_path),
                output_path=str(output_path),
                history_length=args.history_length,
                max_knowledge=args.max_knowledge,
                add_no_passages=add_no_passages
            )

            print(f"  ✓ Saved {len(samples)} samples")

    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
