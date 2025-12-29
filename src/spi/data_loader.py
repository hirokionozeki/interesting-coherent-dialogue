"""
SPI Data Loader

Handles data loading and preprocessing for the SPI model.
Adapted for inference with proposed method components.
"""

import json
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SPIDataset(Dataset):
    """
    Dataset class for SPI model inference.

    Processes Wizard of Wikipedia data and prepares input sequences for BART-based
    knowledge-grounded dialogue generation.
    """

    def __init__(
        self,
        data_path,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 128,
        max_knowledge: int = -1,
    ):
        """
        Initialize SPI dataset.

        Args:
            data_path: Path to preprocessed JSONL data file, or list of paths for multiple files
            tokenizer: Tokenizer for encoding text
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
            max_knowledge: Maximum number of knowledge candidates (-1 for no limit)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_knowledge = max_knowledge

        # Support both single file and multiple files
        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = data_path

        # Load data from all files
        self.data = []
        for path in data_paths:
            with open(path, "r") as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))

        # Extract responses for evaluation
        self.responses = [example["response"] for example in self.data]

    def __len__(self) -> int:
        return len(self.data)

    def _encode_knowledge(self, knowledge: str) -> List[int]:
        """
        Encode a single knowledge sentence.

        Args:
            knowledge: Knowledge sentence text

        Returns:
            Token IDs for the knowledge
        """
        return self.tokenizer.encode(
            knowledge,
            add_special_tokens=False,
            truncation=True,
            max_length=128,
        )

    def _encode_history(self, history: List[str]) -> List[int]:
        """
        Encode dialogue history.

        Args:
            history: List of dialogue turns (utterance strings)

        Returns:
            Token IDs for the history
        """
        # History is a list of utterances
        # Alternate between speaker1 and speaker2 tokens
        SPEAKERS = ["<speaker1>", "<speaker2>"]
        history_text = ""

        # Determine starting speaker based on history length
        # Odd length = speaker1 starts, even length = speaker2 starts
        speaker_idx = 0 if len(history) % 2 == 1 else 1

        for utter in history:
            history_text += utter + SPEAKERS[speaker_idx]
            speaker_idx = (speaker_idx + 1) % 2

        return self.tokenizer.encode(
            history_text.strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=256,
        )

    def __getitem__(self, index: int) -> Dict:
        """
        Get a single example for inference.

        Args:
            index: Index of the example

        Returns:
            Dictionary containing processed inputs and metadata
        """
        example = self.data[index]

        # Extract fields
        knowledges = example.get("knowledges", [])
        if self.max_knowledge > 0:
            knowledges = knowledges[:self.max_knowledge]
        history = example.get("history", [])
        response = example.get("response", "")
        chosen_topic = example.get("chosen_topic", "")

        # Encode history
        history_ids = self._encode_history(history)

        # Encode each knowledge candidate
        knowledge_ids_list = []
        for knowledge in knowledges:
            kn_ids = self._encode_knowledge(knowledge)
            knowledge_ids_list.append(kn_ids)

        # Encode response for evaluation
        response_ids = self.tokenizer.encode(
            response,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_target_length,
        )

        return {
            "index": index,
            "history_ids": history_ids,
            "knowledge_ids_list": knowledge_ids_list,
            "response_ids": response_ids,
            "chosen_topic": chosen_topic,
            "history": history,
            "knowledges": knowledges,
            "response": response,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate function for DataLoader.

        For inference, batch size should be 1, so this is simplified.

        Args:
            batch: List of examples

        Returns:
            Batched dictionary
        """
        if len(batch) != 1:
            raise ValueError("SPI inference currently only supports batch_size=1")

        item = batch[0]

        # Prepare input_ids: [num_knowledge, max_length]
        # Each row is history + one knowledge candidate
        input_ids_list = []
        attention_mask_list = []
        knowledge_mask_list = []

        history_ids = item["history_ids"]

        for kn_ids in item["knowledge_ids_list"]:
            # Combine: <s> history </s> </s> knowledge </s>
            combined = (
                [self.tokenizer.bos_token_id]
                + history_ids
                + [self.tokenizer.eos_token_id, self.tokenizer.eos_token_id]
                + kn_ids
                + [self.tokenizer.eos_token_id]
            )

            # Truncate if too long
            if len(combined) > self.max_source_length:
                combined = combined[:self.max_source_length]

            # Create masks
            attention_mask = [1] * len(combined)

            # Knowledge mask: 1 for knowledge tokens, 0 for history
            history_len = len(history_ids) + 3  # <s> + history + </s> </s>
            kn_mask = [0] * history_len + [1] * (len(combined) - history_len)

            # Pad to max_source_length
            padding_length = self.max_source_length - len(combined)
            combined = combined + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            kn_mask = kn_mask + [0] * padding_length

            input_ids_list.append(combined)
            attention_mask_list.append(attention_mask)
            knowledge_mask_list.append(kn_mask)

        # Convert to tensors: shape [1, num_knowledge, max_length]
        input_ids = torch.tensor([input_ids_list], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask_list], dtype=torch.long)
        knowledge_mask = torch.tensor([knowledge_mask_list], dtype=torch.long)

        # Labels for evaluation
        labels = torch.tensor([item["response_ids"]], dtype=torch.long)

        # Decoder shapes: tuple of (batch_size, num_knowledge, max_length)
        decoder_shapes = (
            input_ids.shape[0],
            input_ids.shape[1],
            input_ids.shape[2]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_knowledge_mask": knowledge_mask,  # Use correct parameter name
            "labels": labels,
            "decoder_shapes": decoder_shapes,  # Pass as tuple, not tensor
            "metadata": {
                "index": item["index"],
                "chosen_topic": item["chosen_topic"],
                "history": item["history"],
                "knowledges": item["knowledges"],
                "response": item["response"],
            }
        }
