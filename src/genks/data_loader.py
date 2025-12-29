"""
GenKS Data Loader

Handles data loading and preprocessing for the GenKS model.
Adapted from the original GenKS implementation.
"""

import copy
from collections import OrderedDict
from typing import Dict, List, Optional

import nltk
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils.evaluation import f1_score


class GenKSDataset(Dataset):
    """
    Dataset class for GenKS model.

    Processes Wizard of Wikipedia data and prepares input sequences for BART-based
    knowledge-grounded dialogue generation.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        psg_filter: Optional[List[List[int]]] = None,
        context_len: int = 256,
        sent_len: int = 64,
        max_length: int = 1024,
        psg_num: int = 1,
        max_id: int = 128,
        test: bool = False,
        use_oracle: bool = False,
        shuffle_id: bool = False,
        add_label: bool = True,
        add_response: bool = True,
        add_label_to_prefix: Optional[bool] = None,
        add_hyperlink: bool = False,
        use_pred_label: Optional[List[str]] = None,
        dialogue_first: bool = True,
        knowledge_response: float = 0.0,
        second_id: bool = False,
        drop_null: bool = True,
        max_num_of_know: Optional[int] = None,
    ):
        """
        Initialize GenKS dataset.

        Args:
            data: List of dialogue examples
            tokenizer: Tokenizer for encoding text
            psg_filter: Pre-computed passage filter for knowledge selection
            context_len: Maximum context length
            sent_len: Maximum sentence length
            max_length: Maximum total sequence length
            psg_num: Number of passages to include
            max_id: Maximum knowledge ID
            test: Whether this is test mode
            use_oracle: Whether to use oracle knowledge
            shuffle_id: Whether to shuffle knowledge IDs
            add_label: Whether to add knowledge label to output
            add_response: Whether to add response to output
            add_label_to_prefix: Whether to add label to input prefix
            add_hyperlink: Whether to add hyperlinks to dialogue
            use_pred_label: Pre-computed predicted labels
            dialogue_first: Whether to put dialogue before knowledge in input
            knowledge_response: Probability of using knowledge as response
            second_id: Whether to include second-best knowledge ID
            drop_null: Whether to drop examples with no valid knowledge
            max_num_of_know: Maximum number of knowledge sentences
        """
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.sent_len = sent_len
        self.max_length = max_length
        self.test = test
        self.psg_filter = psg_filter
        self.psg_num = psg_num
        self.use_oracle = use_oracle
        self.shuffle_id = shuffle_id
        self.max_id = max_id
        self.add_response = add_response
        self.add_label = add_label
        self.response = [example["labels"][0] for example in self.data]
        self.add_label_to_prefix = add_label_to_prefix
        self.add_hyperlink = add_hyperlink
        self.use_pred_label = use_pred_label
        self.dialogue_first = dialogue_first
        self.knowledge_response = knowledge_response
        self.second_id = second_id
        self.drop_null = drop_null
        self.max_num_of_know = max_num_of_know

    def __len__(self) -> int:
        return len(self.data)

    def _build_knowledge_sequence(
        self, example: Dict, id_map: List[int]
    ) -> tuple[List[int], str, Dict[str, int], str]:
        """
        Build knowledge passage sequence.

        Args:
            example: Single dialogue example
            id_map: Mapping of knowledge IDs

        Returns:
            Tuple of (sequence, label, sentence_to_id, second_best)
        """
        knowledge = example["knowledge"]

        # Apply passage filter
        if self.psg_filter is not None:
            positive = [k for k in knowledge if k == example["title"]]
            titles = positive + [k for k in knowledge if k != example["title"]]
            titles = [titles[pid] for pid in self.psg_filter[self.current_index]][: self.psg_num]

            if (
                self.use_oracle
                and example["title"] != "no_passages_used"
                and example["title"] in knowledge
                and example["title"] not in titles
            ):
                titles = [example["title"]] + titles[:-1]

            new_knowledge = OrderedDict()
            for k in titles:
                new_knowledge[k] = knowledge[k]
            knowledge = new_knowledge

        # Build passage sequence
        sequence = []
        sent_id = 0
        label = f"<s{id_map[0]}>"
        sentence_to_id = {}

        sequence += self.tokenizer.encode(
            "\nPassage information.\n", add_special_tokens=False
        )

        # Add "no_passages_used" option
        sentence = "no_passages_used"
        sent_id += 1
        sequence += self.tokenizer.encode(
            f"<s{id_map[sent_id]}>{sentence}<s{id_map[sent_id]}>\n",
            add_special_tokens=False,
        )
        sentence_to_id[sentence] = sent_id
        if sentence == example["checked_sentence"]:
            label = f"<s{id_map[sent_id]}>"

        # Track second-best knowledge
        second_best = ""
        second_best_score = 0

        # Add knowledge passages
        for pid, (title, passage) in enumerate(knowledge.items()):
            sequence += self.tokenizer.encode(
                f"Passage {pid + 1}, Title: {title}\n", add_special_tokens=False
            )

            for sentence in passage:
                if len(sequence) > self.max_length:
                    break

                sent_id += 1
                sequence += self.tokenizer.encode(
                    f"<s{id_map[sent_id]}>{sentence}",
                    truncation=True,
                    max_length=self.sent_len,
                    add_special_tokens=False,
                )
                sequence += self.tokenizer.encode(
                    f"<s{id_map[sent_id]}>\n", add_special_tokens=False
                )
                sentence_to_id[sentence] = sent_id

                if sentence == example["checked_sentence"]:
                    label = f"<s{id_map[sent_id]}>"
                elif self.second_id:
                    score = f1_score(
                        sentence + example["checked_sentence"], [example["labels"][0]]
                    )
                    if score > second_best_score:
                        second_best = f"<s{id_map[sent_id]}>"
                        second_best_score = score

                if self.max_num_of_know is not None and sent_id >= self.max_num_of_know:
                    break

            if self.max_num_of_know is not None and sent_id >= self.max_num_of_know:
                break

        if self.second_id:
            label = label + second_best

        return sequence, label, sentence_to_id, second_best

    def _build_dialogue_sequence(
        self, example: Dict, sentence_to_id: Dict[str, int], id_map: List[int]
    ) -> List[int]:
        """
        Build dialogue history sequence.

        Args:
            example: Single dialogue example
            sentence_to_id: Mapping of sentences to knowledge IDs
            id_map: Mapping of knowledge IDs

        Returns:
            Encoded dialogue sequence
        """
        role = {
            "0_Wizard": "User1: ",
            "1_Apprentice": "User2: ",
            "0_Apprentice": "User2: ",
            "1_Wizard": "User1: ",
            0: "User1: ",
            1: "User2: ",
            "user1": "User1: ",
            "user2": "User2: ",
        }

        context = ""
        for turn in example["context"]:
            speaker = role.get(turn["speaker"], turn["speaker"])
            text = turn["text"]
            kk = ""

            if self.add_hyperlink and "title" in turn:
                kk = f"[{turn['title']}]"
                if turn["checked_sentence"] in sentence_to_id:
                    kk += f"<s{id_map[sentence_to_id[turn['checked_sentence']]]}>"
                kk += " "

            context += f"{speaker}{kk}{text}\n"

        topic = "Chosen topic: " + example["chosen_topic"] + "\n"
        sequence = []
        sequence += self.tokenizer.encode(
            "\nDialogue history.\n", add_special_tokens=False
        )
        sequence += self.tokenizer.encode(topic, add_special_tokens=False)
        sequence += self.tokenizer.encode(context, add_special_tokens=False)[
            -self.context_len :
        ]
        sequence += self.tokenizer.encode(
            "Predict the next knowledge sentence id and response of User1.\n",
            add_special_tokens=False,
        )

        if self.add_label_to_prefix:
            if isinstance(self.add_label_to_prefix, list):
                pred_label = self.add_label_to_prefix[self.current_index]
                sequence += self.tokenizer.encode(
                    f"Selected knowledge = {pred_label}\n", add_special_tokens=False
                )
            else:
                label = self._get_current_label()
                sequence += self.tokenizer.encode(
                    f"Selected knowledge = {label}\n", add_special_tokens=False
                )

        return sequence

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.

        Args:
            index: Index of the example

        Returns:
            Tuple of (input_ids, labels)
        """
        self.current_index = index
        example = self.data[index]

        # Handle ID mapping
        id_map = [i for i in range(2, self.max_id)]
        if self.shuffle_id:
            np.random.shuffle(id_map)
        id_map = [0, 1] + id_map

        # Build knowledge sequence
        passage_sequence, label, sentence_to_id, second_best = self._build_knowledge_sequence(
            example, id_map
        )

        # Drop null examples during training
        if (
            self.drop_null
            and not self.test
            and example["title"] != "no_passages_used"
        ):
            if (
                example["title"] not in example["knowledge"]
                or example["checked_sentence"]
                not in example["knowledge"][example["title"]]
            ):
                return self[np.random.randint(len(self))]

        # Build dialogue sequence
        dialogue_sequence = self._build_dialogue_sequence(example, sentence_to_id, id_map)

        # Combine sequences
        sequence = []
        passage_sequence = passage_sequence[: self.max_length - len(dialogue_sequence)]

        if self.dialogue_first:
            sequence += dialogue_sequence
            sequence += passage_sequence
        else:
            sequence += passage_sequence
            sequence += dialogue_sequence

        # Build target
        target = []
        if self.add_label:
            if isinstance(self.use_pred_label, list):
                target.append(self.use_pred_label[index][0])
            else:
                target.append(f"{label}")

        if self.add_response:
            if (
                self.knowledge_response
                and example["checked_sentence"] != "no_passages_used"
                and np.random.random() < self.knowledge_response
            ):
                target.append(f"{example['checked_sentence']}")
            else:
                target.append(f"{example['labels'][0]}")

        target = " ".join(target)

        # Encode target (BART style)
        labels = self.tokenizer.encode(
            target, truncation=True, max_length=self.context_len, add_special_tokens=True
        )

        return torch.tensor(sequence), torch.tensor(labels)

    def collate_fn(self, data: List[tuple]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.

        Args:
            data: List of (input_ids, labels) tuples

        Returns:
            Batched dictionary with input_ids, attention_mask, and labels
        """
        padding_value = self.tokenizer.pad_token_id
        input_ids, labels = zip(*data)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(padding_value),
            "labels": labels,
        }
