"""
GenKS Inference Module

Handles inference for the GenKS model with proposed method components.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from methods.breakdown_detector import BreakdownDetector
from methods.confidence_classifier import ConfidenceClassifier
from methods.trivia_reranker import TriviaReranker
from genks.data_loader import GenKSDataset
from genks.utils import split_id, lower


class GenKSInference:
    """
    Inference engine for GenKS with proposed method components.
    """

    def __init__(self, config: Dict):
        """
        Initialize GenKS inference engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Knowledge token range: <s0> to <s127> (token IDs: 50265-50392)
        self.kn_token_start = 50265
        self.kn_token_end = 50392

        # Global state for knowledge selection control
        self.global_state = [0]  # 0: all tokens allowed, 1: knowledge selection restricted
        self.global_knowledge_idx = [0]  # Selected knowledge token ID

        # Initialize model and tokenizer
        self._init_model()

        # Initialize proposed method components
        self._init_components()

    def _init_model(self):
        """Initialize model and tokenizer."""
        model_config = self.config["model"]
        self.model_name = model_config["base_model"]
        self.checkpoint_path = model_config["checkpoint"]

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Add special tokens
        special_tokens = (
            [f"<s{i}>" for i in range(128)]
            + ["<s>", "</s>", "<pad>", "<positive>", "<negative>"]
        )
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Load checkpoint
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint: {self.checkpoint_path}")
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        else:
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

    def _init_components(self):
        """Initialize proposed method components based on configuration."""
        variant = self.config.get("variant", "origin")
        methods_config = self.config.get("methods", {})

        # Confidence classifier
        conf_config = methods_config.get("confidence_classification", {})
        self.use_confidence = conf_config.get("enabled", False)
        if self.use_confidence:
            self.confidence_classifier = ConfidenceClassifier(
                conf_sum_prob=conf_config.get("conf_sum_prob", 0.8),
                conf_top1_ratio=conf_config.get("conf_top1_ratio", 0.8),
                unconf_sum_prob=conf_config.get("unconf_sum_prob", 0.65),
            )
        else:
            self.confidence_classifier = None

        # Trivia reranker
        trivia_config = methods_config.get("trivia_reranking", {})
        self.use_trivia = trivia_config.get("enabled", False)
        if self.use_trivia:
            scores_file = trivia_config.get("scores_file")

            # Check if on-demand scoring is enabled
            on_demand_config = trivia_config.get("on_demand", {})
            enable_on_demand = on_demand_config.get("enabled", False)

            if enable_on_demand:
                # Prepare on-demand configuration
                on_demand_params = {
                    "api_key": on_demand_config.get("api_key"),
                    "model": on_demand_config.get("model", "gpt-4o-2024-08-06"),
                    "num_samples": on_demand_config.get("num_samples", 5),
                    "temperature": on_demand_config.get("temperature", 1.0),
                    "cache_scores": on_demand_config.get("cache_scores", True),
                }
                self.trivia_reranker = TriviaReranker(
                    scores_file=scores_file,
                    enable_on_demand=True,
                    on_demand_config=on_demand_params
                )
            else:
                self.trivia_reranker = TriviaReranker(scores_file)
        else:
            self.trivia_reranker = None

        # Breakdown detector
        dbd_config = methods_config.get("breakdown_detection", {})
        self.use_breakdown = dbd_config.get("enabled", False)
        if self.use_breakdown:
            cache_file = dbd_config.get("cache_file")
            model_name = dbd_config.get("model", "gpt-4o-2024-08-06")
            self.breakdown_detector = BreakdownDetector(
                model=model_name, cache_file=cache_file
            )
        else:
            self.breakdown_detector = None

        print(f"Variant: {variant}")
        print(f"Confidence classification: {self.use_confidence}")
        print(f"Trivia reranking: {self.use_trivia}")
        print(f"Breakdown detection: {self.use_breakdown}")

    def _prefix_allowed_tokens_fn(self, batch_id: int, input_ids: torch.Tensor) -> List[int]:
        """
        Control which tokens are allowed during generation.

        Args:
            batch_id: Batch index
            input_ids: Current input token IDs

        Returns:
            List of allowed token IDs
        """
        if self.global_state[0] == 0:
            # Allow all tokens (except special ones)
            return list(range(50395))
        elif self.global_state[0] == 1:
            # Restrict to specific knowledge token
            knowledge_idx = self.global_knowledge_idx[0]
            return list(range(50265)) + [knowledge_idx] + list(range(50393, 50395))
        else:
            raise ValueError(f"Invalid global_state: {self.global_state[0]}")

    def _get_knowledge_candidates(self, input_ids: torch.Tensor) -> List[int]:
        """
        Extract knowledge candidate IDs from input.

        Args:
            input_ids: Input token IDs

        Returns:
            List of knowledge candidate indices (relative to first candidate)
        """
        kn_token_ids = [
            i for i in input_ids[0].tolist()
            if self.kn_token_start <= i <= self.kn_token_end
        ]
        kn_candidates = sorted(list(set(kn_token_ids)))
        # Convert to relative indices (0, 1, 2, ...)
        kn_indices = [token_id - self.kn_token_start - 1 for token_id in kn_candidates]
        return kn_indices

    def _get_knowledge_scores(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[List[float], int]:
        """
        Get knowledge selection scores from model.

        Args:
            batch: Input batch

        Returns:
            Tuple of (knowledge_scores, selected_token_id)
        """
        self.global_state[0] = 0

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=50,
                return_dict_in_generate=True,
                output_scores=True,
                num_beams=1,
                do_sample=False,
            )

        # Find knowledge token position
        token_list = outputs.sequences[0].tolist()
        kn_tokens = [i for i in token_list if i >= self.kn_token_start]

        if len(kn_tokens) == 0:
            raise ValueError("No knowledge selection token in generated text")
        if len(kn_tokens) >= 2:
            print("Warning: Multiple knowledge tokens found, using first one")

        # Get scores for knowledge token
        kn_token_id = kn_tokens[0]
        kn_idx_pos = token_list.index(kn_token_id) - 1
        elem_score_list = outputs.scores[kn_idx_pos][0].tolist()

        # Extract knowledge scores
        knowledge_scores = elem_score_list[self.kn_token_start : self.kn_token_end + 1]

        return knowledge_scores, kn_token_id

    def _select_knowledge_candidates(
        self, knowledge_scores: List[float], knowledge_candidates: List[int], data_idx: int
    ) -> Tuple[List[int], str]:
        """
        Select knowledge candidates using proposed method.

        Args:
            knowledge_scores: Raw knowledge selection scores (all 128 tokens)
            knowledge_candidates: Available knowledge indices
            data_idx: Current data index

        Returns:
            Tuple of (selected_indices, confidence_class)
        """
        # Origin baseline: just use top-1
        if not self.use_confidence:
            top_idx = np.argmax(knowledge_scores).item()
            return [top_idx], "Confident"

        # Pre-process scores for GenKS (slice to match actual candidates)
        # GenKS extracts scores for all 128 special tokens, need to slice:
        # [0] = s0 (unused), [1] = s1 (no_passages_used), [2+] = actual knowledge
        # Slice [1:len(knowledge_candidates)+1] to get: no_passages_used + actual knowledge
        processed_scores = knowledge_scores[1 : len(knowledge_candidates) + 1]

        # Get trivia scores if available
        trivia_scores = None
        if self.use_trivia and self.trivia_reranker is not None:
            # Get trivia scores for this data instance (without no_passages_used)
            raw_trivia_scores = self.trivia_reranker.trivia_scores[data_idx] if self.trivia_reranker.trivia_scores else None
            if raw_trivia_scores is not None:
                # Prepend 0 for no_passages_used to match processed_scores structure
                trivia_scores = [0.0] + raw_trivia_scores[:len(knowledge_candidates)]

        # Use confidence classifier with pre-processed scores
        candidates, confidence_class = self.confidence_classifier.classify(
            processed_scores, knowledge_candidates, trivia_scores
        )

        # Note: Trivia reranking is already done inside confidence_classifier if trivia_scores provided
        # Only apply separate reranking if not already done
        if confidence_class != "Confident" and self.use_trivia and trivia_scores is None:
            candidates = self.trivia_reranker.rerank(
                candidates, data_idx, include_no_passage=True
            )

        return candidates, confidence_class

    def _generate_with_knowledge(
        self, batch: Dict[str, torch.Tensor], knowledge_idx: int
    ) -> str:
        """
        Generate response with specific knowledge.

        Args:
            batch: Input batch
            knowledge_idx: Knowledge index to use

        Returns:
            Generated response text
        """
        # Set knowledge control
        self.global_state[0] = 1
        self.global_knowledge_idx[0] = knowledge_idx + self.kn_token_start + 1

        with torch.no_grad():
            output = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
                num_beams=1,
                do_sample=False,
                prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn,
            )

        output_id, output_text = split_id(
            self.tokenizer.batch_decode(output, skip_special_tokens=True)
        )

        # Verify knowledge control worked
        tmp_kn_id_text = output_id[0].replace("<s", "").replace(">", "")
        if tmp_kn_id_text and knowledge_idx + 1 != int(tmp_kn_id_text):
            print(
                f"Warning: Knowledge control mismatch: expected {knowledge_idx+1}, got {tmp_kn_id_text}"
            )

        return output_text[0]

    def _extract_dialogue_context(self, data_dict: Dict) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Extract dialogue topic and history from data dictionary.

        Args:
            data_dict: Original data dictionary

        Returns:
            Tuple of (dialogue_topic, dialogue_history)
            - dialogue_topic: Topic string
            - dialogue_history: List of (speaker, utterance) tuples
        """
        dialogue_topic = data_dict.get("chosen_topic", "")
        dialogue_history = []

        for turn in data_dict.get("context", []):
            speaker = turn.get("speaker", "")
            utterance = turn.get("text", "")
            dialogue_history.append((speaker, utterance))

        return dialogue_topic, dialogue_history

    def _process_single_example(
        self, batch: Dict[str, torch.Tensor], data_idx: int, data_dict: Dict
    ) -> Dict:
        """
        Process a single example through the full pipeline.

        Args:
            batch: Input batch
            data_idx: Data index
            data_dict: Original data dictionary

        Returns:
            Result dictionary with all generation details
        """
        result = {
            "data_idx": data_idx,
        }

        # Get knowledge candidates
        knowledge_candidates = self._get_knowledge_candidates(batch["input_ids"])
        result["knowledge_candidate_list"] = knowledge_candidates

        # Get knowledge scores
        knowledge_scores, vanilla_kn_token = self._get_knowledge_scores(batch)
        result["knowledge_score_list"] = knowledge_scores
        result["vanilla_kn_select_idx"] = vanilla_kn_token

        # Select knowledge candidates
        kn_select_indices, confidence_class = self._select_knowledge_candidates(
            knowledge_scores, knowledge_candidates, data_idx
        )
        result["kn_select_index_list"] = kn_select_indices
        result["class_label"] = confidence_class

        # Extract dialogue context for breakdown detection
        dialogue_topic, dialogue_history = self._extract_dialogue_context(data_dict)

        # Generate responses for each candidate
        gen_text_dict = {}
        gpt_reason_dict = {}

        for i, selected_kn_idx in enumerate(kn_select_indices):
            gen_text = self._generate_with_knowledge(batch, selected_kn_idx)

            # Check for dialogue breakdown
            is_breakdown = False
            gpt_reason = "No breakdown detection"

            # Skip breakdown detection for no_passages_used (index 0 in GenKS)
            # In GenKS, idx=0 always corresponds to no_passages_used (token s1)
            is_no_passages_used = selected_kn_idx == 0

            if self.use_breakdown and not is_no_passages_used:
                is_breakdown, gpt_reason = self.breakdown_detector.detect(
                    response=gen_text,
                    dialogue_topic=dialogue_topic,
                    dialogue_history=dialogue_history,
                    data_idx=data_idx,
                    knowledge_idx=selected_kn_idx,
                )
            elif is_no_passages_used:
                gpt_reason = "no_passages_used - skip breakdown detection"

            gen_text_dict[selected_kn_idx] = gen_text
            gpt_reason_dict[gen_text] = gpt_reason

            # If not breakdown, use this response
            if not is_breakdown:
                result["gen_text_dict"] = gen_text_dict
                result["gpt_reason_dict"] = gpt_reason_dict
                result["final_response"] = gen_text
                result["final_knowledge_idx"] = selected_kn_idx
                result["final_state"] = "not_breakdown"
                break

            # All candidates were breakdown, use no_passages_used
            if i == len(kn_select_indices) - 1:
                result["gen_text_dict"] = gen_text_dict
                result["gpt_reason_dict"] = gpt_reason_dict

                # Generate with no knowledge
                gen_text = self._generate_with_knowledge(batch, 0)
                result["final_response"] = gen_text
                result["final_knowledge_idx"] = 0
                result["final_state"] = "no_passages_used"

        return result

    def run_inference(
        self,
        data_path: str,
        psg_filter_path: str,
        output_dir: str,
        batch_size: int = 1,
    ) -> Dict:
        """
        Run inference on dataset.

        Args:
            data_path: Path to data JSON file
            psg_filter_path: Path to passage filter JSON file
            output_dir: Output directory for results
            batch_size: Batch size (should be 1 for sequential processing)

        Returns:
            Evaluation metrics dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        # Clear previous results file if it exists
        result_file = Path(output_dir) / "all_results.jsonl"
        if result_file.exists():
            result_file.unlink()

        # Load data (support both single file and multiple files)
        if isinstance(data_path, list):
            print(f"Loading data from {len(data_path)} files")
            data = []
            for path in data_path:
                print(f"  - {path}")
                with open(path) as f:
                    data.extend(json.load(f))
        else:
            print(f"Loading data from {data_path}")
            with open(data_path) as f:
                data = json.load(f)

        print(f"Loading passage filter from {psg_filter_path}")
        with open(psg_filter_path) as f:
            psg_filter = json.load(f)

        # Create dataset
        dataset = GenKSDataset(
            data=data,
            tokenizer=self.tokenizer,
            psg_filter=psg_filter,
            context_len=256,
            sent_len=64,
            max_length=1024,
            psg_num=1,
            shuffle_id=False,
            max_id=128,
            test=True,
            use_oracle=False,
            add_label=True,
            add_response=True,
            add_hyperlink=True,
            add_label_to_prefix=False,
            use_pred_label=None,
            dialogue_first=True,
            second_id=False,
            max_num_of_know=None,
        )

        data_loader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Run inference
        print(f"Running inference on {len(dataset)} examples")
        results = []
        output_texts = []
        true_texts = dataset.response

        for data_idx, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Process example
            result = self._process_single_example(batch, data_idx, data[data_idx])

            # Save result
            results.append(result)
            output_texts.append(result["final_response"])

            # Save incrementally (in case of interruption)
            result_file = Path(output_dir) / "all_results.jsonl"
            with open(result_file, "a") as f:
                f.write(json.dumps(result) + "\n")

        # Save final outputs
        output_file = Path(output_dir) / "output_text.txt"
        with open(output_file, "w") as f:
            for text in output_texts:
                f.write(text + "\n")

        print(f"Results saved to {output_dir}")

        return {
            "num_examples": len(results),
            "output_file": str(output_file),
            "result_file": str(result_file),
        }
