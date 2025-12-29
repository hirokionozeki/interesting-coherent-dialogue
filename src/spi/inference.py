"""
SPI Inference Module

Handles inference for the SPI model with proposed method components.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from methods.breakdown_detector import BreakdownDetector
from methods.confidence_classifier import ConfidenceClassifier
from methods.trivia_reranker import TriviaReranker
from spi.data_loader import SPIDataset
from spi.utils import get_sorted_indices, softmax


class SPIInference:
    """
    Inference engine for SPI with proposed method components.
    """

    def __init__(self, config: Dict):
        """
        Initialize SPI inference engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer
        self._init_model()

        # Initialize proposed method components
        self._init_components()

    def _init_model(self):
        """Initialize model and tokenizer."""
        # Import here to avoid circular dependency
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "baselines" / "SPI"))
        from src.models.bart_modeling import BartForConditionalGenerationWithLangvegin
        from src.utils.utils import add_special_tokens

        model_config = self.config["model"]
        self.model_name = model_config["base_model"]
        self.checkpoint_path = model_config["checkpoint"]

        print(f"Loading model from checkpoint: {self.checkpoint_path}")

        # Load tokenizer from checkpoint (not base model)
        # The checkpoint already contains special tokens, so we load from there
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            use_fast=True,
            add_prefix_space=True
        )

        # Load config from checkpoint
        bart_config = AutoConfig.from_pretrained(self.checkpoint_path)

        # Update config for SPI
        bart_config.latent_size = model_config.get("latent_size", 1024)
        bart_config.sample_latent = model_config.get("sample_latent", False)
        bart_config.kn_selector = model_config.get("kn_selector", "linear")
        bart_config.target_kl = model_config.get("target_kl", 1.0)
        bart_config.attend_latent = model_config.get("attend_latent", False)
        bart_config.attend_latent_w_self = model_config.get("attend_latent_w_self", False)
        bart_config.fuse_z = model_config.get("fuse_z", None)
        bart_config.cls_ratio = model_config.get("cls_ratio", 1.0)
        bart_config.no_kn_decode = model_config.get("no_kn_decode", False)
        bart_config.use_feature = model_config.get("use_feature", "kn")
        bart_config.use_z_for_cls = model_config.get("use_z_for_cls", False)
        bart_config.g_l_steps = model_config.get("g_l_steps", 5)
        bart_config.g_l_step_size = model_config.get("g_l_step_size", 0.3)
        bart_config.verbose = model_config.get("verbose", False)
        bart_config.add_z_mse = model_config.get("add_z_mse", False)
        bart_config.gen_with_noise = model_config.get("gen_with_noise", False)
        bart_config.top_k_kn = model_config.get("top_k_kn", 1)
        bart_config.pseudo_confidence = model_config.get("pseudo_confidence", 1.0)
        bart_config.pseudo_label_only = model_config.get("pseudo_label_only", False)
        bart_config.oracle = model_config.get("oracle", False)
        bart_config.remove_noise = model_config.get("remove_noise", False)
        bart_config.random_choice = model_config.get("random_choice", False)
        bart_config.categorical_prior = model_config.get("categorical_prior", False)

        # Load model from checkpoint
        self.model = BartForConditionalGenerationWithLangvegin.from_pretrained(
            self.checkpoint_path,
            config=bart_config
        )

        # Add special tokens (safe to call - won't add if already present)
        add_special_tokens(self.model, self.tokenizer)

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {self.checkpoint_path}")

    def _init_components(self):
        """Initialize proposed method components based on configuration."""
        variant = self.config.get("variant", "origin")
        methods_config = self.config.get("methods", {})

        # Load precomputed data (SPI-specific: uses pre-computed knowledge scores)
        precomputed_file = methods_config.get("precomputed_file")
        if precomputed_file:
            print(f"Loading precomputed data from: {precomputed_file}")
            with open(precomputed_file, "r") as f:
                precomputed_data = json.load(f)
            self.precomputed_knowledge_scores = precomputed_data.get("all_knowledge_score_list", [])
            self.precomputed_trivia_scores = precomputed_data.get("all_triviascore_list", [])
            self.precomputed_dialog_data = precomputed_data.get("all_dialog_data_list", [])
            print(f"Loaded {len(self.precomputed_knowledge_scores)} precomputed knowledge scores")
            print(f"Loaded {len(self.precomputed_trivia_scores)} precomputed trivia scores")
            print(f"Loaded {len(self.precomputed_dialog_data)} precomputed dialog data")
        else:
            self.precomputed_knowledge_scores = None
            self.precomputed_trivia_scores = None
            self.precomputed_dialog_data = None
            print("Warning: No precomputed data file specified. Using live model inference.")

        # Confidence classifier
        conf_config = methods_config.get("confidence_classification", {})
        self.use_confidence = conf_config.get("enabled", False)
        if self.use_confidence:
            self.confidence_classifier = ConfidenceClassifier(
                conf_sum_prob=conf_config.get("conf_sum_prob", 0.6),
                conf_top1_ratio=conf_config.get("conf_top1_ratio", 0.6),
                unconf_sum_prob=conf_config.get("unconf_sum_prob", 0.4),
                sum_any_kn_prob=conf_config.get("sum_any_kn_prob", 0.6),
                second_first_diff_ratio=conf_config.get("second_first_diff_ratio", 0.5),
                third_second_diff_ratio=conf_config.get("third_second_diff_ratio", 0.5),
            )
        else:
            self.confidence_classifier = None

        # Trivia reranker - not needed if using precomputed data
        trivia_config = methods_config.get("trivia_reranking", {})
        self.use_trivia = trivia_config.get("enabled", False)
        if self.use_trivia and not self.precomputed_trivia_scores:
            # Only load separate trivia file if not using precomputed data
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
            elif scores_file:
                self.trivia_reranker = TriviaReranker(scores_file)
            else:
                self.trivia_reranker = None
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
        print(f"Using precomputed scores: {self.precomputed_knowledge_scores is not None}")

    def _get_knowledge_scores(
        self, batch: Dict[str, torch.Tensor], data_idx: int
    ) -> List[float]:
        """
        Get knowledge selection scores from model or precomputed data.

        Args:
            batch: Input batch
            data_idx: Current data index (for precomputed scores)

        Returns:
            List of knowledge scores (logits)
        """
        # Use precomputed scores if available (matching original SPI implementation)
        if self.precomputed_knowledge_scores is not None:
            if data_idx >= len(self.precomputed_knowledge_scores):
                raise IndexError(
                    f"Data index {data_idx} out of range for precomputed scores "
                    f"(size: {len(self.precomputed_knowledge_scores)})"
                )
            return self.precomputed_knowledge_scores[data_idx]

        # Fall back to live model inference if no precomputed data
        # Reshape inputs
        batch_size = batch["decoder_shapes"][0]
        num_knowledge = batch["decoder_shapes"][1]
        seq_len = batch["decoder_shapes"][2]

        input_ids = batch["input_ids"].view(batch_size, num_knowledge, seq_len)
        attention_mask = batch["attention_mask"].view(batch_size, num_knowledge, seq_len)
        knowledge_mask = batch["decoder_knowledge_mask"].view(batch_size, num_knowledge, seq_len)

        # Forward pass with select_knowledge_index=-1 to get scores
        # Generate decoder_input_ids from labels explicitly to avoid version-dependent issues
        from transformers.models.bart.modeling_bart import shift_tokens_right
        decoder_input_ids = shift_tokens_right(
            batch["labels"],
            self.model.config.pad_token_id,
            self.model.config.decoder_start_token_id
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,  # Explicitly pass 2D decoder_input_ids
                knowledge_mask=knowledge_mask,  # For encoder
                decoder_knowledge_mask=knowledge_mask,  # For decoder
                decoder_shapes=batch["decoder_shapes"],
                labels=batch["labels"],
                select_knowledge_index=torch.tensor(-1),
            )

        # Extract classification logits (knowledge scores)
        classification_logits = outputs.classification_logits[0].tolist()

        return classification_logits

    def _select_knowledge_candidates(
        self, knowledge_scores: List[float], data_idx: int
    ) -> Tuple[List[int], str]:
        """
        Select knowledge candidates using proposed method.

        Args:
            knowledge_scores: Raw knowledge selection scores (pre-aligned for SPI)
            data_idx: Current data index

        Returns:
            Tuple of (selected_indices, confidence_class)
        """
        # Origin baseline: just use top-1
        if not self.use_confidence:
            top_idx = np.argmax(knowledge_scores).item()
            return [top_idx], "Confident"

        # For SPI, knowledge_scores are already pre-aligned (no preprocessing needed)
        knowledge_candidates = list(range(len(knowledge_scores)))

        # Get trivia scores if available (matching original SPI implementation)
        trivia_scores = None
        if self.use_trivia:
            # First check for precomputed trivia scores (from model_pred_all_need_data.json)
            if self.precomputed_trivia_scores is not None:
                if data_idx < len(self.precomputed_trivia_scores):
                    # Precomputed trivia scores are already aligned (including no_passages_used at index 0)
                    trivia_scores = self.precomputed_trivia_scores[data_idx][:len(knowledge_scores)]
            # Fall back to separate trivia reranker file
            elif self.trivia_reranker is not None:
                if self.trivia_reranker.trivia_scores and data_idx < len(self.trivia_reranker.trivia_scores):
                    raw_trivia_scores = self.trivia_reranker.trivia_scores[data_idx]
                    # Prepend 0 for no_passages_used to match knowledge_scores structure
                    trivia_scores = [0.0] + raw_trivia_scores[:len(knowledge_scores)-1]

        # Use confidence classifier with trivia scores
        candidates, confidence_class = self.confidence_classifier.classify(
            knowledge_scores, knowledge_candidates, trivia_scores
        )

        # Note: Trivia reranking is already done inside confidence_classifier if trivia_scores provided
        # Only apply separate reranking if not already done
        if confidence_class != "Confident" and self.use_trivia and trivia_scores is None:
            candidates = self.trivia_reranker.rerank(
                candidates, data_idx, include_no_passage=False
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
        # Reshape inputs
        batch_size = batch["decoder_shapes"][0]
        num_knowledge = batch["decoder_shapes"][1]
        seq_len = batch["decoder_shapes"][2]

        input_ids = batch["input_ids"].view(batch_size, num_knowledge, seq_len)
        attention_mask = batch["attention_mask"].view(batch_size, num_knowledge, seq_len)
        knowledge_mask = batch["decoder_knowledge_mask"].view(batch_size, num_knowledge, seq_len)

        # Create decoder input
        decoder_start_token_id = self.model.config.decoder_start_token_id
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * decoder_start_token_id

        # Generate with selected knowledge
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                knowledge_mask=knowledge_mask,  # For encoder
                decoder_knowledge_mask=knowledge_mask,  # For decoder
                decoder_shapes=batch["decoder_shapes"],
                select_knowledge_index=torch.tensor(knowledge_idx),
                max_length=128,
                num_beams=1,
                do_sample=False,
            )

        # Decode
        decoded_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return decoded_text

    def _extract_dialogue_context(
        self, metadata: Dict, data_idx: int
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Extract dialogue topic and history from precomputed data or metadata.

        Args:
            metadata: Metadata dictionary
            data_idx: Data index

        Returns:
            Tuple of (dialogue_topic, dialogue_history)
            - dialogue_topic: Topic string
            - dialogue_history: List of (speaker, utterance) tuples
        """
        # Use precomputed dialog data if available (matching original SPI implementation)
        if self.precomputed_dialog_data is not None and data_idx < len(self.precomputed_dialog_data):
            dialog_data = self.precomputed_dialog_data[data_idx]
            dialogue_topic = dialog_data.get("topic", "")
            all_utt_list = dialog_data.get("all_utt_list", [])
            all_speaker_list = dialog_data.get("all_speaker_list", [])

            # Build dialogue history with proper speaker labels
            dialogue_history = []
            for utt, speaker in zip(all_utt_list, all_speaker_list):
                dialogue_history.append((speaker, utt))

            return dialogue_topic, dialogue_history

        # Fall back to metadata extraction (for cases without precomputed data)
        dialogue_topic = metadata.get("chosen_topic", "")
        dialogue_history = []

        # History is a list of utterance strings
        # Alternate between speaker1 and speaker2
        history = metadata.get("history", [])
        speaker_idx = 0 if len(history) % 2 == 1 else 1

        for utter in history:
            speaker = "speaker1" if speaker_idx == 0 else "speaker2"
            dialogue_history.append((speaker, utter))
            speaker_idx = (speaker_idx + 1) % 2

        return dialogue_topic, dialogue_history

    def _process_single_example(
        self, batch: Dict[str, torch.Tensor], data_idx: int
    ) -> Dict:
        """
        Process a single example through the full pipeline.

        Args:
            batch: Input batch
            data_idx: Data index

        Returns:
            Result dictionary with all generation details
        """
        result = {
            "data_idx": data_idx,
        }

        metadata = batch["metadata"]

        # Get knowledge scores (precomputed or live inference)
        knowledge_scores = self._get_knowledge_scores(batch, data_idx)
        result["knowledge_score_list"] = knowledge_scores

        # Get number of knowledge candidates
        num_knowledge = len(knowledge_scores)

        # Select knowledge candidates
        kn_select_indices, confidence_class = self._select_knowledge_candidates(
            knowledge_scores, data_idx
        )
        result["kn_select_index_list"] = kn_select_indices
        result["class_label"] = confidence_class

        # Extract dialogue context for breakdown detection
        dialogue_topic, dialogue_history = self._extract_dialogue_context(metadata, data_idx)

        # Generate responses for each candidate
        gen_text_dict = {}
        gpt_reason_dict = {}

        for i, selected_kn_idx in enumerate(kn_select_indices):
            gen_text = self._generate_with_knowledge(batch, selected_kn_idx)

            # Check for dialogue breakdown
            is_breakdown = False
            gpt_reason = "No breakdown detection"

            # Skip breakdown detection for no_passages_used (index 0 in "ours" config)
            # In "ours" configuration with precomputed data, idx=0 is always no_passages_used
            is_no_passages_used = (
                self.precomputed_knowledge_scores is not None and selected_kn_idx == 0
            )

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

            # All candidates were breakdown
            if i == len(kn_select_indices) - 1:
                result["gen_text_dict"] = gen_text_dict
                result["gpt_reason_dict"] = gpt_reason_dict

                # Try with last knowledge (or first if available)
                fallback_idx = num_knowledge - 1 if num_knowledge > 0 else 0
                gen_text = self._generate_with_knowledge(batch, fallback_idx)
                result["final_response"] = gen_text
                result["final_knowledge_idx"] = fallback_idx
                result["final_state"] = "no_passages_used"

        return result

    def run_inference(
        self,
        data_path: str,
        output_dir: str,
        batch_size: int = 1,
    ) -> Dict:
        """
        Run inference on dataset.

        Args:
            data_path: Path to data JSONL file
            output_dir: Output directory for results
            batch_size: Batch size (should be 1 for SPI)

        Returns:
            Evaluation metrics dictionary
        """
        if batch_size != 1:
            raise ValueError("SPI inference currently only supports batch_size=1")

        os.makedirs(output_dir, exist_ok=True)

        # Clear previous results file if it exists
        result_file = Path(output_dir) / "all_results.jsonl"
        if result_file.exists():
            result_file.unlink()

        # Create dataset
        dataset = SPIDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_source_length=512,
            max_target_length=128,
            max_knowledge=-1,  # No limit during inference (same as program/SPI)
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
        true_texts = dataset.responses

        for data_idx, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Process example
            result = self._process_single_example(batch, data_idx)

            # Save result
            results.append(result)
            output_texts.append(result["final_response"])

            # Save incrementally
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
