"""
GenKS Inference Script

Runs inference for GenKS model with proposed method components.
"""

import argparse
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.config import load_config
from genks.inference import GenKSInference


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="Run inference for knowledge-grounded dialogue")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after inference",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override output directory if specified
    if args.output_dir:
        config["output"]["dir"] = args.output_dir

    # Run GenKS inference
    print("Running GenKS inference")
    data_path = run_genks_inference(config, args.batch_size)

    # Run evaluation if requested
    if args.evaluate:
        run_evaluation(config["output"]["dir"], data_path)


def run_genks_inference(config: dict, batch_size: int) -> str:
    """
    Run GenKS inference.

    Args:
        config: Configuration dictionary
        batch_size: Batch size for inference

    Returns:
        Path to data file (last split processed)
    """
    # Get data paths
    data_config = config["data"]
    dataset_path = data_config["dataset_path"]

    # Support both single and multiple splits
    splits = data_config.get("splits")
    split = data_config.get("split")

    if splits:
        # Multiple splits mode: run inference for each split sequentially
        splits_list = splits if isinstance(splits, list) else [splits]
    elif split:
        # Single split mode
        splits_list = [split]
    else:
        raise ValueError("Either 'split' or 'splits' must be specified in config")

    last_data_path = None

    for current_split in splits_list:
        print(f"\n{'='*60}")
        print(f"Processing split: {current_split}")
        print(f"{'='*60}\n")

        # Create split-specific config by deep copying and replacing placeholders
        import copy
        split_config = copy.deepcopy(config)

        # Replace {split} placeholders in all config values
        split_config["data"]["psg_filter"] = data_config["psg_filter"].replace("{split}", current_split)

        methods_config = split_config.get("methods", {})
        if methods_config.get("trivia_reranking", {}).get("enabled"):
            trivia_file = methods_config["trivia_reranking"]["scores_file"]
            methods_config["trivia_reranking"]["scores_file"] = trivia_file.replace("{split}", current_split)

        if methods_config.get("breakdown_detection", {}).get("enabled"):
            dbd_file = methods_config["breakdown_detection"]["cache_file"]
            methods_config["breakdown_detection"]["cache_file"] = dbd_file.replace("{split}", current_split)

        # Initialize inference engine for this split
        inference_engine = GenKSInference(split_config)

        # Construct paths for current split
        data_path = f"{dataset_path}/{current_split}.json"
        psg_filter_path = split_config["data"]["psg_filter"]
        output_dir = config["output"]["dir"] + f"/{current_split}"

        # Run inference
        print(f"Data: {data_path}")
        print(f"Passage filter: {psg_filter_path}")
        print(f"Output: {output_dir}")

        results = inference_engine.run_inference(
            data_path=data_path,
            psg_filter_path=psg_filter_path,
            output_dir=output_dir,
            batch_size=batch_size,
        )

        print(f"\nInference complete for {current_split}!")
        print(f"Processed {results['num_examples']} examples")
        print(f"Results saved to: {results['result_file']}")
        print(f"Output texts saved to: {results['output_file']}")

        last_data_path = data_path

    print(f"\n{'='*60}")
    print(f"All splits processed successfully!")
    print(f"{'='*60}\n")

    return data_path


def run_evaluation(output_dir: str, data_path: str):
    """
    Run evaluation on inference results.

    Args:
        output_dir: Directory containing inference results
        data_path: Path to original data file
    """
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)

    # Import evaluation functions directly (avoid subprocess fork issues)
    baseline_utils_path = Path(__file__).parent.parent.parent / "baselines" / "GenKS" / "utils"
    if str(baseline_utils_path) not in sys.path:
        sys.path.insert(0, str(baseline_utils_path))

    try:
        import json
        import nltk
        from evaluation import eval_all, eval_acc

        def lower(text):
            """Lowercase and tokenize text."""
            if isinstance(text, str):
                text = text.strip().lower()
                text = ' '.join(nltk.word_tokenize(text))
                return text.strip()
            return [lower(item) for item in text]

        # Load predictions
        output_file = Path(output_dir) / "output_text.txt"
        print(f"Loading predictions from: {output_file}")
        predictions = []
        with open(output_file, 'r') as f:
            for line in f:
                predictions.append(line.strip())

        # Load references
        print(f"Loading references from: {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
        references = [example['labels'][0] for example in data]

        # Check length
        if len(predictions) != len(references):
            print(f"Warning: predictions ({len(predictions)}) != references ({len(references)})")
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
            data = data[:min_len]

        # Run evaluation
        print(f"Evaluating {len(predictions)} predictions...")
        print("\nCalculating response generation metrics...")
        response_metrics = eval_all(lower(predictions), lower(references))

        print("Calculating knowledge metrics...")
        knowledge_metrics = eval_acc(predictions, data, cache=None)

        # Combine metrics
        all_metrics = {**response_metrics, **knowledge_metrics}

        # Format and display results
        lines = []
        lines.append("=" * 60)
        lines.append("GenKS Evaluation Results")
        lines.append("=" * 60)
        lines.append("")
        lines.append("Response Generation Metrics:")
        lines.append("-" * 60)
        for key in ['F1', 'ROUGE_1_F1', 'ROUGE_2_F1', 'ROUGE_L_F1',
                    'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR',
                    'ReDist2', 'ReDist3']:
            if key in all_metrics:
                lines.append(f"  {key:20s}: {all_metrics[key]:6.2f}")
        lines.append("")
        lines.append("Knowledge Metrics:")
        lines.append("-" * 60)
        for key in ['KF1', 'EntF1', 'ACC']:
            if key in all_metrics:
                lines.append(f"  {key:20s}: {all_metrics[key]:6.2f}")
        lines.append("")
        lines.append("=" * 60)

        result_text = "\n".join(lines)
        print("\n" + result_text)

        # Save results
        output_path = Path(output_dir)
        text_file = output_path / "evaluation_results.txt"
        with open(text_file, 'w') as f:
            f.write(result_text)
        print(f"\nResults saved to: {text_file}")

        json_file = output_path / "evaluation_results.json"
        with open(json_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Results saved to: {json_file}")

    except Exception as e:
        print(f"Evaluation failed with error:")
        print(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
