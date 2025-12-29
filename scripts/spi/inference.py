"""
SPI Inference Script

Runs inference for SPI model with proposed method components.
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
from spi.inference import SPIInference


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="Run inference for SPI model")
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
        help="Batch size for inference (default: 1, only 1 supported currently)",
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

    # Validate batch size
    if args.batch_size != 1:
        print("Warning: SPI inference currently only supports batch_size=1")
        args.batch_size = 1

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override output directory if specified
    if args.output_dir:
        config["output"]["dir"] = args.output_dir

    # Run SPI inference
    print("Running SPI inference")
    data_path = run_spi_inference(config, args.batch_size)

    # Run evaluation if requested
    if args.evaluate:
        run_evaluation(config["output"]["dir"], data_path)


def run_spi_inference(config: dict, batch_size: int) -> str:
    """
    Run SPI inference.

    Args:
        config: Configuration dictionary
        batch_size: Batch size for inference

    Returns:
        Path to data file
    """
    # Initialize inference engine
    inference_engine = SPIInference(config)

    # Get data paths
    data_config = config["data"]
    data_path = data_config["dataset_path"]
    output_dir = config["output"]["dir"]

    # Run inference
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")

    results = inference_engine.run_inference(
        data_path=data_path,
        output_dir=output_dir,
        batch_size=batch_size,
    )

    print(f"\nInference complete!")
    print(f"Processed {results['num_examples']} examples")
    print(f"Results saved to: {results['result_file']}")
    print(f"Output texts saved to: {results['output_file']}")

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

    # Import evaluation functions
    baseline_utils_path = Path(__file__).parent.parent.parent / "baselines" / "SPI" / "src" / "modules"
    if str(baseline_utils_path) not in sys.path:
        sys.path.insert(0, str(baseline_utils_path))

    try:
        import json
        import numpy as np
        from evaluator import DialogEvaluator
        from transformers import AutoTokenizer

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

        # Create evaluator for response generation and knowledge selection
        evaluator = DialogEvaluator(
            metric_name="f1&rouge&bleu&acc",
            tokenizer=tokenizer,
            eval_selection=True,
            eval_gen=True,
        )

        # Load predictions (response generation)
        output_file = Path(output_dir) / "output_text.txt"
        print(f"Loading predictions from: {output_file}")
        predictions = []
        with open(output_file, 'r') as f:
            for line in f:
                predictions.append(line.strip())

        # Load inference results (knowledge selection)
        results_file = Path(output_dir) / "all_results.jsonl"
        print(f"Loading inference results from: {results_file}")
        knowledge_scores = []
        knowledge_labels = []
        with open(results_file, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line.strip())
                    knowledge_scores.append(result['knowledge_score_list'])
                    # Ground truth is always at index 0 in our data
                    knowledge_labels.append(0)

        # Load references
        print(f"Loading references from: {data_path}")
        references = []
        with open(data_path, 'r') as f:
            for line in f:
                data_item = json.loads(line.strip())
                references.append(data_item['response'])

        # Check length consistency
        if len(predictions) != len(references):
            print(f"Warning: predictions ({len(predictions)}) != references ({len(references)})")
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
            knowledge_scores = knowledge_scores[:min_len]
            knowledge_labels = knowledge_labels[:min_len]

        # Run response generation evaluation
        print(f"\nEvaluating {len(predictions)} predictions...")
        print("Calculating response generation metrics...")
        gen_metrics = evaluator.compute(predictions, references, prefix="test_")

        # Run knowledge selection evaluation
        print("Calculating knowledge selection metrics...")
        # Convert knowledge scores to numpy arrays with proper shapes
        max_len = max(len(scores) for scores in knowledge_scores)
        # Pad scores to same length
        padded_scores = []
        for scores in knowledge_scores:
            padded = scores + [-float('inf')] * (max_len - len(scores))
            padded_scores.append(padded)

        knowledge_scores_np = np.array(padded_scores)
        knowledge_labels_np = np.array(knowledge_labels)

        cls_metrics = evaluator.compute_cls(
            knowledge_scores_np,
            knowledge_labels_np,
            prefix="test_"
        )

        # Combine all metrics
        all_metrics = {**gen_metrics, **cls_metrics}

        # Format and display results
        lines = []
        lines.append("=" * 60)
        lines.append("SPI Evaluation Results")
        lines.append("=" * 60)
        lines.append("")
        lines.append("Response Generation Metrics:")
        lines.append("-" * 60)

        # Define metric order for better readability
        gen_metric_keys = ['test_f1', 'test_rouge1', 'test_rouge2',
                          'test_bleu1', 'test_bleu2', 'test_bleu3', 'test_bleu4']
        for key in gen_metric_keys:
            if key in all_metrics:
                lines.append(f"  {key:25s}: {all_metrics[key]:6.2f}")

        lines.append("")
        lines.append("Knowledge Selection Metrics:")
        lines.append("-" * 60)

        # Note: "accurracy" is a typo in the original evaluator.py
        cls_metric_keys = ['test_accurracy']
        for key in cls_metric_keys:
            if key in all_metrics:
                lines.append(f"  {key:25s}: {all_metrics[key]:6.2f}")

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
        print("\nNote: Evaluation is optional. Inference results are still available.")


if __name__ == "__main__":
    main()
