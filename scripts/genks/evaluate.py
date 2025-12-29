"""
GenKS Evaluation Script

Evaluates GenKS inference results using the same metrics as the original baseline.
"""

import argparse
import json
import sys
from pathlib import Path

# Add baselines/GenKS/utils to path for evaluation functions
baseline_utils_path = Path(__file__).parent.parent / "baselines" / "GenKS" / "utils"
sys.path.insert(0, str(baseline_utils_path))

from evaluation import eval_all, eval_acc


def lower(text):
    """
    Lowercase and tokenize text.

    Args:
        text: Input text (string or list)

    Returns:
        Lowercased and tokenized text
    """
    import nltk

    if isinstance(text, str):
        text = text.strip().lower()
        text = ' '.join(nltk.word_tokenize(text))
        return text.strip()
    return [lower(item) for item in text]


def load_predictions(output_file: str) -> list:
    """
    Load predictions from output text file.

    Args:
        output_file: Path to output_text.txt

    Returns:
        List of prediction strings
    """
    predictions = []
    with open(output_file, 'r') as f:
        for line in f:
            predictions.append(line.strip())
    return predictions


def load_references(data_file: str) -> tuple:
    """
    Load reference data.

    Args:
        data_file: Path to data JSON file

    Returns:
        Tuple of (reference_texts, raw_data)
    """
    with open(data_file, 'r') as f:
        data = json.load(f)

    references = [example['labels'][0] for example in data]
    return references, data


def evaluate(predictions: list, references: list, raw_data: list, use_cache: bool = False):
    """
    Run all evaluation metrics.

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        raw_data: Raw data for knowledge evaluation
        use_cache: Whether to use cached evaluation data

    Returns:
        Dictionary with all evaluation metrics
    """
    print(f"Evaluating {len(predictions)} predictions...")

    # Response generation metrics (F1, ROUGE, BLEU, METEOR, Distinct)
    print("\nCalculating response generation metrics...")
    response_metrics = eval_all(lower(predictions), lower(references))

    # Knowledge metrics (KF1, EntF1, ACC)
    print("Calculating knowledge metrics...")
    cache = None  # Can load pre-computed cache if available
    knowledge_metrics = eval_acc(predictions, raw_data, cache=cache)

    # Combine all metrics
    all_metrics = {**response_metrics, **knowledge_metrics}

    return all_metrics


def format_results(metrics: dict) -> str:
    """
    Format evaluation results as human-readable text.

    Args:
        metrics: Dictionary of evaluation metrics

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("GenKS Evaluation Results")
    lines.append("=" * 60)
    lines.append("")

    # Response Generation Metrics
    lines.append("Response Generation Metrics:")
    lines.append("-" * 60)
    for key in ['F1', 'ROUGE_1_F1', 'ROUGE_2_F1', 'ROUGE_L_F1',
                'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR',
                'ReDist2', 'ReDist3']:
        if key in metrics:
            lines.append(f"  {key:20s}: {metrics[key]:6.2f}")
    lines.append("")

    # Knowledge Metrics
    lines.append("Knowledge Metrics:")
    lines.append("-" * 60)
    for key in ['KF1', 'EntF1', 'ACC']:
        if key in metrics:
            lines.append(f"  {key:20s}: {metrics[key]:6.2f}")
    lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def save_results(metrics: dict, output_dir: str):
    """
    Save evaluation results to files.

    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Output directory
    """
    output_path = Path(output_dir)

    # Save human-readable text file
    text_file = output_path / "evaluation_results.txt"
    with open(text_file, 'w') as f:
        f.write(format_results(metrics))
    print(f"\nResults saved to: {text_file}")

    # Save JSON file
    json_file = output_path / "evaluation_results.json"
    with open(json_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to: {json_file}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate GenKS inference results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing output_text.txt",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to original data JSON file",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached evaluation data (if available)",
    )
    args = parser.parse_args()

    # Load predictions
    output_file = Path(args.output_dir) / "output_text.txt"
    if not output_file.exists():
        print(f"Error: {output_file} not found")
        sys.exit(1)

    print(f"Loading predictions from: {output_file}")
    predictions = load_predictions(output_file)

    # Load references
    if not Path(args.data_file).exists():
        print(f"Error: {args.data_file} not found")
        sys.exit(1)

    print(f"Loading references from: {args.data_file}")
    references, raw_data = load_references(args.data_file)

    # Check length match
    if len(predictions) != len(references):
        print(f"Warning: Number of predictions ({len(predictions)}) != "
              f"number of references ({len(references)})")
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
        raw_data = raw_data[:min_len]

    # Run evaluation
    metrics = evaluate(predictions, references, raw_data, args.use_cache)

    # Display results
    print("\n" + format_results(metrics))

    # Save results
    save_results(metrics, args.output_dir)


if __name__ == "__main__":
    main()
