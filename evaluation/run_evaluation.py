"""
Unified evaluation runner for dialogue generation experiments.

This script provides a convenient interface to run both G-Eval and MEEP
evaluations, aggregate results, and compute statistics.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def parse_geval_score(score_str: str) -> float | None:
    """Parse G-Eval score from string response.

    Args:
        score_str: Raw score string from model (e.g., "4", "3.5", "5/5")

    Returns:
        Parsed score as float, or None if parsing failed
    """
    if not score_str:
        return None

    # Clean up string
    score_str = score_str.strip()

    # Try to extract number
    try:
        # Handle "X/5" format
        if "/" in score_str:
            score_str = score_str.split("/")[0]
        # Handle parentheses or other characters
        score_str = "".join(c for c in score_str if c.isdigit() or c == ".")
        return float(score_str)
    except (ValueError, AttributeError):
        return None


def parse_meep_score(score_str: str) -> float | None:
    """Parse MEEP score from string response.

    Args:
        score_str: Raw score string from model (e.g., "75", "80.5")

    Returns:
        Parsed score as float, or None if parsing failed
    """
    if not score_str:
        return None

    # Clean up string
    score_str = score_str.strip()

    # Try to extract number
    try:
        score_str = "".join(c for c in score_str if c.isdigit() or c == ".")
        return float(score_str)
    except (ValueError, AttributeError):
        return None


def aggregate_geval_results(results_path: Path) -> dict[str, Any]:
    """Aggregate G-Eval results and compute statistics.

    Args:
        results_path: Path to G-Eval results JSON file

    Returns:
        Dictionary with aggregated statistics per metric
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    metrics_stats = {}

    # Extract unique metrics
    all_metrics = set()
    for result in results:
        all_metrics.update(result.get("evaluations", {}).keys())

    # Compute statistics for each metric
    for metric in all_metrics:
        scores = []
        for result in results:
            raw_scores = result.get("evaluations", {}).get(metric, [])
            parsed_scores = [
                s for s in (parse_geval_score(rs) for rs in raw_scores) if s is not None
            ]
            if parsed_scores:
                # Average across multiple samples for this instance
                scores.append(np.mean(parsed_scores))

        if scores:
            metrics_stats[metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "count": len(scores),
            }

    return metrics_stats


def aggregate_meep_results(results_path: Path) -> dict[str, Any]:
    """Aggregate MEEP results and compute statistics.

    Args:
        results_path: Path to MEEP results JSON file

    Returns:
        Dictionary with aggregated statistics
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    scores = []
    for result in results:
        raw_score = result.get("score")
        parsed_score = parse_meep_score(raw_score)
        if parsed_score is not None:
            scores.append(parsed_score)

    if scores:
        return {
            "engagingness": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "count": len(scores),
            }
        }

    return {}


def run_geval(
    input_path: Path,
    output_path: Path,
    api_key: str,
    model: str,
    metrics: list[str],
    n_samples: int,
) -> None:
    """Run G-Eval evaluation.

    Args:
        input_path: Input data path
        output_path: Output results path
        api_key: OpenAI API key
        model: Model name
        metrics: List of metrics to evaluate
        n_samples: Number of samples per evaluation
    """
    cmd = [
        "python",
        "evaluation/g_eval.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--api_key",
        api_key,
        "--model",
        model,
        "--n_samples",
        str(n_samples),
        "--metrics",
    ] + metrics

    print(f"Running G-Eval with metrics: {', '.join(metrics)}")
    subprocess.run(cmd, check=True)


def run_meep(
    input_path: Path,
    output_path: Path,
    api_key: str,
    model: str,
) -> None:
    """Run MEEP evaluation.

    Args:
        input_path: Input data path
        output_path: Output results path
        api_key: OpenAI API key
        model: Model name
    """
    cmd = [
        "python",
        "evaluation/meep.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--api_key",
        api_key,
        "--model",
        model,
    ]

    print("Running MEEP evaluation")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LLM-based evaluation for dialogue generation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file with dialogue data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-0613",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--evaluation_types",
        nargs="+",
        choices=["geval", "meep", "all"],
        default=["all"],
        help="Which evaluations to run",
    )
    parser.add_argument(
        "--geval_metrics",
        nargs="+",
        default=["coherence", "fluency", "informativeness", "interestingness"],
        help="G-Eval metrics to evaluate",
    )
    parser.add_argument(
        "--geval_samples",
        type=int,
        default=20,
        help="Number of samples for G-Eval",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Only aggregate existing results without running evaluation",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_types = args.evaluation_types
    if "all" in eval_types:
        eval_types = ["geval", "meep"]

    # Run evaluations
    if not args.aggregate_only:
        if "geval" in eval_types:
            geval_output = args.output_dir / "geval_results.json"
            run_geval(
                input_path=args.input,
                output_path=geval_output,
                api_key=args.api_key,
                model=args.model,
                metrics=args.geval_metrics,
                n_samples=args.geval_samples,
            )

        if "meep" in eval_types:
            meep_output = args.output_dir / "meep_results.json"
            run_meep(
                input_path=args.input,
                output_path=meep_output,
                api_key=args.api_key,
                model=args.model,
            )

    # Aggregate results
    print("\n" + "=" * 50)
    print("Aggregating Results")
    print("=" * 50)

    aggregated = {}

    if "geval" in eval_types:
        geval_output = args.output_dir / "geval_results.json"
        if geval_output.exists():
            print("\nG-Eval Results:")
            geval_stats = aggregate_geval_results(geval_output)
            aggregated["geval"] = geval_stats
            for metric, stats in geval_stats.items():
                print(f"  {metric.capitalize()}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        else:
            print("\nG-Eval results not found")

    if "meep" in eval_types:
        meep_output = args.output_dir / "meep_results.json"
        if meep_output.exists():
            print("\nMEEP Results:")
            meep_stats = aggregate_meep_results(meep_output)
            aggregated["meep"] = meep_stats
            for metric, stats in meep_stats.items():
                print(f"  {metric.capitalize()}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        else:
            print("\nMEEP results not found")

    # Save aggregated results
    if aggregated:
        agg_path = args.output_dir / "aggregated_results.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        print(f"\nAggregated results saved to {agg_path}")


if __name__ == "__main__":
    main()
