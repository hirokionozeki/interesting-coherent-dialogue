"""
Compare evaluation results across different methods.

This script loads aggregated results from multiple experiments and
generates comparison tables and statistics.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_aggregated_results(result_path: Path) -> dict[str, Any]:
    """Load aggregated results from JSON file.

    Args:
        result_path: Path to aggregated_results.json

    Returns:
        Dictionary with evaluation statistics
    """
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_comparison_table(
    results_dict: dict[str, dict[str, Any]], eval_type: str = "geval"
) -> pd.DataFrame:
    """Create comparison table for evaluation results.

    Args:
        results_dict: Dictionary mapping method names to their results
        eval_type: Type of evaluation ('geval' or 'meep')

    Returns:
        Pandas DataFrame with comparison statistics
    """
    rows = []

    for method_name, results in results_dict.items():
        if eval_type not in results:
            continue

        eval_results = results[eval_type]

        for metric, stats in eval_results.items():
            rows.append(
                {
                    "Method": method_name,
                    "Metric": metric.capitalize(),
                    "Mean": stats["mean"],
                    "Std": stats["std"],
                    "Min": stats["min"],
                    "Max": stats["max"],
                    "Count": stats["count"],
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def print_comparison_table(df: pd.DataFrame, title: str) -> None:
    """Print formatted comparison table.

    Args:
        df: DataFrame with comparison data
        title: Title for the table
    """
    if df.empty:
        print(f"\n{title}: No data available")
        return

    print(f"\n{title}")
    print("=" * 80)

    # Pivot table for better readability
    pivot = df.pivot_table(
        index="Metric", columns="Method", values="Mean", aggfunc="first"
    )

    # Format to 2 decimal places
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    # Print standard deviations
    print("\nStandard Deviations:")
    pivot_std = df.pivot_table(
        index="Metric", columns="Method", values="Std", aggfunc="first"
    )
    print(pivot_std.to_string(float_format=lambda x: f"{x:.2f}"))


def compute_improvements(
    baseline_results: dict[str, Any], method_results: dict[str, Any], eval_type: str = "geval"
) -> dict[str, float]:
    """Compute percentage improvements over baseline.

    Args:
        baseline_results: Results for baseline method
        method_results: Results for comparison method
        eval_type: Type of evaluation ('geval' or 'meep')

    Returns:
        Dictionary mapping metrics to percentage improvements
    """
    if eval_type not in baseline_results or eval_type not in method_results:
        return {}

    baseline = baseline_results[eval_type]
    method = method_results[eval_type]

    improvements = {}
    for metric in baseline.keys():
        if metric in method:
            baseline_mean = baseline[metric]["mean"]
            method_mean = method[metric]["mean"]
            improvement = ((method_mean - baseline_mean) / baseline_mean) * 100
            improvements[metric] = improvement

    return improvements


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare evaluation results across different methods"
    )
    parser.add_argument(
        "--results",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to aggregated_results.json files",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Names for each method (must match order of --results)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Name of baseline method for computing improvements",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file for comparison table",
    )

    args = parser.parse_args()

    if len(args.results) != len(args.names):
        raise ValueError("Number of results paths must match number of names")

    # Load all results
    results_dict = {}
    for result_path, name in zip(args.results, args.names):
        if not result_path.exists():
            print(f"Warning: {result_path} not found, skipping {name}")
            continue
        results_dict[name] = load_aggregated_results(result_path)

    if not results_dict:
        print("No valid results found")
        return

    # Create and print G-Eval comparison
    geval_df = create_comparison_table(results_dict, eval_type="geval")
    if not geval_df.empty:
        print_comparison_table(geval_df, "G-Eval Comparison")

    # Create and print MEEP comparison
    meep_df = create_comparison_table(results_dict, eval_type="meep")
    if not meep_df.empty:
        print_comparison_table(meep_df, "MEEP Comparison")

    # Compute improvements over baseline if specified
    if args.baseline and args.baseline in results_dict:
        print(f"\n{'=' * 80}")
        print(f"Improvements over {args.baseline}")
        print("=" * 80)

        baseline_results = results_dict[args.baseline]

        for method_name, method_results in results_dict.items():
            if method_name == args.baseline:
                continue

            print(f"\n{method_name}:")

            # G-Eval improvements
            geval_improvements = compute_improvements(
                baseline_results, method_results, eval_type="geval"
            )
            if geval_improvements:
                print("  G-Eval:")
                for metric, improvement in geval_improvements.items():
                    sign = "+" if improvement > 0 else ""
                    print(f"    {metric.capitalize()}: {sign}{improvement:.2f}%")

            # MEEP improvements
            meep_improvements = compute_improvements(
                baseline_results, method_results, eval_type="meep"
            )
            if meep_improvements:
                print("  MEEP:")
                for metric, improvement in meep_improvements.items():
                    sign = "+" if improvement > 0 else ""
                    print(f"    {metric.capitalize()}: {sign}{improvement:.2f}%")

    # Save to CSV if requested
    if args.output:
        combined_df = pd.concat(
            [
                geval_df.assign(Evaluation="G-Eval"),
                meep_df.assign(Evaluation="MEEP"),
            ],
            ignore_index=True,
        )
        combined_df.to_csv(args.output, index=False)
        print(f"\nComparison table saved to {args.output}")


if __name__ == "__main__":
    main()
