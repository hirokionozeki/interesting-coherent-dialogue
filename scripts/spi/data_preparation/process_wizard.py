"""
Process Wizard of Wikipedia data for SPI inference.

Creates two variants:
1. Origin: Baseline (without no_passages_used added)
2. Ours: Proposed method (with no_passages_used added to knowledge candidates)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from spi.data_preparation import main as prepare_main


def main():
    """Run data preparation for both SPI variants."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Wizard of Wikipedia data for SPI inference"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["seen", "unseen", "both"],
        default="both",
        help="Which split to process (default: both)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["origin", "ours", "both"],
        default="both",
        help="Which variant to prepare (default: both - creates both origin and ours)"
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=1,
        help="Number of history turns to keep (default: 1)"
    )
    parser.add_argument(
        "--max-knowledge",
        type=int,
        default=-1,
        help="Maximum number of knowledge candidates (-1 for no limit, default: -1)"
    )

    args = parser.parse_args()

    # Temporarily modify sys.argv to pass arguments to the main function
    original_argv = sys.argv.copy()
    sys.argv = [
        sys.argv[0],
        "--variant", args.variant,
        "--split", args.split,
        "--history-length", str(args.history_length),
        "--max-knowledge", str(args.max_knowledge),
    ]

    try:
        prepare_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
