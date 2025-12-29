"""
G-Eval: LLM-based automatic evaluation using GPT-4o.

This script evaluates dialogue responses using multiple metrics:
- Coherence: Consistency with dialogue context
- Fluency: Grammatical correctness and natural flow
- Informativeness: Amount of information provided
- Interestingness: Degree of engagement and intrigue

Note: Engagingness is evaluated separately using MEEP.

Based on the G-Eval framework: https://github.com/nlpyang/geval
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import openai
from tqdm import tqdm


METRICS = ["coherence", "fluency", "informativeness", "interestingness"]

SYSTEM_PROMPT = "You are an expert evaluator of dialogue systems."


def load_prompt(prompt_path: Path) -> str:
    """Load evaluation prompt from file.

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Prompt template string
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def format_dialogue_context(dialogue_history: list[dict[str, str]]) -> str:
    """Format dialogue history into a readable string.

    Args:
        dialogue_history: List of dialogue turns with 'speaker' and 'text' keys

    Returns:
        Formatted dialogue context string
    """
    lines = []
    for turn in dialogue_history:
        speaker = turn.get("speaker", "Speaker")
        text = turn.get("text", "")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def evaluate_single_instance(
    client: openai.OpenAI,
    prompt_template: str,
    dialogue_context: str,
    response: str,
    model: str = "gpt-4o",
    n_samples: int = 20,
    max_retries: int = 5,
) -> list[str]:
    """Evaluate a single dialogue response.

    Args:
        client: OpenAI client instance
        prompt_template: Evaluation prompt template
        dialogue_context: Formatted dialogue history
        response: Generated response to evaluate
        model: OpenAI model to use
        n_samples: Number of samples to generate
        max_retries: Maximum number of retries on error

    Returns:
        List of evaluation responses from the model
    """
    # Format prompt
    prompt = prompt_template.replace("{{Document}}", dialogue_context).replace(
        "{{Response}}", response
    )

    # Retry loop
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,
                max_tokens=5,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=n_samples,
            )

            # Extract responses
            return [choice.message.content for choice in completion.choices]

        except openai.RateLimitError as e:
            print(f"Rate limit error: {e}. Retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

    return []


def run_evaluation(
    input_path: Path,
    output_path: Path,
    prompt_dir: Path,
    api_key: str,
    metrics: list[str],
    model: str = "gpt-4o",
    n_samples: int = 20,
) -> None:
    """Run G-Eval evaluation on dialogue data.

    Args:
        input_path: Path to input JSON file with dialogue data
        output_path: Path to save evaluation results
        prompt_dir: Directory containing prompt files
        api_key: OpenAI API key
        metrics: List of metrics to evaluate
        model: OpenAI model to use
        n_samples: Number of samples per evaluation
    """
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # Load prompts for each metric
    prompts = {}
    for metric in metrics:
        prompt_path = prompt_dir / f"{metric}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompts[metric] = load_prompt(prompt_path)

    # Load input data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} instances from {input_path}")
    print(f"Evaluating metrics: {', '.join(metrics)}")
    print(f"Using model: {model}")
    print(f"Samples per evaluation: {n_samples}")

    # Load existing results if available (for resuming)
    results = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing results")

    # Process each instance
    start_idx = len(results)
    for idx, instance in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
        result = {
            "id": instance.get("id", idx),
            "dialogue_context": instance.get("dialogue_context", ""),
            "response": instance.get("response", ""),
            "evaluations": {},
        }

        # Format dialogue context
        if isinstance(result["dialogue_context"], list):
            dialogue_str = format_dialogue_context(result["dialogue_context"])
        else:
            dialogue_str = result["dialogue_context"]

        # Evaluate each metric
        for metric in metrics:
            try:
                scores = evaluate_single_instance(
                    client=client,
                    prompt_template=prompts[metric],
                    dialogue_context=dialogue_str,
                    response=result["response"],
                    model=model,
                    n_samples=n_samples,
                )
                result["evaluations"][metric] = scores

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"\nFailed to evaluate instance {idx} for metric {metric}: {e}")
                result["evaluations"][metric] = []

        results.append(result)

        # Save intermediate results every 10 instances
        if (idx + 1) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Save final results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete. Results saved to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="G-Eval: LLM-based evaluation for dialogue generation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file with dialogue data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--prompt_dir",
        type=Path,
        default=Path("evaluation/prompts/geval"),
        help="Directory containing prompt files",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=METRICS,
        help="Metrics to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of samples per evaluation",
    )

    args = parser.parse_args()

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    run_evaluation(
        input_path=args.input,
        output_path=args.output,
        prompt_dir=args.prompt_dir,
        api_key=args.api_key,
        metrics=args.metrics,
        model=args.model,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
