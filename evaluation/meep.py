"""
MEEP: Engagingness evaluation metric for dialogue systems.

This script evaluates dialogue responses on a continuous scale from 0 to 100,
focusing on engagingness defined by:
- Variety of response according to context
- Likelihood of encouraging response
- Likelihood of encouraging quality response
- Interestingness
- Specificity
- Likelihood of creating sense of belonging

Based on the MEEP framework: https://github.com/PortNLP/MEEP
"""

import argparse
import json
import time
from pathlib import Path

import openai
from tqdm import tqdm


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
    max_retries: int = 5,
) -> str:
    """Evaluate a single dialogue response for engagingness.

    Args:
        client: OpenAI client instance
        prompt_template: Evaluation prompt template
        dialogue_context: Formatted dialogue history
        response: Generated response to evaluate
        model: OpenAI model to use
        max_retries: Maximum number of retries on error

    Returns:
        Evaluation score as string
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
                temperature=0.0,  # Deterministic output
                max_tokens=5,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=1,  # Single sample
            )

            # Extract response
            return completion.choices[0].message.content

        except openai.RateLimitError as e:
            print(f"Rate limit error: {e}. Retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

    return ""


def run_evaluation(
    input_path: Path,
    output_path: Path,
    prompt_path: Path,
    api_key: str,
    model: str = "gpt-4o",
) -> None:
    """Run MEEP evaluation on dialogue data.

    Args:
        input_path: Path to input JSON file with dialogue data
        output_path: Path to save evaluation results
        prompt_path: Path to the MEEP prompt file
        api_key: OpenAI API key
        model: OpenAI model to use
    """
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # Load prompt
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt_template = load_prompt(prompt_path)

    # Load input data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} instances from {input_path}")
    print(f"Using model: {model}")

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
            "score": None,
        }

        # Format dialogue context
        if isinstance(result["dialogue_context"], list):
            dialogue_str = format_dialogue_context(result["dialogue_context"])
        else:
            dialogue_str = result["dialogue_context"]

        # Evaluate
        try:
            score = evaluate_single_instance(
                client=client,
                prompt_template=prompt_template,
                dialogue_context=dialogue_str,
                response=result["response"],
                model=model,
            )
            result["score"] = score

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"\nFailed to evaluate instance {idx}: {e}")
            result["score"] = None

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
        description="MEEP: Engagingness evaluation for dialogue generation"
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
        "--prompt",
        type=Path,
        default=Path("evaluation/prompts/meep/engagingness.txt"),
        help="Path to MEEP prompt file",
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
        default="gpt-4o",
        help="OpenAI model to use",
    )

    args = parser.parse_args()

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    run_evaluation(
        input_path=args.input,
        output_path=args.output,
        prompt_path=args.prompt,
        api_key=args.api_key,
        model=args.model,
    )


if __name__ == "__main__":
    main()
