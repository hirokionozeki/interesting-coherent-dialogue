#!/bin/bash
# Run LLM-based evaluation for dialogue generation experiments
# Usage: ./scripts/run_llm_evaluation.sh <method> <baseline_name> <input_json> <output_dir> <api_key>

set -e

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <method> <baseline_name> <input_json> <output_dir> <api_key> [model]"
    echo ""
    echo "Arguments:"
    echo "  method        : genks or spi"
    echo "  baseline_name : origin, ours, etc."
    echo "  input_json    : Path to input JSON file with dialogue data"
    echo "  output_dir    : Directory to save evaluation results"
    echo "  api_key       : OpenAI API key"
    echo "  model         : (Optional) OpenAI model (default: gpt-4-0613)"
    echo ""
    echo "Example:"
    echo "  $0 genks ours results/genks/ours_outputs.json results/genks/ours/evaluation \$OPENAI_API_KEY"
    exit 1
fi

METHOD=$1
BASELINE_NAME=$2
INPUT_JSON=$3
OUTPUT_DIR=$4
API_KEY=$5
MODEL=${6:-gpt-4-0613}

echo "=========================================="
echo "LLM-based Evaluation"
echo "=========================================="
echo "Method: $METHOD"
echo "Baseline: $BASELINE_NAME"
echo "Input: $INPUT_JSON"
echo "Output: $OUTPUT_DIR"
echo "Model: $MODEL"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input file not found: $INPUT_JSON"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Running evaluation..."
python evaluation/run_evaluation.py \
    --input "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --api_key "$API_KEY" \
    --model "$MODEL" \
    --evaluation_types all \
    --geval_samples 20

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files:"
echo "  - $OUTPUT_DIR/geval_results.json"
echo "  - $OUTPUT_DIR/meep_results.json"
echo "  - $OUTPUT_DIR/aggregated_results.json"
