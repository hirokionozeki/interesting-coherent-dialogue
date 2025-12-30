# Interesting Coherent Dialogue

This repository contains the official implementation of the paper "[Enhancing Coherence and Interestingness in Knowledge-Grounded Dialogue Generation](https://aclanthology.org/2025.inlg-main.1/)" (INLG 2025).

For citation information, please refer to the [Citation](#citation) section below.

## Overview

This project enhances knowledge-grounded dialogue systems through three key components:

1. **Confidence Classification**: Categorizes knowledge selection confidence (Confident/Undecided/Unclear)
2. **Trivia Score Reranking**: Reorders knowledge candidates by interestingness when confidence is low
3. **Dialogue Breakdown Detection**: Filters incoherent responses using GPT-4o

**Key Feature**: Our method is **training-free** and can be applied to various knowledge-grounded dialogue models without additional training. The implementation demonstrates this by supporting two state-of-the-art baseline methods (GenKS and SPI), which require separate environments due to incompatible dependencies.

## Data and Checkpoints Setup

To reproduce experiments, download required data and model checkpoints (~2.7GB total).

### Automated Setup (Recommended)

Run the setup script:

```bash
bash scripts/setup_data.sh
```

The script will:
- Download data and GenKS checkpoints from [Google Drive](https://drive.google.com/file/d/17KFNM2K17uEQAmZehgfZAho3tweTIiXM/view?usp=sharing) (~2.2GB)
- Guide you through SPI checkpoint setup from the [official repository](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yxucb_connect_ust_hk/Euyhj33uFtdLnBcWe4bHBukB4rjbXSaoWRbG2PZ6Mcdt9Q?e=WCs3ap)
- Verify all files are in place

### What's Included

The download includes:
- **Raw data** (252MB): Wizard of Wikipedia test sets
- **Reference data** (4.9MB): Pre-computed trivia scores
- **Precomputed data** (130MB): Cached results (GPT-4o API calls, knowledge scores)
- **GenKS checkpoints** (2.0GB): BART-large model and knowledge rankers

**Note**: SPI checkpoint (548MB) is downloaded separately from the original SPI repository to ensure proper attribution.

## Installation

### Clone the Repository

This repository uses Git submodules for baseline implementations. Clone with submodules:

```bash
git clone --recursive https://github.com/hirokionozeki/interesting-coherent-dialogue.git
cd interesting-coherent-dialogue
```

Or if you've already cloned the repository:

```bash
git submodule update --init --recursive
```

### About Baseline Submodules

This project includes two baseline methods as Git submodules:
- **GenKS** ([original repository](https://github.com/sunnweiwei/GenKS))
- **SPI** (modified fork from [original repository](https://github.com/HKUST-KnowComp/SPI))

**Note on SPI**: We use a modified fork of SPI that includes necessary changes for integration with this project. The modifications are minimal and only affect model loading and inference compatibility. The original SPI implementation remains unchanged in the upstream repository.

## Quick Start

### 1. Environment Setup

**GenKS Environment:**
```bash
uv venv .venv-genks --python 3.9
source .venv-genks/bin/activate
pip install -r requirements_genks.txt

# Download required data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

**SPI Environment:**
```bash
uv venv .venv-spi --python 3.9
source .venv-spi/bin/activate
pip install -r requirements_spi.txt
```

**API Key Setup:**

Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

This is required for Dialogue Breakdown Detection and optional LLM-based evaluations.

### 2. Data Preparation

**GenKS:**
```bash
source .venv-genks/bin/activate
python scripts/genks/data_preparation/process_wizard.py --split both
```

**SPI:**
```bash
source .venv-spi/bin/activate
python scripts/spi/data_preparation/process_wizard.py --split both
```

This processes raw Wizard of Wikipedia data into the required format. The SPI script creates two variants:
- **Origin**: Baseline (standard knowledge candidates)
- **Ours**: Proposed method (includes "no knowledge used" option)

### 3. Run Inference

**GenKS:**
```bash
source .venv-genks/bin/activate

# Baseline vs Proposed
python scripts/genks/inference.py --config configs/genks/origin.yaml --evaluate
python scripts/genks/inference.py --config configs/genks/ours.yaml --evaluate
```

**SPI:**
```bash
source .venv-spi/bin/activate

# Baseline vs Proposed
python scripts/spi/inference.py --config configs/spi/origin.yaml --evaluate
python scripts/spi/inference.py --config configs/spi/ours.yaml --evaluate
```

The `--evaluate` flag automatically runs evaluation after inference.

### 4. View Results

```bash
cat results/genks/test_mini/evaluation_results.txt
cat results/genks/test_mini/evaluation_results.json
```

## Configurations

**GenKS** (`configs/genks/`):

| Config | Description |
|--------|-------------|
| `origin.yaml` | Baseline GenKS |
| `ours.yaml` | Full proposed method (all components) |

**SPI** (`configs/spi/`):

| Config | Description |
|--------|-------------|
| `origin.yaml` | Baseline SPI |
| `ours.yaml` | Full proposed method |

## Evaluation Metrics

### Response Generation
- **F1**: Token-level F1 score
- **ROUGE-1/2/L**: Summary quality metrics
- **BLEU-1/2/3/4**: N-gram overlap scores
- **METEOR**: Semantic similarity
- **ReDist2/3**: Response diversity

### Knowledge Selection
- **KF1**: Knowledge F1 (overlap with selected knowledge)
- **EntF1**: Entity F1 (named entity overlap)
- **ACC**: Knowledge selection accuracy

### LLM-based Evaluation (Optional)

Additional evaluation using GPT-4o:

**G-Eval** - Evaluates coherence, fluency, and informativeness:
```bash
python evaluation/g_eval.py \
  --input results/genks/ours/all_results.jsonl \
  --output results/genks/ours/geval.json \
  --api_key YOUR_API_KEY
```

**MEEP** - Evaluates engagingness:
```bash
python evaluation/meep.py \
  --input results/genks/ours/all_results.jsonl \
  --output results/genks/ours/meep.json \
  --api_key YOUR_API_KEY
```

See `evaluation/` directory for detailed usage with `--help`.

## Project Structure

```
interesting-coherent-dialogue/
├── src/
│   ├── genks/              # GenKS implementation
│   ├── spi/                # SPI implementation
│   ├── methods/            # Proposed method components
│   │   ├── confidence_classifier.py
│   │   ├── trivia_reranker.py
│   │   ├── trivia_scorer.py
│   │   └── breakdown_detector.py
│   └── utils/              # Shared utilities
│
├── scripts/
│   ├── genks/              # GenKS scripts
│   │   ├── data_preparation/
│   │   ├── inference.py
│   │   └── evaluate.py
│   └── spi/                # SPI scripts
│       ├── data_preparation/
│       └── inference.py
│
├── configs/                # YAML configuration files
│   ├── genks/
│   ├── spi/
│   └── prompts/            # Prompt templates for LLM-based components
│
├── data/
│   ├── raw/                # Raw Wizard of Wikipedia data
│   ├── genks/wizard/       # Processed data for GenKS
│   ├── spi/                # Processed data for SPI (origin/ours)
│   └── reference/          # Reference trivia scores (trivia_scores.json)
│
├── baselines/              # Original baseline code (Git submodules)
│   ├── GenKS/
│   └── SPI/
│
├── results/                # Inference and evaluation results
├── checkpoints/            # Pre-trained model checkpoints
├── evaluation/             # Evaluation scripts (G-Eval, MEEP)
│
├── requirements_genks.txt  # GenKS dependencies
└── requirements_spi.txt    # SPI dependencies
```

## Baseline Methods

This project builds upon two state-of-the-art knowledge-grounded dialogue models:

- **GenKS**: [Generative Knowledge Selection for Knowledge-Grounded Dialogues](https://arxiv.org/abs/2304.04836) (EACL 2023)
  - Sun, W., Ren, P., & Ren, Z.
  - Repository: https://github.com/sunnweiwei/GenKS

- **SPI**: [Diverse and Faithful Knowledge-Grounded Dialogue Generation via Sequential Posterior Inference](https://arxiv.org/abs/2306.01153) (ICML 2023)
  - Xu, Y., Kong, D., Xu, D., Ji, Z., Pang, B., Fung, P., & Wu, Y. N.
  - Original repository: https://github.com/HKUST-KnowComp/SPI
  - Modified fork (used in this project): https://github.com/hirokionozeki/SPI

**Note**: Minor modifications were made to the SPI implementation to enable integration with our proposed methods. These changes are isolated to the forked repository and do not affect the original implementation.

For complete licensing and attribution information, see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Requirements

- Python >= 3.9
- PyTorch ~= 1.13.1
- Transformers ~= 4.30.2
- OpenAI API key (for Dialogue Breakdown Detection)
- See `requirements_genks.txt` and `requirements_spi.txt` for complete dependencies

## License

This project is intended for academic research purposes. When using this code, please ensure compliance with all third-party licenses and provide proper citations as specified in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Citation

```bibtex
@inproceedings{onozeki-inaba-2025-enhancing,
    title = "Enhancing Coherence and Interestingness in Knowledge-Grounded Dialogue Generation",
    author = "Onozeki, Hiroki and Inaba, Michimasa",
    booktitle = "Proceedings of the 18th International Natural Language Generation Conference",
    month = oct,
    year = "2025",
    address = "Hanoi, Vietnam",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.inlg-main.1/",
    pages = "1--19"
}
```
