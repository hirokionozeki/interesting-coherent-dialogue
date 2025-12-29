#!/bin/bash
# Data and Checkpoints Setup Script
# This script downloads and sets up all required data and model checkpoints

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Data and Checkpoints Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Google Drive file ID
GDRIVE_FILE_ID="17KFNM2K17uEQAmZehgfZAho3tweTIiXM"

cd "$REPO_ROOT"

# Step 1: Install gdown if not available
echo -e "${YELLOW}[1/4] Checking dependencies...${NC}"
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    python3 -m pip install -q gdown || {
        echo -e "${RED}Error: Failed to install gdown.${NC}"
        echo "Please install it manually: python3 -m pip install gdown"
        echo ""
        echo "Or download the file manually from:"
        echo "  https://drive.google.com/file/d/17KFNM2K17uEQAmZehgfZAho3tweTIiXM/view?usp=sharing"
        exit 1
    }
fi
echo "gdown is available."

# Step 2: Download data archive from Google Drive
echo -e "${YELLOW}[2/4] Downloading data and checkpoints from Google Drive (~2.2GB)...${NC}"
if [ -f "data_and_checkpoints.tar.gz" ]; then
    echo "Archive already exists. Skipping download."
else
    gdown "$GDRIVE_FILE_ID" -O data_and_checkpoints.tar.gz
fi

# Step 3: Extract archive
echo -e "${YELLOW}[3/4] Extracting archive...${NC}"
tar -xzf data_and_checkpoints.tar.gz

# Step 4: Download SPI checkpoint
echo -e "${YELLOW}[4/4] Setting up SPI checkpoint...${NC}"
echo ""
echo -e "${YELLOW}Manual step required:${NC}"
echo "1. Visit: https://hkustconnect-my.sharepoint.com/:f:/g/personal/yxucb_connect_ust_hk/Euyhj33uFtdLnBcWe4bHBukB4rjbXSaoWRbG2PZ6Mcdt9Q?e=WCs3ap"
echo "2. Download the checkpoint files"
echo "3. Extract and place them in: checkpoints/spi/best_model/"
echo ""
echo -e "${YELLOW}Press Enter after you've completed the SPI checkpoint setup...${NC}"
read -r

# Verify setup
echo ""
echo -e "${YELLOW}Verifying setup...${NC}"
echo ""

MISSING_FILES=0

# Check data files
for file in "data/raw/test_random_split.json" \
            "data/raw/test_topic_split.json" \
            "data/reference/trivia_scores.json" \
            "data/genks/precomputed/seen_full_triviascore.json" \
            "data/genks/precomputed/unseen_full_triviascore.json" \
            "data/genks/precomputed/seen_full_dbd_cache.json" \
            "data/genks/precomputed/unseen_full_dbd_cache.json" \
            "data/spi/precomputed/model_pred_all_need_data.json" \
            "data/spi/precomputed/dbd_result_dict.json" \
            "checkpoints/wow-bart-large/4.pt" \
            "checkpoints/wow-distilbert-psg-rank/0.pt" \
            "checkpoints/psg_filter/wizard.seen_full.0.json" \
            "checkpoints/psg_filter/wizard.unseen_full.0.json" \
            "checkpoints/psg_filter/wizard.test_mini.0.json" \
            "checkpoints/spi/best_model/config.json" \
            "checkpoints/spi/best_model/pytorch_model.bin"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file (missing)"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

echo ""
if [ $MISSING_FILES -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "Setup completed successfully!"
    echo "==========================================${NC}"
    echo ""
    echo "You can now proceed with:"
    echo "  1. Environment setup (see README.md)"
    echo "  2. Data preparation"
    echo "  3. Running inference"
else
    echo -e "${RED}=========================================="
    echo "Setup incomplete: $MISSING_FILES file(s) missing"
    echo "==========================================${NC}"
    echo ""
    echo "Please check the missing files and rerun the script."
    exit 1
fi

# Clean up
echo ""
read -p "Remove downloaded archive (data_and_checkpoints.tar.gz)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm data_and_checkpoints.tar.gz
    echo "Archive removed."
fi

echo ""
echo "Done!"
