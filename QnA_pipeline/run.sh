#!/bin/bash
# filepath: /home/yajatr/Documents/Codes/Acads/Sem5/ANLP/ANLP_Project/QnA_pipeline/run.sh

set -e

echo "=========================================="
echo "QnA Pipeline - Using Pre-computed Chunks"
echo "=========================================="

# Set NCCL environment variables for RTX 4000 series
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Verify chunked data exists
echo ""
echo "Step 1: Verifying chunked data..."
echo "=========================================="

if [ ! -f "chunked_data/train_chunks_encoded.pkl" ]; then
    echo "ERROR: Chunked data not found in chunked_data/"
    echo "Please ensure the following files exist:"
    echo "  - train_chunks_encoded.pkl"
    echo "  - validation_chunks_encoded.pkl"
    echo "  - test_chunks_encoded.pkl"
    exit 1
fi

echo "✓ Found chunked data files:"
ls -lh chunked_data/*_chunks*.pkl

# Step 2: Create QA encoded files
# echo ""
# echo "Step 2: Creating QA encoded files from chunks..."
# echo "=========================================="
# accelerate launch chunk_qa_data.py  # Changed from 'python' to 'accelerate launch'

# # Verify QA files were created
# if [ ! -f "chunked_data/train_qa_encoded.pkl" ]; then
#     echo "ERROR: Failed to create QA encoded files"
#     exit 1
# fi

# echo "✓ QA encoded files created:"
# ls -lh chunked_data/*_qa_encoded.pkl

# Step 3: Train BART QA Model
echo ""
echo "Step 3: Training BART QA Model..."
echo "=========================================="
accelerate launch bart_qa_model.py

# Step 4: Evaluate Optimized Model
echo ""
echo "Step 4: Evaluating Optimized BART QA Model..."
echo "=========================================="
accelerate launch bart_qa_model.py --eval-only

# Step 5: Evaluate Baselines
echo ""
echo "Step 5: Evaluating Baseline Models..."
echo "=========================================="

echo "Evaluating BART Baseline (100 samples)..."
python baseline_qa.py --model bart --dataset deepmind/narrativeqa --num-samples 100

echo "Evaluating LLaMA-3B Baseline (100 samples)..."
python baseline_qa.py --model llama-3b --dataset deepmind/narrativeqa --num-samples 100

echo "Evaluating LLaMA-8B Baseline (50 samples)..."
python baseline_qa.py --model llama-8b --dataset deepmind/narrativeqa --num-samples 50

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results saved in:"
echo "  - models/bart_qa/best_model.pt (optimized model)"
echo "  - baseline_results/*.json (baseline results)"