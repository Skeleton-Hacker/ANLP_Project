#!/bin/bash

# QnA Pipeline Execution Script
# This script runs the complete QA pipeline for NarrativeQA dataset

set -e  # Exit on error

echo "=========================================="
echo "QnA Pipeline - Question Answering"
echo "=========================================="

# Step 1: Semantic Chunking (using pipeline_2 code)
echo ""
echo "Step 1: Running Semantic Chunking..."
echo "=========================================="
accelerate launch semantic_chunking.py

# Step 2: Process QA Data with Embeddings
echo ""
echo "Step 2: Processing QA Data with Chunk Embeddings..."
echo "=========================================="
python chunk_qa_data.py

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