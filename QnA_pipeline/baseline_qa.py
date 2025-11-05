"""
Baseline models for QA (unoptimized BART, LLaMA-3B, LLaMA-8B)
"""
import logging
import torch
import numpy as np
import json
import os
from pathlib import Path
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)
from tqdm import tqdm
from typing import Dict, List
from dotenv import load_dotenv

from load_dataset import load_test

# Load environment variables
load_dotenv(dotenv_path=".env")
HF_TOKEN = os.getenv("HUGGING_FACE_LLAMA_TOKEN")

if HF_TOKEN:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("HuggingFace token loaded successfully")
else:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.warning("No HuggingFace token found in .env file")


def truncate_document(text: str, tokenizer, max_length: int = 1024) -> str:
    """Truncate document to fit in model's context window."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def compute_metrics(preds: List[str], refs: List[str]) -> Dict:
    """Compute ROUGE, BERT scores, and Exact Match for QA."""
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {k: [] for k in ['rouge1', 'rouge2', 'rougeL']}
    exact_matches = []
    
    for pred, ref in zip(preds, refs):
        # ROUGE
        scores = scorer.score(ref, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Exact match (case-insensitive, normalized)
        pred_normalized = pred.lower().strip()
        ref_normalized = ref.lower().strip()
        exact_matches.append(1.0 if pred_normalized == ref_normalized else 0.0)
    
    # BERTScore
    if len(preds) > 0:
        P, R, F1 = bert_score(preds, refs, lang='en', verbose=False)
        bert_f1 = F1.mean().item()
    else:
        bert_f1 = 0.0
    
    return {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL']),
        'bert_f1': bert_f1,
        'exact_match': np.mean(exact_matches)
    }


def evaluate_bart_baseline(
    dataset_name: str = "deepmind/narrativeqa",
    model_name: str = "facebook/bart-large",
    max_doc_length: int = 1024,
    num_samples: int = None,
    device: str = "cuda"
):
    """Evaluate unoptimized BART baseline."""
    logger.info("="*80)
    logger.info(f"Evaluating BART baseline: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Max samples: {num_samples if num_samples else 'all'}")
    logger.info("="*80)
    
    logger.info("Loading tokenizer...")
    tokenizer = BartTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    
    logger.info("Loading model...")
    model = BartForConditionalGeneration.from_pretrained(model_name, token=HF_TOKEN).to(device)
    logger.info(f"Model loaded on device: {device}")
    
    # Load test data
    logger.info("Loading test dataset...")
    stories_dict = load_test(dataset_name, max_samples=num_samples, group_by_story=True)
    logger.info(f"Loaded {len(stories_dict)} stories")
    
    predictions = []
    references = []
    
    total_questions = sum(len(story['questions']) for story in stories_dict.values())
    logger.info(f"Total questions to process: {total_questions}")
    
    question_count = 0
    for story_id, story_data in tqdm(stories_dict.items(), desc="BART Baseline - Processing stories"):
        document = story_data['document'].get('text', '')
        
        for question, answer in zip(story_data['questions'], story_data['answers']):
            question_count += 1
            
            # Truncate document
            doc_truncated = truncate_document(document, tokenizer, max_doc_length - 100)
            
            # Format input: question + context
            input_text = f"question: {question} context: {doc_truncated}"
            
            inputs = tokenizer(input_text, max_length=max_doc_length, truncation=True, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    min_length=5,
                    num_beams=4,
                    early_stopping=True
                )
            
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(pred)
            references.append(answer if isinstance(answer, str) else "")
            
            if question_count % 10 == 0:
                logger.info(f"Processed {question_count}/{total_questions} questions")
    
    logger.info("Computing metrics...")
    metrics = compute_metrics(predictions, references)
    
    logger.info("\n" + "="*80)
    logger.info(f"BART Baseline Results ({dataset_name}):")
    logger.info("="*80)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("="*80)
    
    return metrics, predictions, references


def evaluate_llama_baseline(
    dataset_name: str = "deepmind/narrativeqa",
    model_name: str = "meta-llama/Llama-3.2-3B",
    max_doc_length: int = 2048,
    num_samples: int = None,
    device: str = "cuda"
):
    """Evaluate unoptimized LLaMA baseline."""
    logger.info("="*80)
    logger.info(f"Evaluating LLaMA baseline: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Max samples: {num_samples if num_samples else 'all'}")
    logger.info("="*80)
    
    if not HF_TOKEN:
        logger.error("HuggingFace token not found! LLaMA models require authentication.")
        logger.error("Please ensure HUGGING_FACE_LLAMA_TOKEN is set in ../.env file")
        raise ValueError("Missing HuggingFace token for LLaMA models")
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully")
    
    logger.info("Loading model... (this may take a while)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN
    )
    logger.info("Model loaded successfully")
    
    # Load test data
    logger.info("Loading test dataset...")
    stories_dict = load_test(dataset_name, max_samples=num_samples, group_by_story=True)
    logger.info(f"Loaded {len(stories_dict)} stories")
    
    predictions = []
    references = []
    
    total_questions = sum(len(story['questions']) for story in stories_dict.values())
    logger.info(f"Total questions to process: {total_questions}")
    
    question_count = 0
    for story_id, story_data in tqdm(stories_dict.items(), desc=f"LLaMA Baseline - Processing stories"):
        document = story_data['document'].get('text', '')
        
        for question, answer in zip(story_data['questions'], story_data['answers']):
            question_count += 1
            
            # Truncate document
            doc_truncated = truncate_document(document, tokenizer, max_doc_length - 200)
            
            # Format prompt for LLaMA
            if dataset_name.lower() == "peerqa":
                prompt = f"""You are reviewing a scientific paper. Based on the paper content, answer the reviewer's question.

Paper: {doc_truncated}

Reviewer Question: {question}

Author Response:"""
            else:
                prompt = f"""Based on the following document, answer the question.

Document: {doc_truncated}

Question: {question}

Answer:"""
            
            inputs = tokenizer(prompt, max_length=max_doc_length, truncation=True, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    min_new_tokens=5,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the generated part (exclude prompt)
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            predictions.append(pred)
            references.append(answer if isinstance(answer, str) else "")
            
            if question_count % 10 == 0:
                logger.info(f"Processed {question_count}/{total_questions} questions")
    
    logger.info("Computing metrics...")
    metrics = compute_metrics(predictions, references)
    
    logger.info("\n" + "="*80)
    logger.info(f"LLaMA Baseline Results ({model_name}, {dataset_name}):")
    logger.info("="*80)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("="*80)
    
    return metrics, predictions, references


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bart', 
                       choices=['bart', 'llama-3b', 'llama-8b'])
    parser.add_argument('--dataset', type=str, default='deepmind/narrativeqa')
    parser.add_argument('--num-samples', type=int, default=None)
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("QA Baseline Evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info("="*80)
    
    results_dir = Path("baseline_results")
    results_dir.mkdir(exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")
    
    if args.model == 'bart':
        metrics, preds, refs = evaluate_bart_baseline(
            args.dataset,
            "facebook/bart-large",
            num_samples=args.num_samples
        )
    elif args.model == 'llama-3b':
        metrics, preds, refs = evaluate_llama_baseline(
            args.dataset,
            "meta-llama/Llama-3.2-3B",
            num_samples=args.num_samples
        )
    elif args.model == 'llama-8b':
        metrics, preds, refs = evaluate_llama_baseline(
            args.dataset,
            "meta-llama/Llama-3.1-8B",
            max_doc_length=4096,
            num_samples=args.num_samples
        )
    
    # Save results
    output_file = results_dir / f"{args.model}_{args.dataset.replace('/', '_')}_results.json"
    logger.info(f"Saving results to: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump({
            'model': args.model,
            'dataset': args.dataset,
            'metrics': metrics,
            'num_samples': len(preds),
            'num_predictions': len(preds),
            'num_references': len(refs)
        }, f, indent=2)
    
    logger.info("="*80)
    logger.info("Evaluation Complete!")
    logger.info(f"Results saved to: {output_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()