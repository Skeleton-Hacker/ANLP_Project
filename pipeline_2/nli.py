"""
NLI Evaluation for Summarization Models

Evaluates whether generated summaries are entailed by the source documents
using a pre-trained NLI model (BART-large-MNLI).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLIEvaluator:
    """Natural Language Inference evaluator for summarization"""
    
    def __init__(self, model_name="facebook/bart-large-mnli", device=None):
        """
        Initialize NLI evaluator
        
        Args:
            model_name: HuggingFace model name for NLI
            device: torch device (auto-detected if None)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading NLI model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logger.info("NLI model loaded successfully")
    
    def compute_nli_score(self, premise, hypothesis, max_length=1024):
        """
        Compute NLI scores between premise (source) and hypothesis (summary)
        
        Args:
            premise: Source document text
            hypothesis: Generated summary text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with entailment, neutral, and contradiction scores
        """
        # Tokenize the premise-hypothesis pair
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # BART-MNLI labels: [contradiction, neutral, entailment]
        scores = {
            "contradiction": probs[0][0].item(),
            "neutral": probs[0][1].item(),
            "entailment": probs[0][2].item()
        }
        
        return scores
    
    def evaluate_file(self, source_docs, predictions, batch_size=8):
        """
        Evaluate NLI scores for a list of document-summary pairs
        
        Args:
            source_docs: List of source documents
            predictions: List of generated summaries
            batch_size: Batch size for processing (currently processes one at a time)
            
        Returns:
            Dictionary with aggregated NLI metrics
        """
        if len(source_docs) != len(predictions):
            raise ValueError(f"Mismatch: {len(source_docs)} docs vs {len(predictions)} predictions")
        
        all_scores = []
        entailment_scores = []
        neutral_scores = []
        contradiction_scores = []
        
        logger.info(f"Evaluating {len(predictions)} samples...")
        
        for doc, pred in tqdm(zip(source_docs, predictions), total=len(predictions), desc="Computing NLI"):
            scores = self.compute_nli_score(doc, pred)
            all_scores.append(scores)
            entailment_scores.append(scores["entailment"])
            neutral_scores.append(scores["neutral"])
            contradiction_scores.append(scores["contradiction"])
        
        # Compute aggregate statistics
        results = {
            "mean_entailment": float(np.mean(entailment_scores)),
            "median_entailment": float(np.median(entailment_scores)),
            "std_entailment": float(np.std(entailment_scores)),
            "mean_neutral": float(np.mean(neutral_scores)),
            "mean_contradiction": float(np.mean(contradiction_scores)),
            "num_samples": len(predictions),
            "detailed_scores": all_scores
        }
        
        return results


def load_samples_file(filepath):
    """
    Load samples from the generated text files
    
    Expected format:
    Sample 1:
    Reference: <reference text>
    Generated: <generated text>
    ----------------
    """
    samples = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by sample delimiter
    sample_blocks = content.split('Sample ')
    
    for block in sample_blocks[1:]:  # Skip first empty block
        lines = block.strip().split('\n')
        reference = None
        generated = None
        
        for line in lines:
            if line.startswith('Reference:'):
                reference = line.replace('Reference:', '').strip()
            elif line.startswith('Generated:'):
                generated = line.replace('Generated:', '').strip()
        
        if reference and generated:
            samples.append({
                'reference': reference,
                'generated': generated
            })
    
    return samples


def load_chunked_data_for_sources(split="test"):
    """
    Load source documents from chunked data
    This is needed because we need the original full documents as premises
    """
    import pickle
    
    data_path = Path("chunked_data") / f"{split}_chunks_encoded.pkl"
    
    if not data_path.exists():
        logger.warning(f"Chunked data not found at {data_path}")
        return None
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    # Extract source documents (concatenate chunks)
    sources = []
    for story_id, story in data['stories'].items():
        if 'chunks' in story and len(story['chunks']) > 0:
            # Join chunks to get full document
            full_doc = ' '.join(story['chunks'])
            sources.append(full_doc)
    
    logger.info(f"Loaded {len(sources)} source documents from {split} split")
    return sources


def main():
    """Main evaluation function"""
    
    logger.info("="*80)
    logger.info("NLI EVALUATION FOR SUMMARIZATION MODELS")
    logger.info("="*80)
    
    # Initialize NLI evaluator
    evaluator = NLIEvaluator()
    
    # Load source documents
    logger.info("\nLoading source documents...")
    source_docs = load_chunked_data_for_sources("test")
    
    if source_docs is None:
        logger.error("Could not load source documents. Please ensure chunked_data/test_chunks_encoded.pkl exists")
        return
    
    # Define models to evaluate
    models_config = {
        "BART Base (Pretrained)": {
            "samples_file": "eval_base/samples_bart_base.txt",  # You'll need to generate this
            "output_table": "eval_base/table_bart_base_nli.txt"
        },
        "BART Untrained": {
            "samples_file": "eval_untrained/samples_bart_untrained.txt",
            "output_table": "eval_untrained/table_bart_untrained_nli.txt"
        },
        "BART Finetuned": {
            "samples_file": "eval_trained/samples_bart_finetuned.txt",
            "output_table": "eval_trained/table_bart_finetuned_nli.txt"
        },
        "T5 Base (Pretrained)": {
            "samples_file": "eval_base/samples_t5_base.txt",  # You'll need to generate this
            "output_table": "eval_base/table_t5_base_nli.txt"
        },
        "T5 Untrained": {
            "samples_file": "eval_untrained/samples_t5_untrained.txt",
            "output_table": "eval_untrained/table_t5_untrained_nli.txt"
        },
        "T5 Finetuned": {
            "samples_file": "eval_trained/samples_t5_finetuned.txt",
            "output_table": "eval_trained/table_t5_finetuned_nli.txt"
        }
    }
    
    all_results = {}
    
    # Evaluate each model
    for model_name, config in models_config.items():
        samples_path = Path(config["samples_file"])
        
        if not samples_path.exists():
            logger.warning(f"Skipping {model_name}: {samples_path} not found")
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*80}")
        
        # Load samples
        samples = load_samples_file(samples_path)
        logger.info(f"Loaded {len(samples)} samples")
        
        # Get corresponding source documents
        # Assuming samples are in same order as test set
        num_samples = len(samples)
        sample_sources = source_docs[:num_samples]
        sample_predictions = [s['generated'] for s in samples]
        
        # Evaluate NLI
        results = evaluator.evaluate_file(sample_sources, sample_predictions)
        all_results[model_name] = results
        
        # Print results
        logger.info(f"\nNLI Results for {model_name}:")
        logger.info(f"  Mean Entailment:     {results['mean_entailment']:.4f}")
        logger.info(f"  Median Entailment:   {results['median_entailment']:.4f}")
        logger.info(f"  Std Entailment:      {results['std_entailment']:.4f}")
        logger.info(f"  Mean Neutral:        {results['mean_neutral']:.4f}")
        logger.info(f"  Mean Contradiction:  {results['mean_contradiction']:.4f}")
        
        # Save individual results table
        table_path = Path(config["output_table"])
        table_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(table_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"NLI EVALUATION RESULTS: {model_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Number of samples:      {results['num_samples']}\n\n")
            f.write(f"Mean Entailment:        {results['mean_entailment']:.4f}\n")
            f.write(f"Median Entailment:      {results['median_entailment']:.4f}\n")
            f.write(f"Std Entailment:         {results['std_entailment']:.4f}\n\n")
            f.write(f"Mean Neutral:           {results['mean_neutral']:.4f}\n")
            f.write(f"Mean Contradiction:     {results['mean_contradiction']:.4f}\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Saved results to {table_path}")
    
    # Create comparison table
    if all_results:
        logger.info("\n" + "="*80)
        logger.info("COMPARISON TABLE - ALL MODELS")
        logger.info("="*80 + "\n")
        
        # Console output
        header = f"{'Model':<25} {'Entailment':>12} {'Neutral':>12} {'Contradiction':>12}"
        logger.info(header)
        logger.info("-" * 65)
        
        for model_name, results in all_results.items():
            row = f"{model_name:<25} {results['mean_entailment']:>12.4f} {results['mean_neutral']:>12.4f} {results['mean_contradiction']:>12.4f}"
            logger.info(row)
        
        # Save comparison table
        comparison_file = Path("evaluations") / "nli_comparison_all_models.txt"
        comparison_file.parent.mkdir(exist_ok=True)
        
        with open(comparison_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NLI EVALUATION COMPARISON - ALL MODELS\n")
            f.write("="*80 + "\n\n")
            f.write(header + "\n")
            f.write("-" * 65 + "\n")
            
            for model_name, results in all_results.items():
                row = f"{model_name:<25} {results['mean_entailment']:>12.4f} {results['mean_neutral']:>12.4f} {results['mean_contradiction']:>12.4f}"
                f.write(row + "\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("INTERPRETATION:\n")
            f.write("- Entailment:     Summary is logically derived from source (HIGHER is BETTER)\n")
            f.write("- Neutral:        Summary is unrelated to source\n")
            f.write("- Contradiction:  Summary contradicts source (LOWER is BETTER)\n")
        
        logger.info(f"\nComparison table saved to {comparison_file}")
        
        # Save detailed JSON results
        json_file = Path("evaluations") / "nli_results_detailed.json"
        # Remove detailed_scores for cleaner JSON
        json_results = {
            model: {k: v for k, v in results.items() if k != 'detailed_scores'}
            for model, results in all_results.items()
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Detailed results saved to {json_file}")
    
    logger.info("\n" + "="*80)
    logger.info("NLI EVALUATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()