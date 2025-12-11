#!/usr/bin/env python3
"""
Compare T5 Finetuned, T5 Base, and Gemini Flash on N random samples.
Simple serial processing with 90s delays between Gemini calls.
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import google.generativeai as genai
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Import from the training script
import sys
sys.path.append(str(Path(__file__).parent))
from t5_full_finetune import ChunkDataset, T5ChunkModel, Config, collate_fn

# ---------------------------
# Logging setup
# ---------------------------
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("model_comparator")

# ---------------------------
# Model Comparator
# ---------------------------
class ModelComparator:
    def __init__(
        self,
        config: Config,
        gemini_api_key: Optional[str] = None,
        gemini_model_name: str = "gemini-2.5-flash",
        enable_gemini: bool = True
    ):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        # Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(config.t5_model)

        # Gemini setup
        self.enable_gemini = bool(enable_gemini)
        self.gemini_model_name = gemini_model_name
        if self.enable_gemini:
            if not gemini_api_key:
                raise ValueError("Gemini enabled but no API key provided.")
            genai.configure(api_key=gemini_api_key)
            self.gemini = genai.GenerativeModel(gemini_model_name)
            logger.info(f"Configured Gemini model: {gemini_model_name}")
        else:
            self.gemini = None
            logger.info("Gemini disabled.")

        # Load finetuned model
        logger.info("Loading finetuned T5 model...")
        self.finetuned_model = T5ChunkModel(
            config.t5_model, 
            config.embedding_dim, 
            freeze_decoder=config.freeze_decoder,
            config=config
        )
        model_path = Path(config.output_dir) / "best_model.pt"
        if not model_path.exists():
            logger.error(f"Finetuned model checkpoint not found: {model_path}")
            raise FileNotFoundError(model_path)
        self.finetuned_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.finetuned_model.eval()
        self.finetuned_model = self.finetuned_model.to(self.device)

        # Load base T5
        logger.info("Loading base T5 model...")
        self.base_model = T5ForConditionalGeneration.from_pretrained(config.t5_model)
        self.base_model.eval()
        self.base_model = self.base_model.to(self.device)

    # -----------------------
    # Gemini generator
    # -----------------------
    def generate_gemini_summary(self, text: str) -> str:
        """Generate summary using Gemini. Waits 90s before calling."""
        if not self.enable_gemini:
            return "[GEMINI_DISABLED]"

        prompt = f"Summarize the following story in 3-5 sentences:\n\n{text}\n\nSummary:"
        
        try:
            logger.info("Waiting 10s before Gemini call...")
            time.sleep(10)
            logger.info("Calling Gemini API...")
            response = self.gemini.generate_content(prompt)
            summary = response.text.strip() if hasattr(response, 'text') else str(response).strip()
            logger.info(f"Gemini generated summary (length={len(summary)})")
            return summary
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"[ERROR: {str(e)}]"

    # -----------------------
    # T5 generators
    # -----------------------
    def generate_finetuned_summary(self, embeddings: torch.Tensor, chunk_mask: torch.Tensor) -> str:
        with torch.no_grad():
            embeddings = embeddings.unsqueeze(0).to(self.device)
            chunk_mask = chunk_mask.unsqueeze(0).to(self.device)
            gen_ids = self.finetuned_model.generate(
                embeddings,
                chunk_mask,
                max_len=self.config.max_target_len,
                min_len=self.config.min_target_len,
                num_beams=self.config.num_beams,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                repetition_penalty=self.config.repetition_penalty,
                length_penalty=self.config.length_penalty
            )
            summary = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        logger.info(f"Finetuned T5 generated (length={len(summary)})")
        return summary

    def generate_base_summary(self, text: str) -> str:
        with torch.no_grad():
            input_enc = self.tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
            input_enc = {k: v.to(self.device) for k, v in input_enc.items()}
            gen_ids = self.base_model.generate(
                input_enc['input_ids'],
                attention_mask=input_enc['attention_mask'],
                max_length=self.config.max_target_len,
                min_length=self.config.min_target_len,
                num_beams=self.config.num_beams,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                repetition_penalty=self.config.repetition_penalty
            )
            summary = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        logger.info(f"Base T5 generated (length={len(summary)})")
        return summary

    # -----------------------
    # Metrics
    # -----------------------
    def compute_metrics(self, pred: str, ref: str) -> Dict:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(ref, pred)
        P, R, F1 = bert_score([pred], [ref], lang='en', verbose=False)
        return {
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'bert_precision': float(P[0].item()),
            'bert_recall': float(R[0].item()),
            'bert_f1': float(F1[0].item())
        }

    # -----------------------
    # Main evaluation loop (SERIAL)
    # -----------------------
    def evaluate_samples(self, test_dataset: ChunkDataset, num_samples: int = 25, output_file: Path = Path("evaluations/comparison.json")) -> Dict:
        random.seed(42)
        sample_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results = []
        logger.info(f"Evaluating {len(sample_indices)} samples serially...")

        # Initialize output structure and write header once
        output = {
            'num_samples': 0,
            'timestamp_started': datetime.now().isoformat(),
            'samples': [],
            'average_metrics': {}
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        for idx in tqdm(sample_indices, desc="Processing samples"):
            sample = test_dataset[idx]
            input_text = sample['doc_text']
            ref_summary = sample['target_text']
            story_id = sample.get('story_id', f"idx_{idx}")

            logger.info(f"Processing sample {idx} (story_id: {story_id})")

            # Build chunk mask for finetuned model
            embeddings = sample['embeddings']
            num_chunks = sample.get('num_chunks', len(embeddings))
            chunk_mask = torch.zeros(len(embeddings), dtype=torch.bool)
            chunk_mask[:num_chunks] = True

            # Generate summaries (SERIAL - one at a time)
            finetuned_summary = self.generate_finetuned_summary(embeddings, chunk_mask)
            base_summary = self.generate_base_summary(input_text)
            gemini_summary = self.generate_gemini_summary(input_text)

            # Compute metrics
            finetuned_metrics = self.compute_metrics(finetuned_summary, ref_summary)
            base_metrics = self.compute_metrics(base_summary, ref_summary)
            gemini_metrics = self.compute_metrics(gemini_summary, ref_summary) if not gemini_summary.startswith("[ERROR") and not gemini_summary.startswith("[GEMINI") else {}

            # Build result entry
            result = {
                'sample_index': idx,
                'story_id': story_id,
                'timestamp': datetime.now().isoformat(),
                'input_text': input_text,
                'reference_summary': ref_summary[:1000],
                'finetuned_t5': {
                    'summary': finetuned_summary,
                    'metrics': finetuned_metrics
                },
                'base_t5': {
                    'summary': base_summary,
                    'metrics': base_metrics
                },
                'gemini_flash': {
                    'summary': gemini_summary,
                    'metrics': gemini_metrics
                }
            }
            results.append(result)

            # Immediately update single JSON file after each sample
            current_output = {
                'num_samples': len(results),
                'timestamp_started': output['timestamp_started'],
                'timestamp_latest_update': datetime.now().isoformat(),
                'samples': results,
                'average_metrics': self.aggregate_metrics(results)
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_output, f, indent=2, ensure_ascii=False)

        # Final update at end
        final_output = {
            'num_samples': len(results),
            'timestamp_started': output['timestamp_started'],
            'timestamp_finished': datetime.now().isoformat(),
            'samples': results,
            'average_metrics': self.aggregate_metrics(results)
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved final results to {output_file}")
        return final_output

    def aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Compute average metrics across all samples."""
        models = ['finetuned_t5', 'base_t5', 'gemini_flash']
        metrics_keys = ['rouge1', 'rouge2', 'rougeL', 'bert_precision', 'bert_recall', 'bert_f1']
        
        aggregated = {}
        for model in models:
            values_by_metric = {k: [] for k in metrics_keys}
            for r in results:
                metrics = r.get(model, {}).get('metrics', {})
                for k in metrics_keys:
                    v = metrics.get(k)
                    if v is not None:
                        values_by_metric[k].append(float(v))
            
            model_metrics = {}
            for k in metrics_keys:
                vals = values_by_metric[k] or [0.0]
                model_metrics[k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals))
                }
            aggregated[model] = model_metrics
        
        return aggregated

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Compare T5 Finetuned, T5 Base, and Gemini Flash models")
    parser.add_argument('--gemini-api-key', type=str, default=os.environ.get('GEMINI_API_KEY'), help='Google Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--num-samples', type=int, default=25, help='Number of random samples to evaluate')
    parser.add_argument('--output-file', type=str, default='evaluations/comparison.json', help='Output JSON file')
    parser.add_argument('--no-gemini', action='store_true', help='Disable Gemini (dry-run)')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()

    # Load test dataset
    test_path = Path(config.encoded_data_dir) / "test_chunks_encoded.pkl"
    if not test_path.exists():
        logger.error(f"Test data not found at {test_path}")
        return

    logger.info("Loading test dataset...")
    tokenizer = T5Tokenizer.from_pretrained(config.t5_model)
    test_dataset = ChunkDataset(test_path, tokenizer, config.max_chunks, config.max_target_len, validate=False)

    comparator = ModelComparator(
        config=config,
        gemini_api_key=args.gemini_api_key,
        gemini_model_name="gemini-2.5-flash",
        enable_gemini=not args.no_gemini
    )

    results = comparator.evaluate_samples(test_dataset, num_samples=args.num_samples, output_file=Path(args.output_file))
    logger.info("Evaluation complete!")
    logger.info(f"Average metrics: {json.dumps(results.get('average_metrics', {}), indent=2)}")

if __name__ == "__main__":
    main()
