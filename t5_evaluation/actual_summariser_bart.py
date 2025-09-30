import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import json
import random

# -----------------------------------------------------------
# 1. Metrics
# -----------------------------------------------------------

def compute_rouge(predictions, references):
    """
    Compute ROUGE scores for a list of predictions and references.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            for k in scores:
                scores[k].append(result[k].fmeasure)
        return {k: float(np.mean(v) * 100) for k, v in scores.items()}
    except ImportError:
        print("⚠️ Install rouge-score with: pip install rouge-score")
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

# -----------------------------------------------------------
# 2. Summarization Model Evaluation
# -----------------------------------------------------------

def evaluate_summarization_model(model_name="facebook/bart-large-cnn", num_samples=50, device=None, save_samples=5):
    """
    Evaluates a dedicated summarization model on the NarrativeQA dataset.
    """
    # Use CUDA if available, otherwise CPU. User can set CUDA_VISIBLE_DEVICES to select a specific GPU.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running summarization evaluation on {device} using model: {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    dataset = load_dataset("deepmind/narrativeqa", split="validation").select(range(num_samples))

    predictions, references, examples = [], [], []
    good_examples = []  # Track good examples

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating Summarization")):
        document = sample.get("document", {}).get("text")
        summary = sample.get("document", {}).get("summary", {}).get("text")

        if not document or not document.strip() or not summary or not summary.strip():
            continue

        # Summarization
        inputs = tokenizer(document, max_length=1024, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
        
        pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        predictions.append(pred_summary)
        references.append(summary)

        # Collect good summarization examples (basic quality check)
        if len(pred_summary.split()) > 10 and len(good_examples) < 10:  # At least 10 words
            good_examples.append({
                "reference_summary": summary,
                "prediction": pred_summary,
                "doc_excerpt": document[:300] + "..."
            })

        if random.random() < (save_samples / num_samples):
            examples.append({
                "reference_summary": summary,
                "prediction": pred_summary,
                "doc_excerpt": document[:300] + "..."
            })

    # Compute Metrics
    rouge_scores = compute_rouge(predictions, references) if predictions else {}
    print(f"\nSummarization ROUGE Scores: {rouge_scores}")

    # Print good examples
    print("\n" + "="*80)
    print("GOOD SUMMARIZATION EXAMPLES:")
    print("="*80)
    for i, example in enumerate(good_examples[:10], 1):
        print(f"\n--- Summary Example {i} ---")
        print(f"Reference: {example['reference_summary']}")
        print(f"Prediction: {example['prediction']}")
        print(f"Document: {example['doc_excerpt']}")

    return {
        "model": model_name,
        "rouge": rouge_scores,
        "examples": examples,
        "good_examples": good_examples
    }

# -----------------------------------------------------------
# 3. Run Evaluation
# -----------------------------------------------------------

if __name__ == "__main__":
    results = evaluate_summarization_model(num_samples=50, save_samples=8)
    with open("summarization_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n✅ Saved results + sample predictions to summarization_results.json")
