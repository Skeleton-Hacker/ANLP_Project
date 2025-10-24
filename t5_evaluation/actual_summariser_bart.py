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

def compute_metrics(predictions, references, device="cpu"):
    """
    Compute ROUGE and BERT scores for a list of predictions and references.
    """
    scores = {}

    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            for k in rouge_scores:
                rouge_scores[k].append(result[k].fmeasure)
        scores.update({k: float(np.mean(v) * 100) for k, v in rouge_scores.items()})
    except ImportError:
        print("⚠️ Install rouge-score with: pip install rouge-score")
        scores.update({"rouge1": 0, "rouge2": 0, "rougeL": 0})

    # BERTScore
    try:
        from bert_score import score as bert_score_scorer
        P, R, F1 = bert_score_scorer(predictions, references, lang="en", verbose=False, device=device)
        scores["bert_score_f1"] = float(F1.mean() * 100)
    except ImportError:
        print("⚠️ Install bert-score with: pip install bert-score")
        scores["bert_score_f1"] = 0
        
    return scores

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
    metrics = compute_metrics(predictions, references, device=device) if predictions else {}
    print(f"\nSummarization Metrics: {metrics}")

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
        "metrics": metrics,
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
