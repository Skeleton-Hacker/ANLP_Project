import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import json
import random

# -----------------------------------------------------------
# 1. Metrics
# -----------------------------------------------------------

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def compute_rouge(predictions, references):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            for k in scores:
                scores[k].append(result[k].fmeasure)
        return {k: float(np.mean(v) * 100) for k,v in scores.items()}
    except ImportError:
        print("⚠️ Install rouge-score with: pip install rouge-score")
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

# -----------------------------------------------------------
# 2. Baseline Evaluation
# -----------------------------------------------------------

def evaluate_baseline(model_name="t5-base", num_samples=50, device=None, save_samples=5):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running baseline on {device} ...")

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    dataset = load_dataset("deepmind/narrativeqa", split="validation").select(range(num_samples))

    qa_predictions, qa_references, qa_examples, qa_f1s = [], [], [], []
    sum_predictions, sum_references, sum_examples = [], [], []

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        document = sample["document"]["text"] if "document" in sample else ""
        question = sample["question"]["text"] if "question" in sample else ""
        answer = sample["answers"][0]["text"] if len(sample["answers"]) > 0 else ""
        summary = sample["document"]["summary"]["text"] if "summary" in sample.get("document", {}) else ""

        if not document.strip():
            continue

        # ---------------- QA ----------------
        if question.strip() and answer.strip():
            input_text = f"question: {question} context: {document}"
            inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

            qa_predictions.append(pred)
            qa_references.append(answer)
            qa_f1s.append(compute_f1(pred, answer))

            if random.random() < (save_samples / num_samples):
                qa_examples.append({
                    "question": question,
                    "ground_truth": answer,
                    "prediction": pred,
                    "doc_excerpt": document[:200] + "..."
                })

        # ---------------- Summarization ----------------
        if summary.strip():
            sum_input = f"summarize: {document}"
            sum_inputs = tokenizer(sum_input, max_length=512, truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():
                sum_outputs = model.generate(**sum_inputs, max_length=128, num_beams=4, early_stopping=True)
            sum_pred = tokenizer.decode(sum_outputs[0], skip_special_tokens=True)

            sum_predictions.append(sum_pred)
            sum_references.append(summary)

            if random.random() < (save_samples / num_samples):
                sum_examples.append({
                    "reference_summary": summary,
                    "prediction": sum_pred,
                    "doc_excerpt": document[:200] + "..."
                })

    # Metrics
    avg_f1 = float(np.mean(qa_f1s) * 100) if qa_f1s else 0.0
    rouge_scores = compute_rouge(sum_predictions, sum_references) if sum_predictions else {}

    print(f"\nQA Avg F1: {avg_f1:.2f}%")
    print(f"Summarization ROUGE: {rouge_scores}")

    return {
        "qa": {"f1": avg_f1, "examples": qa_examples},
        "summarization": {"rouge": rouge_scores, "examples": sum_examples}
    }

# -----------------------------------------------------------
# 3. Run baseline
# -----------------------------------------------------------

if __name__ == "__main__":
    results = evaluate_baseline(num_samples=50, save_samples=8)
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n✅ Saved results + sample predictions to baseline_results.json")
