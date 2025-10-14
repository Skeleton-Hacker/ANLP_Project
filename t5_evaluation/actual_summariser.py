import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import json

# -----------------------------------------------------------
# 1. Metrics
# -----------------------------------------------------------

def compute_metrics(predictions, references, device):
    """
    Compute ROUGE and BERT scores for a list of predictions and references.
    Uses multiple GPUs if available: ROUGE on GPU 0, BERT on GPU 1.
    """
    scores = {}
    
    # Determine devices for metric computation
    rouge_device = "cpu"  # ROUGE is CPU-based anyway
    bert_device = device  # Default to the main device
    
    # If multiple GPUs are available, use GPU 1 for BERT score
    if torch.cuda.device_count() > 1:
        bert_device = "cuda:1"
        print(f"Using GPU 1 for BERT score computation")
    
    print(f"Computing ROUGE scores on {rouge_device}, BERT scores on {bert_device}")

    # ROUGE scores (CPU-based)
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        print("Computing ROUGE scores...")
        for pred, ref in tqdm(zip(predictions, references), desc="ROUGE", total=len(predictions)):
            result = scorer.score(ref, pred)
            for k in rouge_scores:
                rouge_scores[k].append(result[k].fmeasure)
        scores.update({k: float(np.mean(v) * 100) for k, v in rouge_scores.items()})
        print(f"ROUGE scores computed: R-1: {scores['rouge1']:.2f}, R-2: {scores['rouge2']:.2f}, R-L: {scores['rougeL']:.2f}")
    except ImportError:
        print("⚠️ Install rouge-score with: pip install rouge-score")
        scores.update({"rouge1": 0, "rouge2": 0, "rougeL": 0})

    # BERTScore (GPU-based if available)
    try:
        from bert_score import score as bert_scorer
        print("Computing BERT scores...")
        P, R, F1 = bert_scorer(predictions, references, lang="en", device=bert_device, verbose=False)
        scores["bert_score"] = F1.mean().item() * 100
        print(f"BERT score computed: {scores['bert_score']:.2f}")
    except ImportError:
        print("⚠️ Install bert-score with: pip install bert-score")
        scores["bert_score"] = 0
    except Exception as e:
        print(f"⚠️ Error computing BERT score: {e}")
        scores["bert_score"] = 0
        
    return scores

# -----------------------------------------------------------
# 2. Summarization Model Evaluation
# -----------------------------------------------------------

def evaluate_summarization_model(device, model_name="allenai/led-large-16384-arxiv", num_samples=10, input_max_length=1024):
    """
    Evaluates a dedicated summarization model on the NarrativeQA dataset for a specific input length.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True).to(device)

    # Shuffle with a fixed seed to use the same random subset for each run
    dataset = load_dataset("deepmind/narrativeqa", split="validation", trust_remote_code=True).shuffle(seed=42).select(range(num_samples))
    
    predictions, references = [], []

    progress_bar = tqdm(dataset, desc=f"Evaluating (len: {input_max_length}) on {device}")

    for sample in progress_bar:
        document = sample.get("document", {}).get("text")
        summary = sample.get("document", {}).get("summary", {}).get("text")

        if not document or not document.strip() or not summary or not summary.strip():
            continue

        inputs = tokenizer(
            document, 
            max_length=input_max_length,  
            return_tensors="pt", 
            truncation=True
        ).to(device)

        with torch.no_grad():
            # Increased max_length for the generated summary
            summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=256, early_stopping=True)
        
        pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        predictions.append(pred_summary)
        references.append(summary)

    # Compute Metrics
    all_scores = compute_metrics(predictions, references, device) if predictions else {}
    
    return all_scores

# -----------------------------------------------------------
# 3. Run Evaluation Sweep
# -----------------------------------------------------------

if __name__ == "__main__":
    NUM_SAMPLES = 100
    MODEL_NAME = "allenai/led-large-16384-arxiv"
    ALL_INPUT_LENGTHS = [1024, 2048, 4096, 8192, 16384]
    
    # Use GPU if available, otherwise CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Total GPUs available: {num_gpus}")
    if num_gpus > 1:
        print("Will use GPU 0 for model inference and GPU 1 for BERT score computation")
    
    results_summary = []
    
    for length in ALL_INPUT_LENGTHS:
        print(f"\nEvaluating with input length: {length}")
        scores = evaluate_summarization_model(
            device=device,
            model_name=MODEL_NAME,
            num_samples=NUM_SAMPLES,
            input_max_length=length
        )
        
        results_summary.append({
            "input_length": length,
            "rouge1": scores.get("rouge1", 0),
            "rouge2": scores.get("rouge2", 0),
            "rougeL": scores.get("rougeL", 0),
            "bert_score": scores.get("bert_score", 0),
        })

    # Save results to JSON file
    output_file = "summarization_results_led.json"
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")

    # Print the final summary
    print("\n" + "="*100)
    print("           PERFORMANCE SWEEP SUMMARY")
    print("="*100)
    print(f"{'Input Length':<15} | {'ROUGE-1':<15} | {'ROUGE-2':<15} | {'ROUGE-L':<15} | {'BERT Score':<15}")
    print("-" * 100)
    for result in results_summary:
        print(f"{result['input_length']:<15} | {result['rouge1']:<15.4f} | {result['rouge2']:<15.4f} | {result['rougeL']:<15.4f} | {result['bert_score']:<15.4f}")
    print("-" * 100)
    
    # Print some analysis
    print("\n📊 ANALYSIS:")
    best_rouge1 = max(results_summary, key=lambda x: x['rouge1'])
    best_rouge2 = max(results_summary, key=lambda x: x['rouge2'])
    best_rougeL = max(results_summary, key=lambda x: x['rougeL'])
    best_bert = max(results_summary, key=lambda x: x['bert_score'])
    
    print(f"Best ROUGE-1: {best_rouge1['rouge1']:.4f} at length {best_rouge1['input_length']}")
    print(f"Best ROUGE-2: {best_rouge2['rouge2']:.4f} at length {best_rouge2['input_length']}")
    print(f"Best ROUGE-L: {best_rougeL['rougeL']:.4f} at length {best_rougeL['input_length']}")
    print(f"Best BERT Score: {best_bert['bert_score']:.4f} at length {best_bert['input_length']}")
    print("="*100)
