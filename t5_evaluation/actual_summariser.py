import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import json
import time
from queue import Queue
from threading import Thread

# -----------------------------------------------------------
# 1. Metrics
# -----------------------------------------------------------

def compute_metrics(predictions, references, device):
    """
    Compute ROUGE and BERT scores for a list of predictions and references.
    """
    scores = {}

    # ROUGE scores
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
        from bert_score import score as bert_scorer
        P, R, F1 = bert_scorer(predictions, references, lang="en", device=device, verbose=False)
        scores["bert_score"] = F1.mean().item() * 100
    except ImportError:
        print("⚠️ Install bert-score with: pip install bert-score")
        scores["bert_score"] = 0
        
    return scores

# -----------------------------------------------------------
# 2. Summarization Model Evaluation
# -----------------------------------------------------------

def evaluate_summarization_model(device, model_name="allenai/led-large-16384-arxiv", num_samples=10, input_max_length=1024, is_main_process=False):
    """
    Evaluates a dedicated summarization model on the NarrativeQA dataset for a specific input length.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True).to(device)

    # Shuffle with a fixed seed to use the same random subset for each run
    dataset = load_dataset("deepmind/narrativeqa", split="validation", trust_remote_code=True).shuffle(seed=42).select(range(num_samples))
    
    predictions, references = [], []

    # Use tqdm only on the main process
    progress_bar = tqdm(dataset, desc=f"Evaluating (len: {input_max_length}) on {device}", disable=not is_main_process)

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

def worker(gpu_id, task_queue, result_queue, model_name, num_samples):
    """
    A worker process that continuously fetches tasks from the queue and evaluates the model.
    """
    device = f"cuda:{gpu_id}"
    print(f"Worker on GPU {gpu_id} started.")
    
    while True:
        input_length = task_queue.get()
        if input_length is None:  # Sentinel value to stop the worker
            break
            
        print(f"GPU {gpu_id}: Starting evaluation for input length {input_length}...")
        
        scores = evaluate_summarization_model(
            device=device,
            model_name=model_name,
            num_samples=num_samples,
            input_max_length=input_length,
            is_main_process=(gpu_id == 0)  # Show progress bar only for the first GPU
        )
        
        result_queue.put({
            "input_length": input_length,
            "rouge1": scores.get("rouge1", 0),
            "rouge2": scores.get("rouge2", 0),
            "rougeL": scores.get("rougeL", 0),
            "bert_score": scores.get("bert_score", 0),
            "gpu_id": gpu_id
        })
        print(f"GPU {gpu_id}: Finished evaluation for input length {input_length}.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    NUM_SAMPLES = 100
    MODEL_NAME = "allenai/led-large-16384-arxiv"
    ALL_INPUT_LENGTHS = [1024, 2048, 4096, 8192, 16384]
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. Running on CPU.")
        # Simplified CPU execution
        results_summary = []
        for length in ALL_INPUT_LENGTHS:
            scores = evaluate_summarization_model(
                device="cpu",
                model_name=MODEL_NAME,
                num_samples=NUM_SAMPLES,
                input_max_length=length,
                is_main_process=True
            )
            results_summary.append({
                "input_length": length,
                "rouge1": scores.get("rouge1", 0),
                "rouge2": scores.get("rouge2", 0),
                "rougeL": scores.get("rougeL", 0),
                "bert_score": scores.get("bert_score", 0),
            })
    else:
        print(f"Found {num_gpus} GPUs.")
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        # Initially, assign one task to each GPU
        for i in range(min(num_gpus, len(ALL_INPUT_LENGTHS))):
            task_queue.put(ALL_INPUT_LENGTHS[i])
        
        remaining_tasks = ALL_INPUT_LENGTHS[num_gpus:]

        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=worker, args=(i, task_queue, result_queue, MODEL_NAME, NUM_SAMPLES))
            p.start()
            processes.append(p)

        results_summary = []
        
        # As results come in, assign new tasks
        for _ in range(len(ALL_INPUT_LENGTHS)):
            result = result_queue.get()
            results_summary.append(result)
            print(f"Main: Received result for length {result['input_length']} from GPU {result['gpu_id']}.")

            if remaining_tasks:
                next_task = remaining_tasks.pop(0)
                print(f"Main: Assigning new task (length: {next_task}) to a free worker.")
                task_queue.put(next_task)

        # Stop all worker processes
        for _ in range(num_gpus):
            task_queue.put(None)

        for p in processes:
            p.join()

    # Sort and print the final summary
    results_summary.sort(key=lambda x: x['input_length'])

    print("\n" + "="*100)
    print("           PERFORMANCE SWEEP SUMMARY")
    print("="*100)
    print(f"{'Input Length':<15} | {'ROUGE-1':<15} | {'ROUGE-2':<15} | {'ROUGE-L':<15} | {'BERT Score':<15}")
    print("-" * 100)
    for result in results_summary:
        print(f"{result['input_length']:<15} | {result['rouge1']:<15.4f} | {result['rouge2']:<15.4f} | {result['rougeL']:<15.4f} | {result.get('bert_score', 0):<15.4f}")
    print("-" * 100)
