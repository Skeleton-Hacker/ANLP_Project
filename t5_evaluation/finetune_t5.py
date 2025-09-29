import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from accelerate import Accelerator
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# 0. Configuration
# -----------------------------------------------------------
class Config:
    MODEL_NAME = "t5-small"
    DATASET_NAME = "deepmind/narrativeqa"
    
    # Training
    TRAIN_BATCH_SIZE = 4 # Reduced for larger model and input size
    VALID_BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 5e-5 # Standard for fine-tuning
    
    # Data
    MAX_INPUT_LENGTH = 1024 # Increased to capture more context
    MAX_TARGET_LENGTH = 128
    TRAIN_SAMPLES = 10000  # Using a larger subset for better training
    VALID_SAMPLES = 1000
    TEST_SAMPLES = 1000
    
    # Processing Optimization
    NUM_PROC = 32  # Use most of your 40 CPUs for preprocessing
    CACHE_DIR = "./cache"  # Directory to save preprocessed datasets
    FORCE_REPROCESS = False  # Set to True to ignore cached data
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 2
    
    # Paths
    OUTPUT_DIR = "t5-narrativeqa-finetuned"
    PLOTS_DIR = "plots"

# -----------------------------------------------------------
# 1. Metrics
# -----------------------------------------------------------

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
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
# 2. Data Loading and Preprocessing
# -----------------------------------------------------------

def clean_text(text):
    """Removes boilerplate, HTML, and extra whitespace. Optimized version."""
    if not text:
        return ""
    
    # Combine regex patterns for better performance
    patterns = [
        (r'ï»¿The Project Gutenberg.*', ''),
        (r'<html>.*</html>', ''),
        (r'<[^>]+>', ''),
        (r'\s+', ' ')
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.DOTALL)
    
    return text.strip()

def preprocess_function(examples, tokenizer):
    # Batch process all examples at once for efficiency
    docs = [clean_text(doc["text"]) for doc in examples["document"]]
    questions = [q["text"] for q in examples["question"]]
    answers = [ans[0]["text"] for ans in examples["answers"]]  # Use first answer
    
    # Create input format in batch
    inputs = [f"question: {q} context: {d}" for q, d in zip(questions, docs)]
    
    # Tokenize in batch - much faster
    model_inputs = tokenizer(
        inputs, 
        max_length=Config.MAX_INPUT_LENGTH, 
        truncation=True, 
        padding="max_length",
        return_tensors=None  # Return lists instead of tensors for better memory usage
    )
    
    # Tokenize labels in batch
    labels = tokenizer(
        text_target=answers, 
        max_length=Config.MAX_TARGET_LENGTH, 
        truncation=True, 
        padding="max_length",
        return_tensors=None
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def create_dataloaders(tokenizer):
    import hashlib
    import pickle
    from pathlib import Path
    
    # Create cache directory
    cache_dir = Path(Config.CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    
    # Create a cache key based on configuration
    cache_key = hashlib.md5(
        f"{Config.MODEL_NAME}_{Config.TRAIN_SAMPLES}_{Config.VALID_SAMPLES}_{Config.MAX_INPUT_LENGTH}_{Config.MAX_TARGET_LENGTH}".encode()
    ).hexdigest()
    
    train_cache_path = cache_dir / f"train_{cache_key}.pkl"
    valid_cache_path = cache_dir / f"valid_{cache_key}.pkl"
    test_cache_path = cache_dir / f"test_{cache_key}.pkl"
    
    # Check if cached datasets exist and should be used
    if (not Config.FORCE_REPROCESS and 
        train_cache_path.exists() and 
        valid_cache_path.exists() and 
        test_cache_path.exists()):
        
        print("🚀 Loading cached preprocessed datasets...")
        with open(train_cache_path, 'rb') as f:
            processed_train = pickle.load(f)
        with open(valid_cache_path, 'rb') as f:
            processed_valid = pickle.load(f)
        with open(test_cache_path, 'rb') as f:
            test_dataset = pickle.load(f)
        print("✅ Cached datasets loaded successfully!")
    else:
        print("📊 Loading and preprocessing datasets...")
        full_dataset = load_dataset(Config.DATASET_NAME, split='train')
        
        # Create a smaller subset for faster processing if needed
        total_samples = Config.TRAIN_SAMPLES + Config.VALID_SAMPLES + Config.TEST_SAMPLES
        subset_indices = list(range(min(total_samples, len(full_dataset))))
        subset = full_dataset.select(subset_indices)

        # Split dataset
        train_val_indices, test_indices = train_test_split(
            subset_indices, test_size=Config.TEST_SAMPLES, random_state=42
        )
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=Config.VALID_SAMPLES, random_state=42
        )

        train_dataset = full_dataset.select(train_indices)
        valid_dataset = full_dataset.select(val_indices)
        test_dataset = full_dataset.select(test_indices)

        print(f"🔄 Preprocessing training data ({len(train_dataset)} samples) using {Config.NUM_PROC} processes...")
        processed_train = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer), 
            batched=True, 
            batch_size=100,  # Process in smaller batches for better memory usage
            num_proc=Config.NUM_PROC,  # Use multiple processes
            remove_columns=train_dataset.column_names,
            desc="Processing training data"
        )
        
        print(f"🔄 Preprocessing validation data ({len(valid_dataset)} samples) using {Config.NUM_PROC} processes...")
        processed_valid = valid_dataset.map(
            lambda x: preprocess_function(x, tokenizer), 
            batched=True, 
            batch_size=100,
            num_proc=Config.NUM_PROC,
            remove_columns=valid_dataset.column_names,
            desc="Processing validation data"
        )
        
        # Save preprocessed datasets to cache
        print("💾 Saving preprocessed datasets to cache...")
        with open(train_cache_path, 'wb') as f:
            pickle.dump(processed_train, f)
        with open(valid_cache_path, 'wb') as f:
            pickle.dump(processed_valid, f)
        with open(test_cache_path, 'wb') as f:
            pickle.dump(test_dataset, f)
        print(f"✅ Cached datasets saved to {cache_dir}")
    
    processed_train.set_format("torch")
    processed_valid.set_format("torch")

    train_dataloader = DataLoader(
        processed_train, 
        shuffle=True, 
        batch_size=Config.TRAIN_BATCH_SIZE,
        num_workers=4,  # Add some parallel data loading
        pin_memory=True
    )
    valid_dataloader = DataLoader(
        processed_valid, 
        batch_size=Config.VALID_BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )
    
    return train_dataloader, valid_dataloader, test_dataset

# -----------------------------------------------------------
# 3. Training and Evaluation Loop
# -----------------------------------------------------------

def train_epoch(model, dataloader, optimizer, lr_scheduler, accelerator):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", disable=not accelerator.is_local_main_process):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, tokenizer, accelerator):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=Config.MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True
            )
            
            generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            # Decode before gathering to avoid issues with padding and extra tokens
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_tokens]
            labels_text = [tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True) for l in labels]

            # Gather decoded text from all processes
            gathered_preds, gathered_labels = accelerator.gather_for_metrics((preds, labels_text))

            all_preds.extend(gathered_preds)
            all_labels.extend(gathered_labels)

    # Ensure metrics are computed only on the main process after gathering
    if accelerator.is_main_process:
        f1 = np.mean([compute_f1(p, l) for p, l in zip(all_preds, all_labels)]) * 100
        rouge = compute_rouge(all_preds, all_labels)
    else:
        f1, rouge = 0.0, {}
    
    return total_loss / len(dataloader), f1, rouge

# -----------------------------------------------------------
# 4. Plotting
# -----------------------------------------------------------

def plot_metrics(metrics, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["valid_loss"], label="Validation Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.title("Loss")
    plt.legend()
    
    # F1
    plt.subplot(1, 3, 2)
    plt.plot(metrics["valid_f1"], label="Validation F1", color='green')
    plt.xlabel("Epochs"); plt.ylabel("F1 Score (%)"); plt.title("Validation F1")
    plt.legend()

    # ROUGE
    plt.subplot(1, 3, 3)
    rouge_scores = metrics["valid_rouge"]
    for k in rouge_scores[0].keys():
        plt.plot([r[k] for r in rouge_scores], label=f"Val {k}")
    plt.xlabel("Epochs"); plt.ylabel("ROUGE"); plt.title("Validation ROUGE")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "finetuning_metrics.png"))
    print(f"✅ Plots saved to {plots_dir}")

# -----------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------

def main():
    accelerator = Accelerator()
    
    # Print configuration for debugging
    if accelerator.is_main_process:
        print(f"🔧 Configuration:")
        print(f"   - CPUs for preprocessing: {Config.NUM_PROC}")
        print(f"   - Cache directory: {Config.CACHE_DIR}")
        print(f"   - Force reprocess: {Config.FORCE_REPROCESS}")
        print(f"   - Train samples: {Config.TRAIN_SAMPLES}")
        print(f"   - Valid samples: {Config.VALID_SAMPLES}")
        print(f"   - Max input length: {Config.MAX_INPUT_LENGTH}")
    
    tokenizer = T5Tokenizer.from_pretrained(Config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(Config.MODEL_NAME)
    
    train_dataloader, valid_dataloader, _ = create_dataloaders(tokenizer)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    num_training_steps = Config.EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )
    
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    metrics = {"train_loss": [], "valid_loss": [], "valid_f1": [], "valid_rouge": []}

    for epoch in range(Config.EPOCHS):
        accelerator.print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
        
        train_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, accelerator)
        valid_loss, valid_f1, valid_rouge = evaluate(model, valid_dataloader, tokenizer, accelerator)
        
        metrics["train_loss"].append(train_loss)
        metrics["valid_loss"].append(valid_loss)
        metrics["valid_f1"].append(valid_f1)
        metrics["valid_rouge"].append(valid_rouge)
        
        accelerator.print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid F1: {valid_f1:.2f}%")
        accelerator.print(f"Valid ROUGE: {valid_rouge}")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            if accelerator.is_main_process:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(Config.OUTPUT_DIR)
                tokenizer.save_pretrained(Config.OUTPUT_DIR)
                accelerator.print(f"✅ Best model saved to {Config.OUTPUT_DIR}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
                accelerator.print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
                break

    if accelerator.is_main_process:
        plot_metrics(metrics, Config.PLOTS_DIR)

if __name__ == "__main__":
    # To run:
    # 1. Configure your accelerator: `accelerate config`
    # 2. Launch the script: `accelerate launch t5_evaluation/finetune_t5.py`
    main()



