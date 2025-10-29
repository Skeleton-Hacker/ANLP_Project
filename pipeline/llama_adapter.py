"""
Memory-Optimized Llama 1B Adapter for Fine-tuning with Chunk Embeddings

Key optimizations:
- FP16/BF16 mixed precision training
- Gradient checkpointing
- Frozen Llama base model (only adapter trainable)
- Reduced batch/chunk sizes (batch_size=4, max_chunks=256)
- Frequent cache clearing during training
- BERTScore on CPU to avoid GPU memory bloat
- Evaluate every 5 epochs instead of every epoch
"""

import logging
import pickle
import os
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
import json
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import argparse

# Setup logging
logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class LlamaAdapterConfig:
    """Configuration for Llama adapter training."""
    # Data settings
    encoded_data_dir: str = "chunked_data"
    output_dir: str = "models/llama_adapter"
    logs_dir: str = "logs"
    
    # Model settings
    llama_model_name: str = "meta-llama/Llama-3.2-1B"
    encoded_dim: int = 64
    adapter_hidden_dim: int = 512
    
    # MEMORY OPTIMIZATION: Reduced settings
    batch_size: int = 12
    num_epochs: int = 300
    learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 32  # Increased from 16 to compensate
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    
    # MEMORY OPTIMIZATION: Reduced chunk limit
    max_chunks_per_story: int = 300
    max_target_length: int = 150
    min_target_length: int = 20
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    
    # Hardware settings
    num_workers: int = 4
    
    # Evaluation settings
    eval_every_n_steps: int = 1000
    eval_every_n_epochs: int = 1  
    use_gradient_checkpointing: bool = True
    
    # MEMORY OPTIMIZATION: Freeze Llama base model
    freeze_llama: bool = True
    
    # Random seed
    seed: int = 42
    # Generation: use beam search by default to improve coherence
    num_beams: int = 3
    # Enable verbose debugging prints/checks during training (optional)
    debug: bool = False


class ChunkEncodingDataset(Dataset):
    """Dataset for chunk encodings with summarization targets."""
    
    def __init__(
        self,
        encoded_data_path: Path,
        tokenizer: AutoTokenizer,
        max_chunks: int = 200,
        max_target_length: int = 150
    ):
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.max_target_length = max_target_length
        
        logger.info(f"Loading encoded data from {encoded_data_path}...")
        with open(encoded_data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.stories = data['stories']
        self.metadata = data['metadata']
        
        self.samples = []
        for story_id, story_data in self.stories.items():
            if 'document' in story_data and 'summary' in story_data['document'] and 'text' in story_data['document']['summary']:
                self.samples.append(story_id)
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.stories)} stories")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        story_id = self.samples[idx]
        story_data = self.stories[story_id]
        
        encoded_embeddings = story_data['encoded_embeddings']
        
        # Truncate to max_chunks
        if len(encoded_embeddings) > self.max_chunks:
            encoded_embeddings = encoded_embeddings[:self.max_chunks]
        
        # Convert to bfloat16 to match adapter/model dtype and avoid matmul dtype errors
        chunk_encodings = torch.FloatTensor(encoded_embeddings).to(torch.bfloat16)
        target_text = story_data['document']['summary']['text']
        
        prompt = "Summarize the following document. Give in paragraph format:\n\n"
        full_text = prompt + target_text # adding the prompt again since it is not listenting to it at all
        
        prompt_encoding = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_target_length + len(prompt_encoding['input_ids'][0]),
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chunk_encodings': chunk_encodings,
            'num_chunks': len(encoded_embeddings),
            'input_ids': full_encoding['input_ids'].squeeze(0),
            'attention_mask': full_encoding['attention_mask'].squeeze(0),
            'prompt_length': len(prompt_encoding['input_ids'][0]),
            'target_text': target_text,
            'document_text': ' '.join(story_data['chunks'])
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length chunk sequences."""
    max_num_chunks = max(item['num_chunks'] for item in batch)
    encoded_dim = batch[0]['chunk_encodings'].shape[1]
    
    padded_chunk_encodings = []
    chunk_attention_masks = []
    
    for item in batch:
        num_chunks = item['num_chunks']
        chunk_encodings = item['chunk_encodings']
        
        if num_chunks < max_num_chunks:
            padding = torch.zeros(max_num_chunks - num_chunks, encoded_dim)
            padded_encodings = torch.cat([chunk_encodings, padding], dim=0)
            attention_mask = torch.cat([
                torch.ones(num_chunks),
                torch.zeros(max_num_chunks - num_chunks)
            ])
        else:
            padded_encodings = chunk_encodings
            attention_mask = torch.ones(num_chunks)
        
        padded_chunk_encodings.append(padded_encodings)
        chunk_attention_masks.append(attention_mask)
    
    return {
        'chunk_encodings': torch.stack(padded_chunk_encodings),
        'chunk_attention_mask': torch.stack(chunk_attention_masks),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'prompt_lengths': torch.tensor([item['prompt_length'] for item in batch]),
        'target_texts': [item['target_text'] for item in batch],
        'document_texts': [item['document_text'] for item in batch]
    }


class LlamaChunkAdapter(nn.Module):
    """Llama model with adapter layer for chunk encodings."""
    
    def __init__(
        self,
        llama_model_name: str,
        encoded_dim: int,
        adapter_hidden_dim: int,
        use_gradient_checkpointing: bool = True,
        freeze_llama: bool = True
    ):
        super(LlamaChunkAdapter, self).__init__()
        
        # MEMORY OPTIMIZATION: Load in bfloat16
        self.llama = AutoModelForCausalLM.from_pretrained(
            llama_model_name,
            torch_dtype=torch.bfloat16,  # Changed from float32
            use_cache=False
        )
        self.llama_hidden_dim = self.llama.config.hidden_size
        
        # MEMORY OPTIMIZATION: Enable gradient checkpointing
        if use_gradient_checkpointing:
            self.llama.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # MEMORY OPTIMIZATION: Freeze Llama parameters
        if freeze_llama:
            for param in self.llama.parameters():
                param.requires_grad = False
            logger.info("Llama base model frozen")
        
        # Adapter layer (always trainable)
        self.adapter = nn.Sequential(
            nn.Linear(encoded_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_dim, self.llama_hidden_dim),
            nn.LayerNorm(self.llama_hidden_dim)
        )
        
        # Ensure adapter is in same dtype as llama
        self.adapter = self.adapter.to(torch.bfloat16)
        
        logger.info(f"Initialized LlamaChunkAdapter with {llama_model_name}")
        logger.info(f"  - Llama hidden dim: {self.llama_hidden_dim}")
        logger.info(f"  - Encoded dim: {encoded_dim}")
        logger.info(f"  - Adapter hidden dim: {adapter_hidden_dim}")
        logger.info(f"  - Dtype: bfloat16")
        logger.info(f"  - Llama frozen: {freeze_llama}")
    
    def forward(
        self,
        chunk_encodings: torch.Tensor,
        chunk_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor
    ):
        batch_size = chunk_encodings.shape[0]
        
        # Convert to bfloat16
        chunk_encodings = chunk_encodings.to(torch.bfloat16)
        
        # Project chunk encodings
        chunk_embeds = self.adapter(chunk_encodings)
        
        # Get token embeddings
        text_embeds = self.llama.model.embed_tokens(input_ids)
        
        # Concatenate
        combined_embeds = torch.cat([chunk_embeds, text_embeds], dim=1)
        combined_attention_mask = torch.cat([chunk_attention_mask, attention_mask], dim=1)
        
        # Forward through Llama
        outputs = self.llama(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            use_cache=False
        )
        
        logits = outputs.logits
        num_chunks = chunk_encodings.shape[1]
        
        # Extract text logits
        text_logits = logits[:, num_chunks:, :]
        
        # Shift for next token prediction
        shift_logits = text_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_attention = attention_mask[:, 1:].contiguous()
        
        # Create loss mask (vectorized)
        # shift_labels correspond to original token positions 1..seq_len-1
        seq_len = shift_labels.size(1)
        # positions (0..seq_len-1) correspond to original positions (1..seq_len)
        pos_ids = torch.arange(seq_len, device=prompt_lengths.device).unsqueeze(0)
        # prompt_lengths has original token counts; include label positions where (pos+1) >= prompt_len
        prompt_lens = prompt_lengths.unsqueeze(1)
        loss_mask = shift_attention.bool() & ((pos_ids + 1) >= prompt_lens)
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        loss = loss.view(shift_labels.shape)
        
        # Apply mask
        masked_loss = loss * loss_mask.float()
        loss = masked_loss.sum() / loss_mask.float().sum().clamp(min=1.0)
        
        return type('Output', (), {'loss': loss, 'logits': logits})()
    
    def generate(
        self,
        chunk_encodings: torch.Tensor,
        chunk_attention_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        max_length: int = 150,
        min_length: int = 20,
        temperature: float = 0.3,
        top_p: float = 0.85,
        repetition_penalty: float = 1.1,
        num_beams: int = 3,
        tokenizer: AutoTokenizer = None
    ):
        # Convert to bfloat16
        chunk_encodings = chunk_encodings.to(torch.bfloat16)
        
        # Project chunks
        chunk_embeds = self.adapter(chunk_encodings)
        prompt_embeds = self.llama.model.embed_tokens(prompt_ids)
        
        # Concatenate
        combined_embeds = torch.cat([chunk_embeds, prompt_embeds], dim=1)
        combined_attention_mask = torch.cat([chunk_attention_mask, prompt_attention_mask], dim=1)
        
        # Determine sampling vs beam search: if num_beams > 1 use beam search (deterministic)
        do_sample_flag = True if num_beams == 1 else False

        outputs = self.llama.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            max_new_tokens=max_length,
            min_new_tokens=min_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample_flag,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id if tokenizer else None,
            eos_token_id=tokenizer.eos_token_id if tokenizer else None,
            use_cache=True
        )
        
        return outputs


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE and BERTScore metrics."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    rouge_results = {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL'])
    }
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    P, R, F1 = bert_score(predictions, references, lang='en', verbose=False, device=device)
    bert_results = {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }
    
    return {**rouge_results, **bert_results}


def debug_data_sample(batch: Dict[str, Any], tokenizer: AutoTokenizer):
    """Print a compact debug view of a batch to verify tokenization and targets."""
    try:
        logger.info("--- DEBUG DATA SAMPLE ---")
        logger.info(f"Chunk encodings shape: {batch['chunk_encodings'].shape}")
        input_ids = batch['input_ids'][0]
        logger.info(f"Input IDs sample (first 30): {input_ids[:30]}")
        decoded_text = tokenizer.decode(input_ids.cpu(), skip_special_tokens=True)
        logger.info(f"Decoded text (first 200 chars): {decoded_text[:200]!r}")
        target = batch.get('target_texts', None) or batch.get('target_text', None)
        if isinstance(target, (list, tuple)):
            target = target[0]
        logger.info(f"Target text (first 200 chars): {str(target)[:200]!r}")
    except Exception as e:
        logger.exception(f"Error while debugging data sample: {e}")


def debug_adapter_output(model: nn.Module, batch: Dict[str, Any]):
    """Run adapter on a batch and log basic statistics to detect NaNs/zeros/outliers."""
    try:
        logger.info("--- DEBUG ADAPTER OUTPUT ---")
        chunk_enc = batch['chunk_encodings']
        
        # Get the device and dtype from the adapter
        device = next(model.adapter.parameters()).device
        dtype = next(model.adapter.parameters()).dtype
        
        # Move to same device and convert to correct dtype
        chunk_enc = chunk_enc.to(device=device, dtype=dtype)
        
        with torch.no_grad():
            out = model.adapter(chunk_enc)
        
        # Convert to float32 for statistics (bfloat16 can have precision issues)
        out_f = out.to(torch.float32)
        
        logger.info(f"Adapter output shape: {out_f.shape}")
        logger.info(f"Adapter output - min: {out_f.min().item():.6f}, max: {out_f.max().item():.6f}")
        logger.info(f"Adapter output - mean: {out_f.mean().item():.6f}, std: {out_f.std().item():.6f}")
        
        # Check for NaNs or extreme values
        if torch.isnan(out_f).any():
            logger.error("WARNING: NaNs detected in adapter output!")
        if torch.isinf(out_f).any():
            logger.error("WARNING: Infs detected in adapter output!")
        if out_f.abs().max() > 1000:
            logger.warning("WARNING: Large values detected in adapter output!")
            
    except Exception as e:
        logger.exception(f"Error while debugging adapter output: {e}")


def evaluate_model(
    model: LlamaChunkAdapter,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    config: LlamaAdapterConfig,
    accelerator: Accelerator,
    desc: str = "Evaluation"
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Evaluate model on a dataset."""
    model.eval()
    all_predictions = []
    all_references = []
    
    eval_pbar = tqdm(dataloader, desc=desc, disable=not accelerator.is_main_process)
    
    prompt = "Summarize the following document. Give in paragraph format:\n\n"
    prompt_encoding = tokenizer(prompt, add_special_tokens=True, return_tensors='pt')
    
    with torch.no_grad():
        for batch in eval_pbar:
            chunk_encodings = batch['chunk_encodings'].to(accelerator.device)
            chunk_attention_mask = batch['chunk_attention_mask'].to(accelerator.device)
            target_texts = batch['target_texts']
            batch_size = chunk_encodings.shape[0]
            
            prompt_ids = prompt_encoding['input_ids'].repeat(batch_size, 1).to(accelerator.device)
            prompt_attention_mask = prompt_encoding['attention_mask'].repeat(batch_size, 1).to(accelerator.device)
            
            generated_ids = model.generate(
                chunk_encodings=chunk_encodings,
                chunk_attention_mask=chunk_attention_mask,
                prompt_ids=prompt_ids,
                prompt_attention_mask=prompt_attention_mask,
                max_length=config.max_target_length,
                min_length=config.min_target_length,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        repetition_penalty=config.repetition_penalty,
                        num_beams=getattr(config, 'num_beams', 1),
                tokenizer=tokenizer
            )
            
            num_prefix_tokens = prompt_ids.shape[1]
            generated_ids = generated_ids[:, num_prefix_tokens:]
            
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(target_texts)
            
            # MEMORY OPTIMIZATION: Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    metrics = compute_metrics(all_predictions, all_references)
    
    return metrics, all_predictions, all_references


def train_model(
    model: LlamaChunkAdapter,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    config: LlamaAdapterConfig,
    accelerator: Accelerator,
    log_file: Path
) -> LlamaChunkAdapter:
    """Train the Llama adapter model."""
    # MEMORY OPTIMIZATION: Only optimize adapter parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW(
        trainable_params,  # Only adapter params
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    # Optional debug: show a single data sample and adapter stats
    if config.debug and accelerator.is_main_process:
        try:
            sample_batch = next(iter(train_dataloader))
            # Run tokenization / sample debug on CPU-friendly data
            debug_data_sample(sample_batch, tokenizer)
            debug_adapter_output(accelerator.unwrap_model(model), sample_batch)
        except Exception:
            logger.exception("Failed to run debug checks on a training batch.")
    
    best_val_loss = float('inf')
    best_val_rouge = 0.0
    patience_counter = 0
    global_step = 0
    
    if accelerator.is_main_process:
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Training started at {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Configuration:\n{json.dumps(config.__dict__, indent=2)}\n")
            f.write(f"{'='*80}\n\n")
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        train_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]",
            disable=not accelerator.is_main_process
        )
        
        for batch_idx, batch in enumerate(train_pbar):
            model.zero_grad(set_to_none=True)
            outputs = model(
                chunk_encodings=batch['chunk_encodings'],
                chunk_attention_mask=batch['chunk_attention_mask'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                prompt_lengths=batch['prompt_lengths']
            )
            
            loss = outputs.loss / config.gradient_accumulation_steps
            accelerator.backward(loss)

            
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(trainable_params, config.max_grad_norm)
                # Optional debug: log gradient norms for trainable parameters
                if config.debug and accelerator.is_main_process:
                    try:
                        for name, param in model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                gnorm = param.grad.norm().item()
                                logger.info(f"Grad norm - {name}: {gnorm:.6f}")
                    except Exception:
                        logger.exception("Error when logging gradient norms")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # MEMORY OPTIMIZATION: Clear cache more frequently
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection
                if global_step % 5 == 0:
                    gc.collect()
            
            train_loss += loss.item() * config.gradient_accumulation_steps
            num_train_batches += 1

             # AGGRESSIVE MEMORY CLEANUP
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection every batch
            gc.collect()
            
            train_pbar.set_postfix({
                'loss': loss.item() * config.gradient_accumulation_steps,
                'lr': scheduler.get_last_lr()[0]
            })
        
        avg_train_loss = train_loss / num_train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        val_pbar = tqdm(
            val_dataloader,
            desc=f"Epoch {epoch+1}/{config.num_epochs} [Val Loss]",
            disable=not accelerator.is_main_process
        )
        
        with torch.no_grad():
            for batch in val_pbar:
                outputs = model(
                    chunk_encodings=batch['chunk_encodings'],
                    chunk_attention_mask=batch['chunk_attention_mask'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    prompt_lengths=batch['prompt_lengths']
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                num_val_batches += 1
                
                val_pbar.set_postfix({'val_loss': loss.item()})
        
        avg_val_loss = val_loss / num_val_batches
        
        # Evaluate with metrics every N epochs
        should_evaluate = (epoch + 1) % config.eval_every_n_epochs == 0
        if should_evaluate:
            if accelerator.is_main_process:
                logger.info(f"\nEvaluating with ROUGE and BERTScore...")
            
            val_metrics, val_predictions, val_references = evaluate_model(
                accelerator.unwrap_model(model),
                val_dataloader,
                tokenizer,
                config,
                accelerator,
                desc=f"Epoch {epoch+1} [Val Metrics]"
            )
            
            # Aggressive memory cleanup after evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        else:
            # Use dummy metrics when not evaluating
            val_metrics = {
                'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
                'bertscore_f1': 0.0, 'bertscore_precision': 0.0, 'bertscore_recall': 0.0
            }
            val_predictions, val_references = [], []
        
        # Log results
        if accelerator.is_main_process:
            if should_evaluate:
                log_msg = (
                    f"\nEpoch {epoch+1}/{config.num_epochs}\n"
                    f"  Train Loss: {avg_train_loss:.6f}\n"
                    f"  Val Loss: {avg_val_loss:.6f}\n"
                    f"  ROUGE-1: {val_metrics['rouge1']:.4f}\n"
                    f"  ROUGE-2: {val_metrics['rouge2']:.4f}\n"
                    f"  ROUGE-L: {val_metrics['rougeL']:.4f}\n"
                    f"  BERTScore F1: {val_metrics['bertscore_f1']:.4f}\n"
                )
            else:
                log_msg = (
                    f"\nEpoch {epoch+1}/{config.num_epochs}\n"
                    f"  Train Loss: {avg_train_loss:.6f}\n"
                    f"  Val Loss: {avg_val_loss:.6f}\n"
                    f"  (Metrics evaluation skipped)\n"
                )
            logger.info(log_msg)
            
            with open(log_file, 'a') as f:
                f.write(log_msg + "\n")
            
            # Log sample predictions only when evaluated
            if should_evaluate and len(val_predictions) >= 5:
                import random
                random_indices = random.sample(range(len(val_predictions)), 5)
                
                sample_log = f"\nSample Predictions (Epoch {epoch+1}):\n"
                sample_log += "-" * 50 + "\n"
                
                for i, idx in enumerate(random_indices):
                    pred = val_predictions[idx]
                    ref = val_references[idx]
                    sample_log += f"Sample {i+1}:\n"
                    sample_log += f"  Predicted: {pred}\n"
                    sample_log += f"  Reference: {ref}\n"
                    sample_log += "\n"
                
                logger.info(sample_log)
                with open(log_file, 'a') as f:
                    f.write(sample_log)
        
        # Early stopping (only when evaluated)
        if should_evaluate:
            val_rouge = val_metrics['rougeL']
            if val_rouge > best_val_rouge:
                best_val_rouge = val_rouge
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                if accelerator.is_main_process:
                    best_model_path = Path(config.output_dir) / "best_model"
                    best_model_path.mkdir(parents=True, exist_ok=True)
                    
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), best_model_path / "model.pt")
                    tokenizer.save_pretrained(best_model_path)
                    
                    with open(best_model_path / "config.json", 'w') as f:
                        json.dump(config.__dict__, f, indent=2)
                    
                    logger.info(f"Saved best model with ROUGE-L: {best_val_rouge:.4f}")
                    
                    with open(log_file, 'a') as f:
                        f.write(f"Saved best model at epoch {epoch+1} with ROUGE-L: {best_val_rouge:.4f}\n\n")
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    if accelerator.is_main_process:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        with open(log_file, 'a') as f:
                            f.write(f"Early stopping triggered after {epoch+1} epochs\n\n")
                    break
        
        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Training completed at {datetime.now().isoformat()}\n")
            f.write(f"Best validation ROUGE-L: {best_val_rouge:.4f}\n")
            f.write(f"Best validation loss: {best_val_loss:.6f}\n")
            f.write(f"{'='*80}\n\n")
    
    return model


def evaluate_finetuned(
    config: LlamaAdapterConfig,
    accelerator: Accelerator,
    log_file: Path
):
    """Evaluate fine-tuned model on test set."""
    if accelerator.is_main_process:
        logger.info("\n" + "="*80)
        logger.info("Evaluating Fine-tuned Model on Test Set")
        logger.info("="*80)
    
    test_data_path = Path(config.encoded_data_dir) / "test_chunks_encoded.pkl"
    if not test_data_path.exists():
        if accelerator.is_main_process:
            logger.error(f"Test data not found: {test_data_path}")
        return
    
    best_model_path = Path(config.output_dir) / "best_model" / "model.pt"
    if not best_model_path.exists():
        if accelerator.is_main_process:
            logger.error(f"Fine-tuned model not found: {best_model_path}")
            with open(log_file, 'a') as f:
                f.write(f"ERROR: Fine-tuned model not found at {best_model_path}\n")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(config.llama_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure padding side consistent between training and generation
    tokenizer.padding_side = "left"
    
    test_dataset = ChunkEncodingDataset(
        test_data_path,
        tokenizer,
        max_chunks=config.max_chunks_per_story,
        max_target_length=config.max_target_length
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    finetuned_model = LlamaChunkAdapter(
        llama_model_name=config.llama_model_name,
        encoded_dim=config.encoded_dim,
        adapter_hidden_dim=config.adapter_hidden_dim,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        freeze_llama=config.freeze_llama
    )
    
    if accelerator.is_main_process:
        logger.info(f"Loading model from {best_model_path}")
    
    finetuned_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    finetuned_model = finetuned_model.to(accelerator.device)
    
    finetuned_model, test_dataloader = accelerator.prepare(finetuned_model, test_dataloader)
    
    test_metrics, test_preds, test_refs = evaluate_model(
        accelerator.unwrap_model(finetuned_model),
        test_dataloader,
        tokenizer,
        config,
        accelerator,
        desc="Test Set Evaluation"
    )
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    if accelerator.is_main_process:
        test_log = (
            f"\nTest Set Results:\n"
            f"  ROUGE-1: {test_metrics['rouge1']:.4f}\n"
            f"  ROUGE-2: {test_metrics['rouge2']:.4f}\n"
            f"  ROUGE-L: {test_metrics['rougeL']:.4f}\n"
            f"  BERTScore Precision: {test_metrics['bertscore_precision']:.4f}\n"
            f"  BERTScore Recall: {test_metrics['bertscore_recall']:.4f}\n"
            f"  BERTScore F1: {test_metrics['bertscore_f1']:.4f}\n"
        )
        logger.info(test_log)
        
        with open(log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("TEST SET EVALUATION\n")
            f.write("="*80 + "\n")
            f.write(test_log + "\n")
        
        # Save test results
        test_results_path = Path("evaluations") / "test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Saved test results to {test_results_path}")


def evaluate_base_model(
    base_model,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    config: LlamaAdapterConfig,
    accelerator: Accelerator,
    desc: str = "Base Model Evaluation"
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Evaluate a plain base Llama model (no adapter) using full document text as input."""
    base_model.eval()
    all_predictions = []
    all_references = []
    
    eval_pbar = tqdm(dataloader, desc=desc, disable=not accelerator.is_main_process)
    
    prompt = "Summarize the following document. Give in paragraph format:\n\n"
    prompt_encoding = tokenizer(prompt, add_special_tokens=True, return_tensors='pt')
    prompt_len = len(prompt_encoding['input_ids'])
    
    # Leave room for generated tokens
    max_input_length = 2048
    
    with torch.no_grad():
        for batch in eval_pbar:
            document_texts = batch['document_texts']
            target_texts = batch['target_texts']
            batch_size = len(document_texts)
            
            # For each document, create full input: prompt + document_text
            full_inputs = [prompt + doc for doc in document_texts]
            
            # Tokenize batch with proper max length
            encodings = tokenizer(
                full_inputs,
                max_length=max_input_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(accelerator.device)
            attention_mask = encodings['attention_mask'].to(accelerator.device)
            
            # Generate summaries
            generated_ids = base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_target_length,
                min_new_tokens=config.min_target_length,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                num_beams=1,  # Use greedy decoding to save memory
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Decode predictions (remove input part)
            predictions = []
            for i in range(batch_size):
                pred_ids = generated_ids[i]
                # Remove the prompt part (prompt_len tokens)
                gen_ids = pred_ids[prompt_len:]
                pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                predictions.append(pred_text)
            
            all_predictions.extend(predictions)
            all_references.extend(target_texts)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    metrics = compute_metrics(all_predictions, all_references)
    return metrics, all_predictions, all_references
def compare_adapter_vs_base(
    config: LlamaAdapterConfig,
    accelerator: Accelerator,
    log_file: Path
):
    """Load the finetuned adapter model and the plain base model, evaluate both on the test set, and report a comparison."""
    if accelerator.is_main_process:
        logger.info("Comparing finetuned adapter model vs base Llama model on test set")

    test_data_path = Path(config.encoded_data_dir) / "test_chunks_encoded.pkl"
    if not test_data_path.exists():
        if accelerator.is_main_process:
            logger.error(f"Test data not found: {test_data_path}")
        return

    best_model_path = Path(config.output_dir) / "best_model" / "model.pt"
    if not best_model_path.exists():
        if accelerator.is_main_process:
            logger.error(f"Fine-tuned model not found: {best_model_path}")
            with open(log_file, 'a') as f:
                f.write(f"ERROR: Fine-tuned model not found at {best_model_path}\n")
        return

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.llama_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Dataset/Dataloader
    test_dataset = ChunkEncodingDataset(
        test_data_path,
        tokenizer,
        max_chunks=config.max_chunks_per_story,
        max_target_length=config.max_target_length
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )



    # Load finetuned adapter model
    finetuned_model = LlamaChunkAdapter(
        llama_model_name=config.llama_model_name,
        encoded_dim=config.encoded_dim,
        adapter_hidden_dim=config.adapter_hidden_dim,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        freeze_llama=config.freeze_llama
    )
    if accelerator.is_main_process:
        logger.info(f"Loading finetuned adapter model from {best_model_path}")

    finetuned_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    finetuned_model = finetuned_model.to(accelerator.device)

    # Load base model (plain Llama)
    if accelerator.is_main_process:
        logger.info(f"Loading base model {config.llama_model_name} for comparison")

    base_model = AutoModelForCausalLM.from_pretrained(
        config.llama_model_name,
        torch_dtype=torch.bfloat16,
        use_cache=True
    )
    base_model = base_model.to(accelerator.device)

    # Prepare models/dataloaders for distributed env
    finetuned_model, base_model, test_dataloader = accelerator.prepare(
        finetuned_model, base_model, test_dataloader
    )

    # Create separate dataloader for base model with smaller batch size to save memory
    base_dataloader = DataLoader(
        test_dataset,
        batch_size=4,  # Smaller batch size for base model
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    base_dataloader = accelerator.prepare(base_dataloader)

    # Evaluate finetuned adapter
    adapter_metrics, adapter_preds, adapter_refs = evaluate_model(
        accelerator.unwrap_model(finetuned_model),
        test_dataloader,
        tokenizer,
        config,
        accelerator,
        desc="Adapter Model Evaluation"
    )

    # Evaluate base model (only uses text prompt)
    base_metrics, base_preds, base_refs = evaluate_base_model(
        accelerator.unwrap_model(base_model),
        base_dataloader,
        tokenizer,
        config,
        accelerator,
        desc="Base Model Evaluation"
    )

    # Save and log comparison
    comparison = {
        'adapter_metrics': adapter_metrics,
        'base_metrics': base_metrics,
        'max_input_length_base_model': 2048
    }

    if accelerator.is_main_process:
        comp_path = Path("evaluations") / "llama_adapter_vs_base_comparison.json"
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved adapter vs base comparison to {comp_path}")

        # Print sample predictions
        import random
        num_samples = 3
        if len(adapter_preds) >= num_samples:
            indices = random.sample(range(len(adapter_preds)), num_samples)
            
            sample_log = "\n" + "="*80 + "\n"
            sample_log += "SAMPLE PREDICTIONS FROM ADAPTER MODEL\n"
            sample_log += "="*80 + "\n"
            
            print(sample_log, end="")
            for i, idx in enumerate(indices):
                sample = f"Sample {i+1}:\n"
                sample += f"  Predicted: {adapter_preds[idx]}\n"
                sample += f"  Reference: {adapter_refs[idx]}\n"
                sample += "\n"
                print(sample, end="")
                sample_log += sample
            
            sample_log += "\n" + "="*80 + "\n"
            sample_log += "SAMPLE PREDICTIONS FROM BASE MODEL\n"
            sample_log += "="*80 + "\n"
            
            print(sample_log, end="")
            for i, idx in enumerate(indices):
                sample = f"Sample {i+1}:\n"
                sample += f"  Predicted: {base_preds[idx]}\n"
                sample += f"  Reference: {adapter_refs[idx]}\n"  # Note: adapter_refs is same as base_refs
                sample += "\n"
                print(sample, end="")
                sample_log += sample

        # Print comparison table
        table_log = "\n" + "="*80 + "\n"
        table_log += "ADAPTER VS BASE MODEL COMPARISON (LLAMA)\n"
        table_log += "="*80 + "\n"
        table_log += f"{'Metric':<20} {'Adapter':<10} {'Base':<10} {'Difference':<10}\n"
        table_log += "-"*60 + "\n"
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1', 'bertscore_precision', 'bertscore_recall']:
            adapter_val = adapter_metrics.get(metric, 0.0)
            base_val = base_metrics.get(metric, 0.0)
            diff = adapter_val - base_val
            table_log += f"{metric:<20} {adapter_val:<10.4f} {base_val:<10.4f} {diff:<+10.4f}\n"
        table_log += "="*80 + "\n"
        
        print(table_log, end="")

        # Save sample predictions and table to files
        if len(adapter_preds) >= num_samples:
            samples_path = Path("evaluations") / "llama_sample_predictions.txt"
            with open(samples_path, 'w') as f:
                f.write(sample_log)
            logger.info(f"Saved sample predictions to {samples_path}")

        table_path = Path("evaluations") / "llama_comparison_table.txt"
        with open(table_path, 'w') as f:
            f.write(table_log)
        logger.info(f"Saved comparison table to {table_path}")

        # Append to log
        with open(log_file, 'a') as f:
            if len(adapter_preds) >= num_samples:
                f.write(sample_log)
            f.write(table_log)
            f.write("\nAdapter vs Base Comparison:\n")
            f.write(json.dumps(comparison, indent=2))
            f.write("\n")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return comparison


def main():
    """Main training and evaluation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = LlamaAdapterConfig()
    
    # MEMORY OPTIMIZATION: Enable mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16",  # Use bfloat16 mixed precision
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=False)  # Changed to False
        ]
    )

    # Command-line arguments: allow eval-only or comparison modes
    parser = argparse.ArgumentParser(description="Llama adapter training/evaluation")
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate the saved finetuned adapter on the test set and exit')
    parser.add_argument('--compare-base', action='store_true', help='Evaluate finetuned adapter and the plain base model and save a comparison')
    parser.add_argument('--model-path', type=str, default=None, help='Path to a saved finetuned model.pt (optional)')
    args = parser.parse_args()
    
    set_seed(config.seed)
    
    if accelerator.is_main_process:
        logger.info("="*80)
        logger.info("Memory-Optimized Llama 1B Adapter Fine-tuning Pipeline")
        logger.info("="*80)
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Mixed precision: bf16")
        logger.info(f"Llama Model: {config.llama_model_name}")
        logger.info(f"Encoded dim: {config.encoded_dim}")
        logger.info(f"Adapter hidden dim: {config.adapter_hidden_dim}")
        logger.info(f"Llama frozen: {config.freeze_llama}")
        logger.info(f"Gradient checkpointing: {config.use_gradient_checkpointing}")
        logger.info(f"Max chunks per story: {config.max_chunks_per_story}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes}")
        logger.info(f"Epochs: {config.num_epochs}")
        logger.info(f"Learning rate: {config.learning_rate}")
        logger.info(f"Evaluate every N epochs: {config.eval_every_n_epochs}")
        logger.info("="*80)
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
        Path("evaluations").mkdir(parents=True, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.logs_dir) / f"llama_adapter_training_{timestamp}.log"
    
    if accelerator.is_main_process:
        with open(log_file, 'w') as f:
            f.write(f"Memory-Optimized Llama 1B Adapter Training Log\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")

    # If user requested eval-only or comparison, run that and exit early
    if args.eval_only:
        if accelerator.is_main_process:
            logger.info("Running in --eval-only mode: evaluating saved finetuned model and exiting.")
        evaluate_finetuned(config, accelerator, log_file)
        return

    if args.compare_base:
        if accelerator.is_main_process:
            logger.info("Running in --compare-base mode: comparing adapter vs base and exiting.")
        compare_adapter_vs_base(config, accelerator, log_file)
        return
    
    if accelerator.is_main_process:
        logger.info("\nLoading datasets...")
    
    train_data_path = Path(config.encoded_data_dir) / "train_chunks_encoded.pkl"
    val_data_path = Path(config.encoded_data_dir) / "validation_chunks_encoded.pkl"
    
    if not train_data_path.exists() or not val_data_path.exists():
        if accelerator.is_main_process:
            logger.error("Encoded data not found. Please run chunk_embeddings.py first.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(config.llama_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure padding side consistent between training and generation
    tokenizer.padding_side = "left"
    
    train_dataset = ChunkEncodingDataset(
        train_data_path,
        tokenizer,
        max_chunks=config.max_chunks_per_story,
        max_target_length=config.max_target_length
    )
    
    val_dataset = ChunkEncodingDataset(
        val_data_path,
        tokenizer,
        max_chunks=config.max_chunks_per_story,
        max_target_length=config.max_target_length
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    if accelerator.is_main_process:
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    model = LlamaChunkAdapter(
        llama_model_name=config.llama_model_name,
        encoded_dim=config.encoded_dim,
        adapter_hidden_dim=config.adapter_hidden_dim,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        freeze_llama=config.freeze_llama
    )
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Percentage trainable: {(trainable_params/total_params)*100:.2f}%")
        logger.info(f"Memory savings: ~{(1 - trainable_params/total_params)*100:.1f}% fewer gradients")
    
    if accelerator.is_main_process:
        logger.info("\n" + "="*80)
        logger.info("Starting Training")
        logger.info("="*80)
    
    trained_model = train_model(
        model,
        train_dataloader,
        val_dataloader,
        tokenizer,
        config,
        accelerator,
        log_file
    )
    
    accelerator.wait_for_everyone()
    
    evaluate_finetuned(config, accelerator, log_file)
    
    if accelerator.is_main_process:
        logger.info("\n" + "="*80)
        logger.info("Pipeline Completed Successfully!")
        logger.info("="*80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Model saved in: {config.output_dir}")


if __name__ == "__main__":
    main()