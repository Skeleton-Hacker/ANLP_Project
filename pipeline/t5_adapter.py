"""
T5 Adapter for Fine-tuning with Chunk Embeddings

This module fine-tunes T5-small for summarization using chunk encodings instead of text.
It includes:
- Loading encoded chunks from pickle files
- Adapter layer to project chunk encodings to T5 embedding space
- Fine-tuning with early stopping
- Evaluation using ROUGE and BERTScore
- Comprehensive logging and model checkpointing
"""

import logging
import pickle
import os
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
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from transformers.modeling_outputs import BaseModelOutput
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
import json
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Setup logging
logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class T5AdapterConfig:
    """Configuration for T5 adapter training."""
    # Data settings
    encoded_data_dir: str = "chunked_data"
    output_dir: str = "models/t5_adapter"
    logs_dir: str = "logs"
    
    # Model settings
    t5_model_name: str = "t5-small"
    encoded_dim: int = 64  # Dimension of encoded chunks from autoencoder
    adapter_hidden_dim: int = 256  # Hidden dimension for adapter layer
    
    # Training settings
    batch_size: int = 24
    num_epochs: int = 300
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    
    # Generation settings
    max_chunks_per_story: int = 800  # Maximum number of chunks to use per story
    max_target_length: int = 150  # Maximum length for generated summaries
    min_target_length: int = 20
    num_beams: int = 4
    repetition_penalty: float = 2.5
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    
    # Hardware settings
    num_workers: int = 4
    
    # Evaluation settings
    eval_every_n_steps: int = 500
    
    # Random seed
    seed: int = 42


class ChunkEncodingDataset(Dataset):
    """Dataset for chunk encodings with summarization targets."""
    
    def __init__(
        self,
        encoded_data_path: Path,
        tokenizer: T5Tokenizer,
        max_chunks: int = 20,
        max_target_length: int = 128
    ):
        """Initialize dataset.
        
        Args:
            encoded_data_path: Path to encoded pickle file.
            tokenizer: T5 tokenizer for target sequences.
            max_chunks: Maximum number of chunks to use per story.
            max_target_length: Maximum length for target summaries.
        """
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.max_target_length = max_target_length
        
        # Load encoded data
        logger.info(f"Loading encoded data from {encoded_data_path}...")
        with open(encoded_data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.stories = data['stories']
        self.metadata = data['metadata']
        
        # Create list of samples (story_id)
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
        
        # Get encoded embeddings
        encoded_embeddings = story_data['encoded_embeddings']  # Shape: (num_chunks, encoded_dim)
        
        # Truncate or pad to max_chunks
        if len(encoded_embeddings) > self.max_chunks:
            encoded_embeddings = encoded_embeddings[:self.max_chunks]
        
        # Convert to tensor
        chunk_encodings = torch.FloatTensor(encoded_embeddings)
        
        # Get target summary
        target_text = story_data['document']['summary']['text']
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chunk_encodings': chunk_encodings,
            'num_chunks': len(encoded_embeddings),
            'target_ids': target_encoding['input_ids'].squeeze(0),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(0),
            'target_text': target_text
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length chunk sequences."""
    max_num_chunks = max(item['num_chunks'] for item in batch)
    encoded_dim = batch[0]['chunk_encodings'].shape[1]
    
    # Pad chunk encodings
    padded_chunk_encodings = []
    attention_masks = []
    
    for item in batch:
        num_chunks = item['num_chunks']
        chunk_encodings = item['chunk_encodings']
        
        # Pad to max_num_chunks
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
        attention_masks.append(attention_mask)
    
    return {
        'chunk_encodings': torch.stack(padded_chunk_encodings),  # (batch, max_chunks, encoded_dim)
        'chunk_attention_mask': torch.stack(attention_masks),  # (batch, max_chunks)
        'target_ids': torch.stack([item['target_ids'] for item in batch]),
        'target_attention_mask': torch.stack([item['target_attention_mask'] for item in batch]),
        'target_texts': [item['target_text'] for item in batch]
    }


class T5ChunkAdapter(nn.Module):
    """T5 model with adapter layer for chunk encodings."""
    
    def __init__(
        self,
        t5_model_name: str,
        encoded_dim: int,
        adapter_hidden_dim: int
    ):
        """Initialize T5 adapter model.
        
        Args:
            t5_model_name: Name of the T5 model.
            encoded_dim: Dimension of encoded chunks.
            adapter_hidden_dim: Hidden dimension for adapter.
        """
        super(T5ChunkAdapter, self).__init__()
        
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_hidden_dim = self.t5.config.d_model
        
        # Adapter layer: project encoded_dim -> t5_hidden_dim
        self.adapter = nn.Sequential(
            nn.Linear(encoded_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_dim, self.t5_hidden_dim),
            nn.LayerNorm(self.t5_hidden_dim)
        )
        
        logger.info(f"Initialized T5ChunkAdapter with {t5_model_name}")
        logger.info(f"  - T5 hidden dim: {self.t5_hidden_dim}")
        logger.info(f"  - Encoded dim: {encoded_dim}")
        logger.info(f"  - Adapter hidden dim: {adapter_hidden_dim}")
        logger.info(f"  - Full model fine-tuning enabled")
    
    def forward(
        self,
        chunk_encodings: torch.Tensor,
        chunk_attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_attention_mask: torch.Tensor
    ):
        """Forward pass.
        
        Args:
            chunk_encodings: (batch, num_chunks, encoded_dim)
            chunk_attention_mask: (batch, num_chunks)
            target_ids: (batch, seq_len)
            target_attention_mask: (batch, seq_len)
        """
        batch_size, num_chunks, _ = chunk_encodings.shape
        
        # Project chunk encodings to T5 hidden dim
        encoder_hidden_states = self.adapter(chunk_encodings)  # (batch, num_chunks, t5_hidden_dim)
        
        # Prepare labels (replace padding token id with -100)
        labels = target_ids.clone()
        labels[target_attention_mask == 0] = -100
        
        # Create proper encoder outputs object
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )
        
        # Forward through T5 with custom encoder hidden states
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=chunk_attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate(
        self,
        chunk_encodings: torch.Tensor,
        chunk_attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        min_length: int = 20
    ):
        """Generate summaries.
        
        Args:
            chunk_encodings: (batch, num_chunks, encoded_dim)
            chunk_attention_mask: (batch, num_chunks)
            max_length: Maximum generation length.
            num_beams: Number of beams for beam search.
            repetition_penalty: Penalty for repeating tokens.
            length_penalty: Penalty for longer sequences.
            no_repeat_ngram_size: Size of n-grams that cannot be repeated.
            min_length: Minimum generation length.
        """
        # Project chunk encodings to T5 hidden dim
        encoder_hidden_states = self.adapter(chunk_encodings)
        
        # Create proper encoder outputs object
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )
        
        # Generate
        outputs = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=chunk_attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )
        
        return outputs


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE and BERTScore metrics.
    
    Args:
        predictions: List of predicted summaries.
        references: List of reference summaries.
        
    Returns:
        Dictionary of metrics.
    """
    # ROUGE scores
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
    
    # BERTScore
    P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
    bert_results = {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }
    
    return {**rouge_results, **bert_results}


def evaluate_model(
    model: T5ChunkAdapter,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    config: T5AdapterConfig,
    accelerator: Accelerator,
    desc: str = "Evaluation"
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Evaluate model on a dataset.
    
    Args:
        model: T5 adapter model.
        dataloader: DataLoader for evaluation.
        tokenizer: T5 tokenizer.
        config: Configuration.
        accelerator: Accelerator instance.
        desc: Description for progress bar.
        
    Returns:
        Tuple of (metrics, predictions, references).
    """
    model.eval()
    all_predictions = []
    all_references = []
    
    eval_pbar = tqdm(dataloader, desc=desc, disable=not accelerator.is_main_process)
    
    with torch.no_grad():
        for batch in eval_pbar:
            chunk_encodings = batch['chunk_encodings'].to(accelerator.device)
            chunk_attention_mask = batch['chunk_attention_mask'].to(accelerator.device)
            target_texts = batch['target_texts']
            
            # Generate
            generated_ids = model.generate(
                chunk_encodings=chunk_encodings,
                chunk_attention_mask=chunk_attention_mask,
                max_length=config.max_target_length,
                min_length=config.min_target_length,
                num_beams=config.num_beams,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size
            )
            
            # Decode
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(target_texts)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_references)
    
    return metrics, all_predictions, all_references


def train_model(
    model: T5ChunkAdapter,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    config: T5AdapterConfig,
    accelerator: Accelerator,
    log_file: Path
) -> T5ChunkAdapter:
    """Train the T5 adapter model.
    
    Args:
        model: T5 adapter model.
        train_dataloader: Training dataloader.
        val_dataloader: Validation dataloader.
        tokenizer: T5 tokenizer.
        config: Configuration.
        accelerator: Accelerator instance.
        log_file: Path to log file.
        
    Returns:
        Trained model.
    """
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    # Training loop
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
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        train_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]",
            disable=not accelerator.is_main_process
        )
        
        for batch_idx, batch in enumerate(train_pbar):
            # Forward pass
            outputs = model(
                chunk_encodings=batch['chunk_encodings'],
                chunk_attention_mask=batch['chunk_attention_mask'],
                target_ids=batch['target_ids'],
                target_attention_mask=batch['target_attention_mask']
            )
            
            loss = outputs.loss / config.gradient_accumulation_steps
            
            # Backward pass
            accelerator.backward(loss)
            
            # Update parameters
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            train_loss += loss.item() * config.gradient_accumulation_steps
            num_train_batches += 1
            
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
                    target_ids=batch['target_ids'],
                    target_attention_mask=batch['target_attention_mask']
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                num_val_batches += 1
                
                val_pbar.set_postfix({'val_loss': loss.item()})
        
        avg_val_loss = val_loss / num_val_batches
        
        # Evaluate with metrics
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
        
        # Log results
        if accelerator.is_main_process:
            log_msg = (
                f"\nEpoch {epoch+1}/{config.num_epochs}\n"
                f"  Train Loss: {avg_train_loss:.6f}\n"
                f"  Val Loss: {avg_val_loss:.6f}\n"
                f"  ROUGE-1: {val_metrics['rouge1']:.4f}\n"
                f"  ROUGE-2: {val_metrics['rouge2']:.4f}\n"
                f"  ROUGE-L: {val_metrics['rougeL']:.4f}\n"
                f"  BERTScore F1: {val_metrics['bertscore_f1']:.4f}\n"
            )
            logger.info(log_msg)
            
            with open(log_file, 'a') as f:
                f.write(log_msg + "\n")
            
            # Log 5 random sample predictions
            if len(val_predictions) >= 5:
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
        
        # Early stopping based on ROUGE-L
        val_rouge = val_metrics['rougeL']
        if val_rouge > best_val_rouge:
            best_val_rouge = val_rouge
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
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
    config: T5AdapterConfig,
    accelerator: Accelerator,
    log_file: Path
):
    """Evaluate fine-tuned model on test set.
    
    Args:
        config: Configuration.
        accelerator: Accelerator instance.
        log_file: Path to log file.
    """
    if accelerator.is_main_process:
        logger.info("\n" + "="*80)
        logger.info("Evaluating Fine-tuned Model on Test Set")
        logger.info("="*80)
    
    # Load test data
    test_data_path = Path(config.encoded_data_dir) / "test_chunks_encoded.pkl"
    if not test_data_path.exists():
        if accelerator.is_main_process:
            logger.error(f"Test data not found: {test_data_path}")
        return
    
    # Check if model exists
    best_model_path = Path(config.output_dir) / "best_model" / "model.pt"
    if not best_model_path.exists():
        if accelerator.is_main_process:
            logger.error(f"Fine-tuned model not found: {best_model_path}")
            with open(log_file, 'a') as f:
                f.write(f"ERROR: Fine-tuned model not found at {best_model_path}\n")
        return
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.t5_model_name)
    
    # Create test dataset and dataloader
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
    
    # Load fine-tuned model
    finetuned_model = T5ChunkAdapter(
        t5_model_name=config.t5_model_name,
        encoded_dim=config.encoded_dim,
        adapter_hidden_dim=config.adapter_hidden_dim
    )
    
    if accelerator.is_main_process:
        logger.info(f"Loading model from {best_model_path}")
    
    finetuned_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    finetuned_model = finetuned_model.to(accelerator.device)
    
    # Prepare with accelerator
    finetuned_model, test_dataloader = accelerator.prepare(finetuned_model, test_dataloader)
    
    # Evaluate
    test_metrics, test_preds, test_refs = evaluate_model(
        accelerator.unwrap_model(finetuned_model),
        test_dataloader,
        tokenizer,
        config,
        accelerator,
        desc="Test Set Evaluation"
    )
    
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
        test_results_path = Path(config.output_dir) / "test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Saved test results to {test_results_path}")


def main():
    """Main training and evaluation pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = T5AdapterConfig()
    
    # Initialize Accelerator with kwargs for DDP
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    )
    
    # Set seed
    set_seed(config.seed)
    
    if accelerator.is_main_process:
        logger.info("="*80)
        logger.info("T5 Adapter Fine-tuning Pipeline")
        logger.info("="*80)
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"T5 Model: {config.t5_model_name}")
        logger.info(f"Encoded dim: {config.encoded_dim}")
        logger.info(f"Adapter hidden dim: {config.adapter_hidden_dim}")
        logger.info(f"Full model fine-tuning: Enabled")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
        logger.info(f"Epochs: {config.num_epochs}")
        logger.info(f"Learning rate: {config.learning_rate}")
        logger.info("="*80)
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    # Setup log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.logs_dir) / f"t5_adapter_training_{timestamp}.log"
    
    if accelerator.is_main_process:
        with open(log_file, 'w') as f:
            f.write(f"T5 Adapter Training Log\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
    
    # Load data
    if accelerator.is_main_process:
        logger.info("\nLoading datasets...")
    
    train_data_path = Path(config.encoded_data_dir) / "train_chunks_encoded.pkl"
    val_data_path = Path(config.encoded_data_dir) / "validation_chunks_encoded.pkl"
    
    if not train_data_path.exists() or not val_data_path.exists():
        if accelerator.is_main_process:
            logger.error("Encoded data not found. Please run chunk_embeddings.py first.")
        return
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.t5_model_name)
    
    # Create datasets
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
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    if accelerator.is_main_process:
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = T5ChunkAdapter(
        t5_model_name=config.t5_model_name,
        encoded_dim=config.encoded_dim,
        adapter_hidden_dim=config.adapter_hidden_dim
    )
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Percentage trainable: {(trainable_params/total_params)*100:.2f}%")
    
    # Train model
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
    
    # Evaluate fine-tuned model on test set
    if accelerator.is_main_process:
        logger.info("\n" + "="*80)
        logger.info("Starting Test Set Evaluation")
        logger.info("="*80)
    
    evaluate_finetuned(config, accelerator, log_file)
    
    if accelerator.is_main_process:
        logger.info("\n" + "="*80)
        logger.info("Pipeline Completed Successfully!")
        logger.info("="*80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Model saved in: {config.output_dir}")


if __name__ == "__main__":
    main()
