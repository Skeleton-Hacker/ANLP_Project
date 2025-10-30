"""
T5 Full Model Fine-tuning with Chunk Embeddings

This module fine-tunes T5-small encoder for summarization using chunk encodings.
It includes:
- Loading encoded chunks from pickle files
- Direct projection layer to map chunk encodings to T5 embedding space
- Fine-tuning encoder only (decoder frozen)
- Evaluation using ROUGE and BERTScore
- Comprehensive logging and model checkpointing
"""

import logging
import pickle
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
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
class T5FullConfig:
    """Configuration for T5 full model training."""
    # Data settings
    encoded_data_dir: str = "chunked_data"
    output_dir: str = "models/t5_full"
    logs_dir: str = "logs"
    
    # Model settings
    t5_model_name: str = "t5-small"
    encoded_dim: int = 64  # Dimension of encoded chunks from autoencoder
    
    # Training settings
    batch_size: int = 24
    num_epochs: int = 300
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    freeze_decoder: bool = True  # Only finetune encoder
    
    # Generation settings
    max_chunks_per_story: int = 800  # Maximum number of chunks to use per story
    max_target_length: int = 150  # Maximum length for generated summaries
    min_target_length: int = 20
    num_beams: int = 6
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
        max_chunks: int = 800,
        max_target_length: int = 150
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
            if 'encoded_embeddings' in story_data and len(story_data['encoded_embeddings']) > 0:
                self.samples.append(story_id)
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.stories)} stories")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        story_id = self.samples[idx]
        story_data = self.stories[story_id]
        
        # Get encoded embeddings
        encoded_embeddings = story_data['encoded_embeddings']  # Shape: (num_chunks, encoded_dim)
        
        # Truncate to max_chunks
        if len(encoded_embeddings) > self.max_chunks:
            encoded_embeddings = encoded_embeddings[:self.max_chunks]
        
        # Convert to tensor
        chunk_encodings = torch.FloatTensor(encoded_embeddings)
        
        # Get target summary
        target_text = story_data['document']['summary']['text']
        
        # Get document text (concatenated chunks)
        document_text = ' '.join(story_data['chunks']) if 'chunks' in story_data else ""
        
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
            'target_text': target_text,
            'document_text': document_text
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
            ], dim=0)
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
        'target_texts': [item['target_text'] for item in batch],
        'document_texts': [item['document_text'] for item in batch]
    }


class T5ChunkEncoder(nn.Module):
    """T5 model with projection layer for chunk encodings. Only encoder is finetuned."""
    
    def __init__(
        self,
        t5_model_name: str,
        encoded_dim: int,
        freeze_decoder: bool = True
    ):
        """Initialize T5 chunk encoder model.
        
        Args:
            t5_model_name: Name of the T5 model.
            encoded_dim: Dimension of encoded chunks.
            freeze_decoder: Whether to freeze decoder parameters.
        """
        super(T5ChunkEncoder, self).__init__()
        
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_hidden_dim = self.t5.config.d_model
        
        # Simple projection layer: encoded_dim -> t5_hidden_dim
        self.projection = nn.Linear(encoded_dim, self.t5_hidden_dim)
        
        # Freeze decoder if specified
        if freeze_decoder:
            logger.info("Freezing T5 decoder parameters...")
            for param in self.t5.decoder.parameters():
                param.requires_grad = False
            
            # Also freeze lm_head
            for param in self.t5.lm_head.parameters():
                param.requires_grad = False
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    
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
            target_ids: (batch, target_length)
            target_attention_mask: (batch, target_length)
            
        Returns:
            Loss and logits.
        """
        # Project chunk encodings to T5 embedding space
        encoder_hidden_states = self.projection(chunk_encodings)  # (batch, num_chunks, t5_hidden_dim)
        
        # Create encoder outputs
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )
        
        # Prepare decoder inputs (shift target_ids right)
        decoder_input_ids = self.t5._shift_right(target_ids)
        
        # Forward through decoder
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=chunk_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=target_attention_mask,
            labels=target_ids,
            return_dict=True
        )
        
        return outputs
    
    def generate(
        self,
        chunk_encodings: torch.Tensor,
        chunk_attention_mask: torch.Tensor,
        max_length: int = 150,
        num_beams: int = 6,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        min_length: int = 20
    ):
        """Generate summaries from chunk encodings.
        
        Args:
            chunk_encodings: (batch, num_chunks, encoded_dim)
            chunk_attention_mask: (batch, num_chunks)
            max_length: Maximum generation length.
            num_beams: Number of beams for beam search.
            repetition_penalty: Repetition penalty.
            length_penalty: Length penalty.
            no_repeat_ngram_size: No repeat n-gram size.
            min_length: Minimum generation length.
            
        Returns:
            Generated token IDs.
        """
        # Project chunk encodings to T5 embedding space
        encoder_hidden_states = self.projection(chunk_encodings)
        
        # Create encoder outputs
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )
        
        # Generate
        generated_ids = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=chunk_attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True
        )
        
        return generated_ids


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
    model: T5ChunkEncoder,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    config: T5FullConfig,
    accelerator: Accelerator,
    desc: str = "Evaluation"
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Evaluate model on a dataset.
    
    Args:
        model: T5 chunk encoder model.
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
            chunk_encodings = batch['chunk_encodings']
            chunk_attention_mask = batch['chunk_attention_mask']
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
            
            # Decode predictions
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(target_texts)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_references)
    
    return metrics, all_predictions, all_references


def train_model(
    model: T5ChunkEncoder,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    config: T5FullConfig,
    accelerator: Accelerator,
    log_file: Path
) -> T5ChunkEncoder:
    """Train the T5 chunk encoder model.
    
    Args:
        model: T5 chunk encoder model.
        train_dataloader: Training dataloader.
        val_dataloader: Validation dataloader.
        tokenizer: T5 tokenizer.
        config: Configuration.
        accelerator: Accelerator instance.
        log_file: Path to log file.
        
    Returns:
        Trained model.
    """
    # Optimizer (only optimize parameters that require grad)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
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
        logger.info("Starting training...")
        logger.info(f"Number of training steps: {num_training_steps}")
        logger.info(f"Number of epochs: {config.num_epochs}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        train_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            disable=not accelerator.is_main_process
        )
        
        for step, batch in enumerate(train_pbar):
            chunk_encodings = batch['chunk_encodings']
            chunk_attention_mask = batch['chunk_attention_mask']
            target_ids = batch['target_ids']
            target_attention_mask = batch['target_attention_mask']
            
            # Forward pass
            outputs = model(
                chunk_encodings=chunk_encodings,
                chunk_attention_mask=chunk_attention_mask,
                target_ids=target_ids,
                target_attention_mask=target_attention_mask
            )
            
            loss = outputs.loss / config.gradient_accumulation_steps
            
            # Backward pass
            accelerator.backward(loss)
            
            # Update weights
            if (step + 1) % config.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Periodic evaluation
            if global_step > 0 and global_step % config.eval_every_n_steps == 0:
                if accelerator.is_main_process:
                    logger.info(f"\nRunning evaluation at step {global_step}...")
                
                val_metrics, _, _ = evaluate_model(
                    accelerator.unwrap_model(model),
                    val_dataloader,
                    tokenizer,
                    config,
                    accelerator,
                    desc="Validation"
                )
                
                if accelerator.is_main_process:
                    logger.info(f"Step {global_step} Validation Metrics:")
                    for metric_name, metric_value in val_metrics.items():
                        logger.info(f"  {metric_name}: {metric_value:.4f}")
                
                model.train()
        
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches
        
        if accelerator.is_main_process:
            logger.info(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            logger.info("Running validation...")
        
        # Validation
        val_metrics, val_preds, val_refs = evaluate_model(
            accelerator.unwrap_model(model),
            val_dataloader,
            tokenizer,
            config,
            accelerator,
            desc="Validation"
        )
        
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1} Validation Metrics:")
            for metric_name, metric_value in val_metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # Save checkpoint if best model
            val_rouge_avg = (val_metrics['rouge1'] + val_metrics['rouge2'] + val_metrics['rougeL']) / 3
            
            if val_rouge_avg > best_val_rouge:
                best_val_rouge = val_rouge_avg
                best_val_loss = avg_epoch_loss
                patience_counter = 0
                
                logger.info(f"New best model! Average ROUGE: {val_rouge_avg:.4f}")
                
                # Save model
                best_model_dir = Path(config.output_dir) / "best_model"
                best_model_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    best_model_dir / "model.pt"
                )
                
                # Save tokenizer
                tokenizer.save_pretrained(best_model_dir)
                
                # Save config
                with open(best_model_dir / "config.json", 'w') as f:
                    json.dump(vars(config), f, indent=2)
                
                logger.info(f"Model saved to {best_model_dir}")
                
                # Save sample predictions
                with open(best_model_dir / "sample_predictions.txt", 'w') as f:
                    for i in range(min(5, len(val_preds))):
                        f.write(f"=== Example {i + 1} ===\n")
                        f.write(f"Reference: {val_refs[i]}\n")
                        f.write(f"Prediction: {val_preds[i]}\n\n")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{config.early_stopping_patience}")
                
                if patience_counter >= config.early_stopping_patience:
                    logger.info("Early stopping triggered!")
                    break
        
        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info("Training completed!")
        logger.info(f"Best validation ROUGE (avg): {best_val_rouge:.4f}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    return model


def evaluate_finetuned(
    config: T5FullConfig,
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
        logger.info("=" * 50)
        logger.info("Evaluating fine-tuned model on test set")
    
    # Load test data
    test_data_path = Path(config.encoded_data_dir) / "test_chunks_encoded.pkl"
    if not test_data_path.exists():
        logger.error(f"Test data not found at {test_data_path}")
        return
    
    # Check if model exists
    best_model_path = Path(config.output_dir) / "best_model" / "model.pt"
    if not best_model_path.exists():
        logger.error(f"Fine-tuned model not found at {best_model_path}")
        logger.error("Please train the model first!")
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
    finetuned_model = T5ChunkEncoder(
        t5_model_name=config.t5_model_name,
        encoded_dim=config.encoded_dim,
        freeze_decoder=config.freeze_decoder
    )
    
    if accelerator.is_main_process:
        logger.info(f"Loading fine-tuned model from {best_model_path}")
    
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
        logger.info("\n" + "=" * 50)
        logger.info("Test Set Results (Fine-tuned Model):")
        logger.info("=" * 50)
        for metric_name, metric_value in test_metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # Save test results
        results_path = Path(config.output_dir) / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        logger.info(f"\nTest results saved to {results_path}")
        
        # Save sample predictions
        sample_preds_path = Path(config.output_dir) / "test_sample_predictions.txt"
        with open(sample_preds_path, 'w') as f:
            for i in range(min(10, len(test_preds))):
                f.write(f"=== Test Example {i + 1} ===\n")
                f.write(f"Reference: {test_refs[i]}\n")
                f.write(f"Prediction: {test_preds[i]}\n\n")
        
        logger.info(f"Sample predictions saved to {sample_preds_path}")


def evaluate_base_model(
    base_model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    config: T5FullConfig,
    accelerator: Accelerator,
    desc: str = "Base Model Evaluation"
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Evaluate a plain base T5 model (no finetuning) using full document text as input.
    
    Args:
        base_model: Base T5 model.
        dataloader: DataLoader for evaluation.
        tokenizer: T5 tokenizer.
        config: Configuration.
        accelerator: Accelerator instance.
        desc: Description for progress bar.
        
    Returns:
        Tuple of (metrics, predictions, references).
    """
    base_model.eval()
    all_predictions = []
    all_references = []
    
    eval_pbar = tqdm(dataloader, desc=desc, disable=not accelerator.is_main_process)
    
    # Maximum input length for T5 (leave room for generation)
    max_input_length = 512
    
    with torch.no_grad():
        for batch in eval_pbar:
            document_texts = batch['document_texts']
            target_texts = batch['target_texts']
            
            # Tokenize document texts
            input_encodings = tokenizer(
                document_texts,
                max_length=max_input_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(accelerator.device)
            
            # Generate
            generated_ids = base_model.generate(
                input_ids=input_encodings['input_ids'],
                attention_mask=input_encodings['attention_mask'],
                max_length=config.max_target_length,
                min_length=config.min_target_length,
                num_beams=config.num_beams,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                early_stopping=True
            )
            
            # Decode predictions
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(target_texts)
    
    metrics = compute_metrics(all_predictions, all_references)
    return metrics, all_predictions, all_references


def compare_finetuned_vs_base(
    config: T5FullConfig,
    accelerator: Accelerator,
    log_file: Path
):
    """Load the finetuned model and the plain base model, evaluate both on the test set, and report a comparison.
    
    Args:
        config: Configuration.
        accelerator: Accelerator instance.
        log_file: Path to log file.
    """
    if accelerator.is_main_process:
        logger.info("=" * 50)
        logger.info("Comparing Fine-tuned Model vs Base Model")

    test_data_path = Path(config.encoded_data_dir) / "test_chunks_encoded.pkl"
    if not test_data_path.exists():
        logger.error(f"Test data not found at {test_data_path}")
        return

    best_model_path = Path(config.output_dir) / "best_model" / "model.pt"
    if not best_model_path.exists():
        logger.error(f"Fine-tuned model not found at {best_model_path}")
        logger.error("Please train the model first!")
        return

    # Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.t5_model_name)

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

    # Load finetuned model
    finetuned_model = T5ChunkEncoder(
        t5_model_name=config.t5_model_name,
        encoded_dim=config.encoded_dim,
        freeze_decoder=config.freeze_decoder
    )
    if accelerator.is_main_process:
        logger.info(f"Loading fine-tuned model from {best_model_path}")

    finetuned_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    finetuned_model = finetuned_model.to(accelerator.device)

    # Load base model (plain T5)
    if accelerator.is_main_process:
        logger.info(f"Loading base T5 model: {config.t5_model_name}")

    base_model = T5ForConditionalGeneration.from_pretrained(config.t5_model_name)
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

    # Evaluate finetuned model
    finetuned_metrics, finetuned_preds, finetuned_refs = evaluate_model(
        accelerator.unwrap_model(finetuned_model),
        test_dataloader,
        tokenizer,
        config,
        accelerator,
        desc="Fine-tuned Model Evaluation"
    )

    # Evaluate base model
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
        'finetuned_metrics': finetuned_metrics,
        'base_metrics': base_metrics,
        'max_input_length_base_model': 512
    }

    if accelerator.is_main_process:
        logger.info("\n" + "=" * 50)
        logger.info("COMPARISON RESULTS")
        logger.info("=" * 50)
        
        # Create evaluations directory
        evaluations_dir = Path("evaluations")
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison results
        comparison_path = evaluations_dir / "t5_enc_finetuned_vs_base_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"\nComparison saved to {comparison_path}")
        
        # Create comparison table
        logger.info("\nMetric Comparison:")
        logger.info("-" * 70)
        logger.info(f"{'Metric':<25} {'Fine-tuned':>15} {'Base':>15} {'Improvement':>15}")
        logger.info("-" * 70)
        
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']:
            finetuned_val = finetuned_metrics[metric]
            base_val = base_metrics[metric]
            improvement = ((finetuned_val - base_val) / base_val * 100) if base_val != 0 else 0
            
            logger.info(f"{metric:<25} {finetuned_val:>15.4f} {base_val:>15.4f} {improvement:>14.2f}%")
        
        logger.info("-" * 70)
        
        # Save comparison table to file
        table_path = evaluations_dir / "t5_enc_comparison_table.txt"
        with open(table_path, 'w') as f:
            f.write("COMPARISON: Fine-tuned T5 (Encoder Only) vs Base T5\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"{'Metric':<25} {'Fine-tuned':>15} {'Base':>15} {'Improvement':>15}\n")
            f.write("-" * 70 + "\n")
            
            for metric in ['rouge1', 'rouge2', 'rougeL', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']:
                finetuned_val = finetuned_metrics[metric]
                base_val = base_metrics[metric]
                improvement = ((finetuned_val - base_val) / base_val * 100) if base_val != 0 else 0
                
                f.write(f"{metric:<25} {finetuned_val:>15.4f} {base_val:>15.4f} {improvement:>14.2f}%\n")
            
            f.write("-" * 70 + "\n")
        
        logger.info(f"Comparison table saved to {table_path}")
        
        # Save sample predictions from both models
        sample_path = evaluations_dir / "t5_enc_sample_predictions.txt"
        with open(sample_path, 'w') as f:
            f.write("Sample Predictions: Fine-tuned vs Base Model\n")
            f.write("=" * 70 + "\n\n")
            
            for i in range(min(10, len(finetuned_preds))):
                f.write(f"=== Example {i + 1} ===\n")
                f.write(f"Reference:\n{finetuned_refs[i]}\n\n")
                f.write(f"Fine-tuned Model:\n{finetuned_preds[i]}\n\n")
                f.write(f"Base Model:\n{base_preds[i]}\n\n")
                f.write("-" * 70 + "\n\n")
        
        logger.info(f"Sample predictions comparison saved to {sample_path}")

    return comparison


def main():
    """Main training and evaluation pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = T5FullConfig()
    
    # Initialize Accelerator with kwargs for DDP
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    )

    # Command-line arguments: allow eval-only or comparison modes
    parser = argparse.ArgumentParser(description="T5 full model (encoder) training/evaluation")
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate the saved finetuned model on the test set and exit')
    parser.add_argument('--compare-base', action='store_true', help='Evaluate finetuned model and the plain base model and save a comparison')
    parser.add_argument('--model-path', type=str, default=None, help='Path to a saved finetuned model.pt (optional)')
    args = parser.parse_args()
    
    # Set seed
    set_seed(config.seed)
    
    if accelerator.is_main_process:
        logger.info("=" * 50)
        logger.info("T5 Full Model Fine-tuning (Encoder Only)")
        logger.info("=" * 50)
        logger.info(f"Configuration:")
        logger.info(f"  Model: {config.t5_model_name}")
        logger.info(f"  Encoded dim: {config.encoded_dim}")
        logger.info(f"  Freeze decoder: {config.freeze_decoder}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Number of epochs: {config.num_epochs}")
        logger.info(f"  Max chunks per story: {config.max_chunks_per_story}")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info("=" * 50)
        
        # Create directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    # Setup log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.logs_dir) / f"t5_full_training_{timestamp}.log"
    
    if accelerator.is_main_process:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    # If user requested eval-only or comparison, run that and exit early
    if args.eval_only:
        evaluate_finetuned(config, accelerator, log_file)
        return

    if args.compare_base:
        compare_finetuned_vs_base(config, accelerator, log_file)
        return
    
    # Load data
    if accelerator.is_main_process:
        logger.info("Loading datasets...")
    
    train_data_path = Path(config.encoded_data_dir) / "train_chunks_encoded.pkl"
    val_data_path = Path(config.encoded_data_dir) / "validation_chunks_encoded.pkl"
    
    if not train_data_path.exists() or not val_data_path.exists():
        logger.error(f"Training or validation data not found!")
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
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = T5ChunkEncoder(
        t5_model_name=config.t5_model_name,
        encoded_dim=config.encoded_dim,
        freeze_decoder=config.freeze_decoder
    )
    
    if accelerator.is_main_process:
        logger.info("Model initialized successfully")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    if accelerator.is_main_process:
        logger.info("\n" + "=" * 50)
        logger.info("Starting Training")
        logger.info("=" * 50)
    
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
        logger.info("\n" + "=" * 50)
        logger.info("Training Complete - Evaluating on Test Set")
        logger.info("=" * 50)
    
    evaluate_finetuned(config, accelerator, log_file)
    
    if accelerator.is_main_process:
        logger.info("\n" + "=" * 50)
        logger.info("All Done!")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
