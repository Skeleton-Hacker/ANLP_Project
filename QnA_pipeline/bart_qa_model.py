"""
BART Fine-tuning for Question Answering with Chunk Embeddings
"""
import logging
import pickle
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator
from tqdm import tqdm
import json
from rouge_score import rouge_scorer
from bert_score import score as bert_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Config:
    # Data
    encoded_data_dir: str = "chunked_data"
    output_dir: str = "models/bart_qa"
    dataset_name: str = "deepmind/narrativeqa"

    # Model
    bart_model: str = "facebook/bart-base"
    embedding_dim: int = 1024  # BGE-M3 embedding dimension
    freeze_decoder: bool = False

    # Training
    batch_size: int = 8
    num_epochs: int = 25  # Changed from 'epochs'
    learning_rate: float = 5e-4  # Changed from 'lr'
    weight_decay: float = 1e-3
    warmup_steps: int = 0
    warmup_ratio: float = 0.0  # Added for scheduler
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    patience: int = 20
    eval_every_n_epochs: int = 1

    # Generation
    max_chunks: int = 800
    max_answer_len: int = 150  # Changed from 'max_answer_len'
    min_answer_len: int = 20   # Changed from 'min_answer_len'
    num_beams: int = 6

    # Question encoding
    use_question_embedding: bool = True
    question_dim: int = 1024

    # Architecture
    projection_dropout: float = 0.1
    use_chunk_pos_embeddings: bool = True

    seed: int = 42


class QADataset(Dataset):
    """Dataset for QA with chunk embeddings."""
    
    def __init__(self, pkl_path: Path, tokenizer: BartTokenizer, config: Config):
        accelerator = Accelerator()
        with accelerator.main_process_first():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        
        self.tokenizer = tokenizer
        self.config = config
        
        self.samples = []
        for sample_id, sample_data in data['samples'].items():
            if 'chunk_embeddings' in sample_data and len(sample_data['chunk_embeddings']) > 0:
                if 'answer' in sample_data and sample_data['answer']:
                    self.samples.append(sample_id)
        
        self.data = data['samples']
        logger.info(f"Loaded {len(self.samples)} QA samples from {pkl_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        sample = self.data[sample_id]
        
        # Get chunk embeddings
        chunk_embeddings = np.array(sample['chunk_embeddings'], dtype=np.float32)[:self.config.max_chunks]
        num_chunks = chunk_embeddings.shape[0]
        
        # Get question embedding
        question_emb = np.array(sample['question_embedding'], dtype=np.float32)
        
        # Get answer
        answer = sample['answer'] if isinstance(sample['answer'], str) else ""
        
        # Tokenize answer
        answer_enc = self.tokenizer(
            answer,
            max_length=self.config.max_answer_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chunk_embeddings': torch.from_numpy(chunk_embeddings),
            'question_embedding': torch.from_numpy(question_emb),
            'num_chunks': num_chunks,
            'answer_ids': answer_enc['input_ids'].squeeze(0),
            'answer_mask': answer_enc['attention_mask'].squeeze(0),
            'answer_text': answer,
            'question_text': sample['question']
        }


def collate_fn(batch):
    """Collate function for QA batches."""
    batch_size = len(batch)
    max_chunks = max(b['num_chunks'] for b in batch)
    embed_dim = batch[0]['chunk_embeddings'].shape[1]
    question_dim = batch[0]['question_embedding'].shape[0]
    
    # Pad chunk embeddings
    chunk_embeddings_padded = torch.zeros((batch_size, max_chunks, embed_dim), dtype=torch.float)
    chunk_masks = torch.zeros((batch_size, max_chunks), dtype=torch.bool)
    question_embeddings = torch.zeros((batch_size, question_dim), dtype=torch.float)
    
    answer_ids = torch.stack([b['answer_ids'] for b in batch], dim=0)
    answer_mask = torch.stack([b['answer_mask'] for b in batch], dim=0)
    
    for i, b in enumerate(batch):
        n = b['num_chunks']
        chunk_embeddings_padded[i, :n, :] = b['chunk_embeddings']
        chunk_masks[i, :n] = 1
        question_embeddings[i] = b['question_embedding']
    
    return {
        'chunk_embeddings': chunk_embeddings_padded,
        'chunk_mask': chunk_masks,
        'question_embedding': question_embeddings,
        'answer_ids': answer_ids,
        'answer_mask': answer_mask,
        'answer_texts': [b['answer_text'] for b in batch],
        'question_texts': [b['question_text'] for b in batch]
    }


class BartQAModel(nn.Module):
    """BART model for QA with chunk embeddings and question encoding."""
    
    def __init__(self, bart_model: str, embed_dim: int, question_dim: int, config: Config):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model, use_safetensors=True)
        self.config = config
        
        # Project chunks to BART dimension
        self.chunk_projection = nn.Linear(embed_dim, self.bart.config.d_model)
        
        # Project question to BART dimension
        if config.use_question_embedding:
            self.question_projection = nn.Linear(question_dim, self.bart.config.d_model)
        
        self.layernorm = nn.LayerNorm(self.bart.config.d_model)
        self.dropout = nn.Dropout(config.projection_dropout)
        
        # Chunk positional embeddings
        if config.use_chunk_pos_embeddings:
            max_positions = config.max_chunks + 1  # +1 for question
            self.chunk_pos_emb = nn.Embedding(max_positions, self.bart.config.d_model)
        else:
            self.chunk_pos_emb = None
        
        if config.freeze_decoder:
            for param in self.bart.model.decoder.parameters():
                param.requires_grad = False
            logger.info("Decoder frozen")
    
    def _encode(self, chunk_embeddings: torch.Tensor, question_embedding: torch.Tensor, chunk_mask: torch.Tensor):
        """Encode chunks and question together."""
        # Project chunks
        chunks_projected = self.chunk_projection(chunk_embeddings)
        
        # Project and expand question
        if self.config.use_question_embedding:
            question_projected = self.question_projection(question_embedding)
            question_projected = question_projected.unsqueeze(1)
            
            # Concatenate question at the beginning
            x = torch.cat([question_projected, chunks_projected], dim=1)
            
            # Extend mask for question
            question_mask = torch.ones(chunk_mask.size(0), 1, dtype=torch.bool, device=chunk_mask.device)
            extended_mask = torch.cat([question_mask, chunk_mask], dim=1)
        else:
            x = chunks_projected
            extended_mask = chunk_mask
        
        x = self.layernorm(x)
        x = self.dropout(x)
        
        # Add positional embeddings
        if self.chunk_pos_emb is not None:
            seq_len = x.size(1)
            pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
            pos_emb = self.chunk_pos_emb(pos_ids)
            x = x + pos_emb
        
        # Pass through BART encoder
        attention_mask = extended_mask.to(dtype=torch.long)
        encoder_outputs = self.bart.model.encoder(
            inputs_embeds=x,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return encoder_outputs, extended_mask
    
    def forward(self, chunk_embeddings, question_embedding, chunk_mask, answer_ids, answer_mask):
        """Forward pass for training."""
        encoder_outputs, extended_mask = self._encode(chunk_embeddings, question_embedding, chunk_mask)
        
        # Prepare labels
        labels = answer_ids.clone()
        labels[labels == self.bart.config.pad_token_id] = -100
        
        outputs = self.bart(
            encoder_outputs=encoder_outputs,
            attention_mask=extended_mask.to(dtype=torch.long),
            decoder_attention_mask=answer_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, chunk_embeddings, question_embedding, chunk_mask, max_len=100, min_len=5, num_beams=4):
        """Generate answers."""
        encoder_outputs, extended_mask = self._encode(chunk_embeddings, question_embedding, chunk_mask)
        
        return self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=extended_mask.to(dtype=torch.long),
            max_length=max_len,
            min_length=min_len,
            num_beams=num_beams,
            early_stopping=True
        )


def compute_metrics(preds: List[str], refs: List[str]) -> Dict:
    """Compute ROUGE, BERT scores, and Exact Match for QA."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {k: [] for k in ['rouge1', 'rouge2', 'rougeL']}
    
    exact_matches = []
    
    for pred, ref in zip(preds, refs):
        # ROUGE
        scores = scorer.score(ref, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Exact match (normalized)
        pred_normalized = pred.lower().strip()
        ref_normalized = ref.lower().strip()
        exact_matches.append(1.0 if pred_normalized == ref_normalized else 0.0)
    
    # BERTScore
    if len(preds) > 0:
        P, R, F1 = bert_score(preds, refs, lang='en', verbose=False)
        bert_f1 = F1.mean().item()
    else:
        bert_f1 = 0.0
    
    return {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL']),
        'bert_f1': bert_f1,
        'exact_match': np.mean(exact_matches)
    }


def evaluate(model, dataloader, tokenizer, config, accelerator, desc="Eval"):
    """Evaluate model and compute metrics."""
    model.eval()
    all_preds = []
    all_refs = []
    total_loss = 0.0  # ✅ Track validation loss
    num_batches = 0
    
    # Unwrap model for generation (needed for DDP/Accelerate)
    unwrapped_model = accelerator.unwrap_model(model)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, disable=not accelerator.is_main_process):
            chunk_embeds = batch['chunk_embeddings']
            chunk_mask = batch['chunk_mask']
            question_embeds = batch['question_embedding']
            answer_ids = batch['answer_ids']
            answer_mask = batch['answer_mask']
            answer_texts = batch['answer_texts']
            
            # Compute loss
            outputs = model(
                chunk_embeddings=chunk_embeds,
                question_embedding=question_embeds,
                chunk_mask=chunk_mask,
                answer_ids=answer_ids,
                answer_mask=answer_mask
            )
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # Generate predictions (for metrics)
            gen_ids = unwrapped_model.generate(
                chunk_embeddings=chunk_embeds,
                question_embedding=question_embeds,
                chunk_mask=chunk_mask,
                max_len=config.max_answer_len,
                min_len=config.min_answer_len,
                num_beams=config.num_beams
            )
            
            # Decode predictions
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            
            all_preds.extend(preds)
            all_refs.extend(answer_texts)
    
    # Compute average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Gather predictions and loss from all processes
    all_preds = accelerator.gather_for_metrics(all_preds)
    all_refs = accelerator.gather_for_metrics(all_refs)
    avg_loss = accelerator.reduce(torch.tensor(avg_loss).to(accelerator.device), reduction="mean").item()
    
    # Compute metrics on main process
    if accelerator.is_main_process:
        metrics = compute_metrics(all_preds, all_refs)
        metrics['loss'] = avg_loss  # ✅ Add loss to metrics
        return metrics, all_preds, all_refs
    else:
        return {'loss': avg_loss}, [], []


def train(model, train_dl, val_dl, tokenizer, config, accelerator):
    """Training loop with validation."""
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    num_training_steps = len(train_dl) * config.num_epochs
    num_warmup_steps = config.warmup_steps
    
    if num_warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        from torch.optim.lr_scheduler import ConstantLR
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=1)
    
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )
    
    if accelerator.is_main_process:
        logger.info(f"Starting training for {config.num_epochs} epochs")
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps} ({'disabled' if num_warmup_steps == 0 else 'enabled'})")
        logger.info(f"Early stopping patience: {config.patience} epochs (based on validation loss)")
    
    best_val_loss = float('inf')
    best_epoch = 0
    global_step = 0
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        epoch_steps = 0
        
        train_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]", 
                        disable=not accelerator.is_main_process)
        
        for batch in train_bar:
            chunk_embeds = batch['chunk_embeddings']
            chunk_mask = batch['chunk_mask']
            question_embeds = batch['question_embedding']
            answer_ids = batch['answer_ids']
            answer_mask = batch['answer_mask']
            
            outputs = model(
                chunk_embeddings=chunk_embeds,
                question_embedding=question_embeds,
                chunk_mask=chunk_mask,
                answer_ids=answer_ids,
                answer_mask=answer_mask
            )
            
            loss = outputs.loss
            accelerator.backward(loss / config.grad_accum_steps)
            
            if (global_step + 1) % config.grad_accum_steps == 0:
                if config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            
            if accelerator.is_main_process:
                train_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{epoch_loss/epoch_steps:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
        
        avg_train_loss = epoch_loss / epoch_steps
        
        # Validation
        if accelerator.is_main_process:
            logger.info(f"\nEpoch {epoch+1}/{config.num_epochs} - Avg train loss: {avg_train_loss:.4f}")
            logger.info("Running validation...")
        
        val_metrics, val_preds, val_refs = evaluate(
            model, val_dl, tokenizer, config, accelerator, desc="Validation"
        )
        
        if accelerator.is_main_process:
            val_loss = val_metrics.get('loss', float('inf'))
            
            logger.info(f"Validation metrics:")
            logger.info(f"  loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                if k != 'loss':
                    logger.info(f"  {k}: {v:.4f}")
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save checkpoint
                output_dir = Path(config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                unwrapped_model = accelerator.unwrap_model(model)
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': vars(config),
                }
                
                torch.save(checkpoint, output_dir / "best_model.pt")
                
                # ✅ Save predictions and references for best model
                predictions_file = output_dir / f"best_predictions_epoch_{epoch+1}.json"
                predictions_data = {
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'metrics': val_metrics,
                    'predictions': [
                        {
                            'prediction': pred,
                            'reference': ref
                        }
                        for pred, ref in zip(val_preds, val_refs)
                    ]
                }
                
                with open(predictions_file, 'w') as f:
                    json.dump(predictions_data, f, indent=2)
                
                logger.info(f"✓ Saved best model (Val Loss: {best_val_loss:.4f}, improved by {improvement:.4f})")
                logger.info(f"✓ Saved predictions to {predictions_file}")
                logger.info(f"✓ Best epoch so far: {best_epoch}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{config.patience} (Best val loss: {best_val_loss:.4f})")
            
            # ✅ Also save predictions for every epoch (optional - for debugging)
            all_predictions_file = output_dir / f"val_predictions_epoch_{epoch+1}.json"
            all_predictions_data = {
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'metrics': val_metrics,
                'predictions': [
                    {
                        'prediction': pred,
                        'reference': ref
                    }
                    for pred, ref in zip(val_preds, val_refs)
                ]
            }
            
            with open(all_predictions_file, 'w') as f:
                json.dump(all_predictions_data, f, indent=2)
            logger.info(f"✓ Saved epoch predictions to {all_predictions_file}")
            
            # Early stopping
            if patience_counter >= config.patience:
                logger.info(f"\n{'='*80}")
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                logger.info(f"Best epoch: {best_epoch} with validation loss: {best_val_loss:.4f}")
                logger.info(f"{'='*80}")
                break
        
        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info("\n" + "="*80)
        logger.info(f"Training complete!")
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Total epochs run: {epoch + 1}")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()
    
    config = Config()
    accelerator = Accelerator(gradient_accumulation_steps=config.grad_accum_steps)
    
    if accelerator.is_main_process:
        logger.info("="*80)
        logger.info(f"BART QA Training - {config.dataset_name}")
        logger.info("="*80)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    tokenizer = BartTokenizer.from_pretrained(config.bart_model)
    
    if args.eval_only:
        test_path = Path(config.encoded_data_dir) / "test_qa_encoded.pkl"
        model_path = Path(config.output_dir) / "best_model.pt"
        
        if not test_path.exists() or not model_path.exists():
            logger.error("Test data or model not found")
            return
        
        test_ds = QADataset(test_path, tokenizer, config)
        test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        
        model = BartQAModel(config.bart_model, config.embedding_dim, config.question_dim, config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model, test_dl = accelerator.prepare(model, test_dl)
        
        test_metrics, test_preds, test_refs = evaluate(
            model, test_dl, tokenizer, config, accelerator, desc="Test"
        )
        
        if accelerator.is_main_process and test_metrics:
            logger.info("\nTest Results:")
            logger.info(f"  ROUGE-1: {test_metrics['rouge1']:.4f}")
            logger.info(f"  ROUGE-L: {test_metrics['rougeL']:.4f}")
            logger.info(f"  BERT F1: {test_metrics['bert_f1']:.4f}")
            logger.info(f"  Exact Match: {test_metrics['exact_match']:.4f}")
        
        return
    
    # Training
    train_path = Path(config.encoded_data_dir) / "train_qa_encoded.pkl"
    val_path = Path(config.encoded_data_dir) / "validation_qa_encoded.pkl"
    
    if not train_path.exists() or not val_path.exists():
        logger.error("Training or validation data not found")
        return
    
    train_ds = QADataset(train_path, tokenizer, config)
    val_ds = QADataset(val_path, tokenizer, config)
    
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = BartQAModel(config.bart_model, config.embedding_dim, config.question_dim, config)
    
    train(model, train_dl, val_dl, tokenizer, config, accelerator)


if __name__ == "__main__":
    main()