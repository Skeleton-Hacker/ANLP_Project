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
    encoded_data_dir: str = "qa_chunked_data"
    output_dir: str = "models/bart_qa"
    dataset_name: str = "deepmind/narrativeqa"

    # Model
    bart_model: str = "facebook/bart-base"
    embedding_dim: int = 1024  # BGE-M3 embedding dimension
    freeze_decoder: bool = False

    # Training
    batch_size: int = 8
    epochs: int = 1000
    lr: float = 5e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 500
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    patience: int = 20
    eval_every_n_epochs: int = 1

    # Generation
    max_chunks: int = 800
    max_answer_len: int = 150
    min_answer_len: int = 20
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
    """Evaluate the model."""
    model.eval()
    preds, refs = [], []
    losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, disable=not accelerator.is_main_process):
            chunk_embeddings = batch['chunk_embeddings'].to(accelerator.device)
            question_embedding = batch['question_embedding'].to(accelerator.device)
            chunk_mask = batch['chunk_mask'].to(accelerator.device)
            answer_ids = batch['answer_ids'].to(accelerator.device)
            answer_mask = batch['answer_mask'].to(accelerator.device)
            
            # Loss
            outputs = model(chunk_embeddings, question_embedding, chunk_mask, answer_ids, answer_mask)
            losses.append(outputs.loss.item())
            
            # Generate
            gen_ids = model.generate(
                chunk_embeddings, question_embedding, chunk_mask,
                max_len=config.max_answer_len,
                min_len=config.min_answer_len,
                num_beams=config.num_beams
            )
            
            gen_ids = accelerator.pad_across_processes(gen_ids, dim=1, pad_index=tokenizer.pad_token_id)
            gen_ids = accelerator.gather_for_metrics(gen_ids)
            
            if accelerator.is_main_process:
                gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                preds.extend(gen_texts)
                refs.extend(batch['answer_texts'])
    
    if accelerator.is_main_process:
        metrics = compute_metrics(preds, refs)
        metrics['loss'] = np.mean(losses)
        return metrics, preds, refs
    
    return None, [], []


def train(model, train_dl, val_dl, tokenizer, config, accelerator):
    """Training loop."""
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    total_steps = len(train_dl) * config.epochs // config.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)
    
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )
    
    best_score = 0.0
    patience_counter = 0
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.epochs}", disable=not accelerator.is_main_process)
        
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                chunk_embeddings = batch['chunk_embeddings']
                question_embedding = batch['question_embedding']
                chunk_mask = batch['chunk_mask']
                answer_ids = batch['answer_ids']
                answer_mask = batch['answer_mask']
                
                outputs = model(chunk_embeddings, question_embedding, chunk_mask, answer_ids, answer_mask)
                loss = outputs.loss
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        # Evaluate
        if (epoch + 1) % config.eval_every_n_epochs == 0:
            val_metrics, val_preds, val_refs = evaluate(model, val_dl, tokenizer, config, accelerator, desc="Validation")
            
            if accelerator.is_main_process and val_metrics:
                logger.info(f"\nEpoch {epoch+1} Validation:")
                logger.info(f"  Loss: {val_metrics['loss']:.4f}")
                logger.info(f"  ROUGE-1: {val_metrics['rouge1']:.4f}")
                logger.info(f"  ROUGE-L: {val_metrics['rougeL']:.4f}")
                logger.info(f"  BERT F1: {val_metrics['bert_f1']:.4f}")
                logger.info(f"  Exact Match: {val_metrics['exact_match']:.4f}")
                
                # Save best model
                score = val_metrics['rougeL'] + val_metrics['exact_match']
                if score > best_score:
                    best_score = score
                    patience_counter = 0
                    torch.save(accelerator.unwrap_model(model).state_dict(), 
                              Path(config.output_dir) / "best_model.pt")
                    logger.info(f"  New best model saved! Score: {score:.4f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= config.patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break


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