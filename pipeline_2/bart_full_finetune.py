"""
BART Fine-tuning with BGE-M3 Chunk Embeddings

Trains BART on chunk embeddings from BGE-M3 model for summarization.
Uses accelerate for multi-GPU training.

Run: accelerate launch bart_finetuning.py [--eval-only] [--compare-base]
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
from transformers.modeling_outputs import BaseModelOutput
from accelerate import Accelerator
from tqdm import tqdm
import json
from rouge_score import rouge_scorer
from bert_score import score as bert_score

logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Config:
    # Data
    encoded_data_dir: str = "chunked_data"
    output_dir: str = "models/bart"
    
    # Model
    bart_model: str = "facebook/bart-base"
    embedding_dim: int = 1024  # BGE-M3 embedding dimension
    
    # Training
    batch_size: int = 16
    epochs: int = 20
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    patience: int = 5
    eval_every_n_steps: int = 500
    
    # Generation
    max_chunks: int = 800
    max_target_len: int = 150
    min_target_len: int = 20
    num_beams: int = 6
    
    # Hardware
    seed: int = 42


class ChunkDataset(Dataset):
    def __init__(self, pkl_path: Path, tokenizer: BartTokenizer, max_chunks: int, max_tgt_len: int):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.max_tgt_len = max_tgt_len
        
        self.stories = data['stories']
        self.samples = [sid for sid, s in self.stories.items() 
                       if 'chunk_embeddings' in s and len(s['chunk_embeddings']) > 0]
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        story_id = self.samples[idx]
        story = self.stories[story_id]
        
        # Get embeddings and truncate
        embeddings = story['chunk_embeddings'][:self.max_chunks]
        
        # Get target summary
        target = story['document']['summary']['text']
        target_enc = self.tokenizer(
            target, max_length=self.max_tgt_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        
        return {
            'embeddings': torch.FloatTensor(embeddings),
            'num_chunks': len(embeddings),
            'target_ids': target_enc['input_ids'].squeeze(0),
            'target_mask': target_enc['attention_mask'].squeeze(0),
            'target_text': target,
            'doc_text': ' '.join(story['chunks']) if 'chunks' in story else ""
        }


def collate_fn(batch):
    max_chunks = max(b['num_chunks'] for b in batch)
    embed_dim = batch[0]['embeddings'].shape[1]
    
    embeddings_list, masks_list = [], []
    for b in batch:
        emb = b['embeddings']
        if len(emb) < max_chunks:
            emb = torch.cat([emb, torch.zeros(max_chunks - len(emb), embed_dim)])
            mask = torch.cat([torch.ones(b['num_chunks']), torch.zeros(max_chunks - b['num_chunks'])])
        else:
            mask = torch.ones(max_chunks)
        embeddings_list.append(emb)
        masks_list.append(mask)
    
    return {
        'embeddings': torch.stack(embeddings_list),
        'chunk_mask': torch.stack(masks_list),
        'target_ids': torch.stack([b['target_ids'] for b in batch]),
        'target_mask': torch.stack([b['target_mask'] for b in batch]),
        'target_texts': [b['target_text'] for b in batch],
        'doc_texts': [b['doc_text'] for b in batch]
    }


class BartChunkModel(nn.Module):
    def __init__(self, bart_model: str, embed_dim: int):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model, use_safetensors=True)
        self.projection = nn.Linear(embed_dim, self.bart.config.d_model)
        
        logger.info(f"Total params: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(f"Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, embeddings, chunk_mask, target_ids, target_mask):
        encoder_hidden = self.projection(embeddings)
        encoder_out = BaseModelOutput(last_hidden_state=encoder_hidden)
        
        outputs = self.bart(
            encoder_outputs=encoder_out,
            attention_mask=chunk_mask,
            decoder_attention_mask=target_mask,
            labels=target_ids,
            return_dict=True
        )
        return outputs
    
    def generate(self, embeddings, chunk_mask, max_len=150, min_len=20, num_beams=6):
        encoder_hidden = self.projection(embeddings)
        encoder_out = BaseModelOutput(last_hidden_state=encoder_hidden)
        
        return self.bart.generate(
            encoder_outputs=encoder_out,
            attention_mask=chunk_mask,
            max_length=max_len,
            min_length=min_len,
            num_beams=num_beams,
            early_stopping=True
        )


def compute_metrics(preds: List[str], refs: List[str]) -> Dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {k: [] for k in ['rouge1', 'rouge2', 'rougeL']}
    
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    P, R, F1 = bert_score(preds, refs, lang='en', verbose=False)
    
    return {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL']),
        'bert_p': P.mean().item(),
        'bert_r': R.mean().item(),
        'bert_f1': F1.mean().item(),
    }


def evaluate(model, dataloader, tokenizer, config, accelerator, desc="Eval"):
    model.eval()
    preds, refs = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, disable=not accelerator.is_main_process):
            gen_ids = model.generate(
                batch['embeddings'], batch['chunk_mask'],
                max_len=config.max_target_len,
                min_len=config.min_target_len,
                num_beams=config.num_beams
            )
            preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            refs.extend(batch['target_texts'])
    
    return compute_metrics(preds, refs), preds, refs


def train(model, train_dl, val_dl, tokenizer, config, accelerator):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    
    total_steps = len(train_dl) * config.epochs // config.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)
    
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )
    
    best_rouge = 0
    patience = 0
    global_step = 0
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)
        for step, batch in enumerate(pbar):
            outputs = model(batch['embeddings'], batch['chunk_mask'], 
                          batch['target_ids'], batch['target_mask'])
            loss = outputs.loss / config.grad_accum_steps
            
            accelerator.backward(loss)
            if (step + 1) % config.grad_accum_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * config.grad_accum_steps
            pbar.set_postfix({'loss': f"{loss.item() * config.grad_accum_steps:.4f}"})
            
            if global_step > 0 and global_step % config.eval_every_n_steps == 0:
                metrics, _, _ = evaluate(accelerator.unwrap_model(model), val_dl, tokenizer, 
                                        config, accelerator, "Val")
                if accelerator.is_main_process:
                    logger.info(f"Step {global_step}: {metrics}")
                
                avg_rouge = np.mean([metrics['rouge1'], metrics['rouge2'], metrics['rougeL']])
                if avg_rouge > best_rouge:
                    best_rouge = avg_rouge
                    patience = 0
                    model_path = Path(config.output_dir) / "best_model.pt"
                    torch.save(accelerator.unwrap_model(model).state_dict(), model_path)
                else:
                    patience += 1
                    if patience >= config.patience:
                        if accelerator.is_main_process:
                            logger.info(f"Early stopping at step {global_step}")
                        return
                
                model.train()
        
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} avg loss: {epoch_loss/len(train_dl):.4f}")


def eval_only(config, accelerator):
    if accelerator.is_main_process:
        logger.info("Evaluation mode")
    
    test_path = Path(config.encoded_data_dir) / "test_chunks_encoded.pkl"
    model_path = Path(config.output_dir) / "best_model.pt"
    
    if not test_path.exists() or not model_path.exists():
        logger.error("Test data or model not found")
        return
    
    tokenizer = BartTokenizer.from_pretrained(config.bart_model, use_safetensors=True)
    test_ds = ChunkDataset(test_path, tokenizer, config.max_chunks, config.max_target_len)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = BartChunkModel(config.bart_model, config.embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(accelerator.device)
    
    model, test_dl = accelerator.prepare(model, test_dl)
    metrics, preds, refs = evaluate(accelerator.unwrap_model(model), test_dl, tokenizer, 
                                    config, accelerator, "Test")
    
    if accelerator.is_main_process:
        logger.info("Test Results:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        Path("evaluations").mkdir(exist_ok=True)
        with open("evaluations/results.json", 'w') as f:
            json.dump(metrics, f, indent=2)


def compare_base(config, accelerator):
    if accelerator.is_main_process:
        logger.info("Comparing finetuned vs base model")
    
    test_path = Path(config.encoded_data_dir) / "test_chunks_encoded.pkl"
    model_path = Path(config.output_dir) / "best_model.pt"
    
    if not test_path.exists() or not model_path.exists():
        logger.error("Test data or model not found")
        return
    
    tokenizer = BartTokenizer.from_pretrained(config.bart_model, use_safetensors=True)
    test_ds = ChunkDataset(test_path, tokenizer, config.max_chunks, config.max_target_len)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Finetuned model
    model = BartChunkModel(config.bart_model, config.embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model, test_dl = accelerator.prepare(model, test_dl)
    ft_metrics, ft_preds, ft_refs = evaluate(accelerator.unwrap_model(model), test_dl, 
                                             tokenizer, config, accelerator, "Finetuned")
    
    # Base model
    base_model = BartForConditionalGeneration.from_pretrained(config.bart_model, use_safetensors=True)
    base_model = base_model.to(accelerator.device)
    base_model.eval()
    
    base_preds, base_refs = [], []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Base", disable=not accelerator.is_main_process):
            input_enc = tokenizer(batch['doc_texts'], max_length=512, padding='max_length',
                                 truncation=True, return_tensors='pt').to(accelerator.device)
            gen_ids = base_model.generate(input_enc['input_ids'], input_enc['attention_mask'],
                                         max_length=config.max_target_len)
            base_preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            base_refs.extend(batch['target_texts'])
    
    base_metrics = compute_metrics(base_preds, base_refs)
    
    if accelerator.is_main_process:
        logger.info("Comparison Results:")
        logger.info(f"{'Metric':<15} {'Finetuned':>12} {'Base':>12} {'Improvement':>12}")
        logger.info("-" * 55)
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bert_f1']:
            ft_val = ft_metrics[metric]
            base_val = base_metrics[metric]
            improvement = ((ft_val - base_val) / base_val * 100) if base_val else 0
            logger.info(f"{metric:<15} {ft_val:>12.4f} {base_val:>12.4f} {improvement:>11.1f}%")
        
        Path("evaluations").mkdir(exist_ok=True)
        with open("evaluations/comparison.json", 'w') as f:
            json.dump({'finetuned': ft_metrics, 'base': base_metrics}, f, indent=2)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--compare-base', action='store_true')
    args = parser.parse_args()
    
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    from accelerate.utils import DistributedDataParallelKwargs

    # Create DDP kwargs with find_unused_parameters=True
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Pass to Accelerator via kwargs_handlers
    accelerator = Accelerator(
        gradient_accumulation_steps=config.grad_accum_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("BART Finetuning with BGE-M3 Embeddings")
        logger.info("=" * 60)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    if args.eval_only:
        eval_only(config, accelerator)
        return
    
    if args.compare_base:
        compare_base(config, accelerator)
        return
    
    # Training
    tokenizer = BartTokenizer.from_pretrained(config.bart_model, use_safetensors=True)
    
    train_path = Path(config.encoded_data_dir) / "train_chunks_encoded.pkl"
    val_path = Path(config.encoded_data_dir) / "validation_chunks_encoded.pkl"
    
    if not train_path.exists() or not val_path.exists():
        logger.error("Training or validation data not found")
        return
    
    train_ds = ChunkDataset(train_path, tokenizer, config.max_chunks, config.max_target_len)
    val_ds = ChunkDataset(val_path, tokenizer, config.max_chunks, config.max_target_len)
    
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = BartChunkModel(config.bart_model, config.embedding_dim)
    
    if accelerator.is_main_process:
        logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        logger.info("Starting training...")
    
    train(model, train_dl, val_dl, tokenizer, config, accelerator)
    eval_only(config, accelerator)


if __name__ == "__main__":
    main()