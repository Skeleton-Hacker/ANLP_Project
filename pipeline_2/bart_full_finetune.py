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
    freeze_decoder: bool = True  # Whether to freeze the BART decoder parameters

    # Training
    batch_size: int = 8
    epochs: int = 1000
    lr: float = 5e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 500
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    patience: int = 3
    eval_every_n_epochs: int = 1  # Evaluate every epoch

    # Generation
    max_chunks: int = 800
    max_target_len: int = 150
    min_target_len: int = 20
    num_beams: int = 6

    # Hardware / misc
    seed: int = 42
    projection_dropout: float = 0.1
    use_chunk_pos_embeddings: bool = True  # learned positional embeddings for chunks


class ChunkDataset(Dataset):
    def __init__(self, pkl_path: Path, tokenizer: BartTokenizer, max_chunks: int, max_tgt_len: int):
        accelerator = Accelerator()
        with accelerator.main_process_first():
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.max_tgt_len = max_tgt_len

        self.stories = data['stories']
        # Only keep stories that actually have chunk_embeddings and summary text
        self.samples = [
            sid for sid, s in self.stories.items()
            if 'chunk_embeddings' in s and len(s['chunk_embeddings']) > 0 and
               'document' in s and 'summary' in s['document'] and 'text' in s['document']['summary']
        ]

        logger.info(f"Loaded {len(self.samples)} samples from {pkl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        story_id = self.samples[idx]
        story = self.stories[story_id]

        # Get embeddings and truncate
        embeddings = np.array(story['chunk_embeddings'], dtype=np.float32)[:self.max_chunks]
        num_chunks = embeddings.shape[0]

        # Get target summary
        target = story['document']['summary']['text']
        target_enc = self.tokenizer(
            target, max_length=self.max_tgt_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        target_ids = target_enc['input_ids'].squeeze(0)  # (max_tgt_len,)
        target_mask = target_enc['attention_mask'].squeeze(0)  # (max_tgt_len,)

        return {
            'embeddings': torch.from_numpy(embeddings),  # (num_chunks, embed_dim)
            'num_chunks': num_chunks,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'target_text': target,
            'doc_text': ' '.join(story['chunks']) if 'chunks' in story else ""
        }


def collate_fn(batch):
    # Batch is list of dicts
    batch_size = len(batch)
    max_chunks = max(b['num_chunks'] for b in batch)
    embed_dim = batch[0]['embeddings'].shape[1]

    embeddings_padded = torch.zeros((batch_size, max_chunks, embed_dim), dtype=torch.float)
    chunk_masks = torch.zeros((batch_size, max_chunks), dtype=torch.bool)

    target_ids = torch.stack([b['target_ids'] for b in batch], dim=0)
    target_mask = torch.stack([b['target_mask'] for b in batch], dim=0)

    for i, b in enumerate(batch):
        n = b['num_chunks']
        embeddings_padded[i, :n, :] = b['embeddings']
        chunk_masks[i, :n] = 1  # True for actual chunks

    return {
        'embeddings': embeddings_padded,         # (B, S, D)
        'chunk_mask': chunk_masks,              # (B, S) bool
        'target_ids': target_ids,               # (B, T)
        'target_mask': target_mask,             # (B, T)
        'target_texts': [b['target_text'] for b in batch],
        'doc_texts': [b['doc_text'] for b in batch]
    }


class BartChunkModel(nn.Module):
    def __init__(self, bart_model: str, embed_dim: int, freeze_decoder: bool = True, config: Config = None):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model, use_safetensors=True)
        self.projection = nn.Linear(embed_dim, self.bart.config.d_model)

        # Optional small LayerNorm and dropout to stabilize projection
        self.layernorm = nn.LayerNorm(self.bart.config.d_model)
        self.dropout = nn.Dropout(config.projection_dropout if config is not None else 0.1)

        # Learned chunk positional embeddings (optional) - important because we're feeding chunk sequences
        self.use_chunk_pos = config.use_chunk_pos_embeddings if config is not None else True
        if self.use_chunk_pos:
            # max chunks same as training config; allocate reasonably large (can be larger than max_chunks)
            max_positions = config.max_chunks if config is not None else 1024
            self.chunk_pos_emb = nn.Embedding(max_positions, self.bart.config.d_model)
        else:
            self.chunk_pos_emb = None

        # Initialize projection weights
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

        if freeze_decoder:
            logger.info("Freezing BART decoder parameters...")
            for param in self.bart.model.decoder.parameters():
                param.requires_grad = False

        # Log parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")
        logger.info(f"Frozen params: {total_params - trainable_params:,}")

    def _encode(self, embeddings: torch.Tensor, chunk_mask: torch.Tensor):
        """
        embeddings: (B, S, embed_dim)
        chunk_mask: (B, S) bool
        Return: encoder_outputs (the same structure returned by HF encoder)
        """
        # Project to model dim
        x = self.projection(embeddings)              # (B, S, d_model)
        x = self.layernorm(x)
        x = self.dropout(x)

        # Add learned chunk positional embeddings if enabled
        if self.use_chunk_pos and self.chunk_pos_emb is not None:
            seq_len = x.size(1)
            pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)  # (B, S)
            pos_emb = self.chunk_pos_emb(pos_ids)  # (B, S, d_model)
            x = x + pos_emb

        # Pass through the BART encoder by providing inputs_embeds (this ensures positional embeddings and encoder layers are applied)
        # BART encoder expects attention_mask as (B, S) where 1 -> keep token, 0 -> pad. Here chunk_mask is bool, convert to int.
        attention_mask = chunk_mask.to(dtype=torch.long)
        encoder_outputs = self.bart.model.encoder(inputs_embeds=x, attention_mask=attention_mask, return_dict=True)
        return encoder_outputs

    def forward(self, embeddings, chunk_mask, target_ids, target_mask):
        """
        embeddings: (B, S, embed_dim)
        chunk_mask: (B, S) bool
        target_ids: (B, T) token ids (int)
        target_mask: (B, T) attention mask for decoder inputs
        """
        # Ensure masks are bool on same device
        if not isinstance(chunk_mask, torch.BoolTensor) and chunk_mask.dtype != torch.bool:
            chunk_mask = chunk_mask.bool()

        encoder_outputs = self._encode(embeddings, chunk_mask)

        # Prepare labels: replace pad token id with -100 so loss ignores them
        labels = target_ids.clone()
        labels[labels == self.bart.config.pad_token_id] = -100

        outputs = self.bart(
            encoder_outputs=encoder_outputs,                      # pass encoder outputs
            attention_mask=chunk_mask.to(dtype=torch.long),
            decoder_attention_mask=target_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    def generate(self, embeddings, chunk_mask, max_len=150, min_len=20, num_beams=6):
        # Ensure masks are bool
        if not isinstance(chunk_mask, torch.BoolTensor) and chunk_mask.dtype != torch.bool:
            chunk_mask = chunk_mask.bool()

        encoder_outputs = self._encode(embeddings, chunk_mask)

        # For generation, pass encoder_outputs as tuple(last_hidden_state, )
        return self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=chunk_mask.to(dtype=torch.long),
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

    # bert_score returns tensors (P, R, F)
    if len(preds) > 0:
        P, R, F1 = bert_score(preds, refs, lang='en', verbose=False)
        bert_p = P.mean().item()
        bert_r = R.mean().item()
        bert_f1 = F1.mean().item()
    else:
        bert_p = bert_r = bert_f1 = 0.0

    return {
        'rouge1': float(np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0),
        'rouge2': float(np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0),
        'rougeL': float(np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0),
        'bert_p': bert_p,
        'bert_r': bert_r,
        'bert_f1': bert_f1,
    }


def evaluate(model, dataloader, tokenizer, config, accelerator, desc="Eval"):
    model.eval()
    preds, refs = [], []
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, disable=not accelerator.is_main_process):
            # Move batch tensors to device
            embeddings = batch['embeddings'].to(accelerator.device)
            chunk_mask = batch['chunk_mask'].to(accelerator.device)
            target_ids = batch['target_ids'].to(accelerator.device)
            target_mask = batch['target_mask'].to(accelerator.device)

            # Calculate loss
            outputs = model(embeddings, chunk_mask, target_ids, target_mask)
            losses.append(outputs.loss.item())

            # Generate predictions
            gen_ids = model.generate(
                embeddings, chunk_mask,
                max_len=config.max_target_len,
                min_len=config.min_target_len,
                num_beams=config.num_beams
            )
            # gen_ids may be on device; decode on CPU-friendly
            preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            refs.extend(batch['target_texts'])

    metrics = compute_metrics(preds, refs)
    metrics['loss'] = float(np.mean(losses) if losses else 0.0)

    return metrics, preds, refs


def setup_logging(output_dir: str):
    """Setup logging to file and console"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set root logger level
    logging.getLogger().setLevel(logging.INFO)

    # File handler
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Remove any existing handlers
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)

    # Add handlers to root logger
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)

    return log_file


def print_sample_outputs(preds: List[str], refs: List[str], doc_texts: List[str], epoch: int, num_samples: int = 5):
    """Print sample model outputs for monitoring progress and save to txt file"""
    logger.info(f"\n{'='*80}")
    logger.info(f"EPOCH {epoch} - DETAILED SAMPLE OUTPUTS")
    logger.info(f"{'='*80}")

    # Also save to text file
    output_file = Path("evaluations") / f"sample_outputs_{epoch}.txt"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"EPOCH {epoch} - SAMPLE OUTPUTS\n")
        f.write("="*80 + "\n\n")
        
        for i in range(min(num_samples, len(preds))):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Reference: {refs[i]}\n")
            f.write(f"Generated: {preds[i]}\n")

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(refs[i], preds[i])
            P, R, F1 = bert_score([preds[i]], [refs[i]], lang='en', verbose=False)

            rouge1 = scores['rouge1'].fmeasure
            rouge2 = scores['rouge2'].fmeasure
            rougeL = scores['rougeL'].fmeasure
            bert_f1 = F1.mean().item()

            f.write(f"ROUGE-1: {rouge1:.4f} | ROUGE-2: {rouge2:.4f} | ROUGE-L: {rougeL:.4f} | BERT-F1: {bert_f1:.4f}\n")
            f.write("-" * 80 + "\n\n")

            # Also log to console
            logger.info(f"\nSample {i+1}:")
            logger.info(f"Reference: {refs[i]}")
            logger.info(f"Generated: {preds[i]}")
            logger.info(f"ROUGE-1: {rouge1:.4f} | ROUGE-2: {rouge2:.4f} | ROUGE-L: {rougeL:.4f} | BERT-F1: {bert_f1:.4f}")
            logger.info("=" * 80)


def train(model, train_dl, val_dl, tokenizer, config, accelerator):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)

    total_steps = len(train_dl) * config.epochs // config.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )

    best_val_loss = float('inf')
    patience = 0
    global_step = 0

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rouge1': [],
        'val_rouge2': [],
        'val_rougeL': [],
        'val_bert_f1': []
    }

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)
        for step, batch in enumerate(pbar):
            # Move to device
            embeddings = batch['embeddings'].to(accelerator.device)
            chunk_mask = batch['chunk_mask'].to(accelerator.device)
            target_ids = batch['target_ids'].to(accelerator.device)
            target_mask = batch['target_mask'].to(accelerator.device)

            outputs = model(embeddings, chunk_mask, target_ids, target_mask)
            loss = outputs.loss / config.grad_accum_steps

            accelerator.backward(loss)
            if (step + 1) % config.grad_accum_steps == 0:
                # clip grads
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * config.grad_accum_steps
            pbar.set_postfix({'loss': f"{loss.item() * config.grad_accum_steps:.4f}"})

        # Calculate average training loss for the epoch
        avg_train_loss = epoch_loss / len(train_dl)
        history['train_loss'].append(avg_train_loss)

        # Evaluate every epoch
        if (epoch + 1) % config.eval_every_n_epochs == 0:
            val_metrics, val_preds, val_refs = evaluate(
                accelerator.unwrap_model(model), val_dl, tokenizer,
                config, accelerator, "Validation"
            )

            history['val_loss'].append(val_metrics['loss'])
            history['val_rouge1'].append(val_metrics['rouge1'])
            history['val_rouge2'].append(val_metrics['rouge2'])
            history['val_rougeL'].append(val_metrics['rougeL'])
            history['val_bert_f1'].append(val_metrics['bert_f1'])

            if accelerator.is_main_process:
                # Log metrics
                logger.info(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
                logger.info(f"Val Loss: {val_metrics['loss']:.4f} | " +
                          f"ROUGE-1: {val_metrics['rouge1']:.4f} | " +
                          f"ROUGE-2: {val_metrics['rouge2']:.4f} | " +
                          f"ROUGE-L: {val_metrics['rougeL']:.4f} | " +
                          f"BERT-F1: {val_metrics['bert_f1']:.4f}")

                # Save metrics to JSON
                metrics_file = Path(config.output_dir) / "training_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(history, f, indent=2)

            # Early stopping based on validation loss
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience = 0
                model_path = Path(config.output_dir) / "best_model.pt"
                torch.save(accelerator.unwrap_model(model).state_dict(), model_path)
                if accelerator.is_main_process:
                    logger.info(f"  New best model saved! Val Loss: {best_val_loss:.4f}")
            else:
                patience += 1
                if accelerator.is_main_process:
                    logger.info(f"  Early stopping patience: {patience}/{config.patience}")

                if patience >= config.patience:
                    if accelerator.is_main_process:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        accelerator.wait_for_everyone()



def compare_base(config, accelerator):
    if accelerator.is_main_process:
        logger.info("Comparing finetuned vs base model")

    test_path = Path(config.encoded_data_dir) / "test_chunks_encoded.pkl"
    model_path = Path(config.output_dir) / "best_model.pt"

    if not test_path.exists() or not model_path.exists():
        logger.error("Test data or model not found")
        return

    tokenizer = BartTokenizer.from_pretrained(config.bart_model)
    test_ds = ChunkDataset(test_path, tokenizer, config.max_chunks, config.max_target_len)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # Finetuned model
    model = BartChunkModel(config.bart_model, config.embedding_dim, freeze_decoder=config.freeze_decoder, config=config)
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
            # Tokenize doc_texts directly for baseline
            input_enc = tokenizer(batch['doc_texts'], max_length=512, padding='max_length',
                                 truncation=True, return_tensors='pt').to(accelerator.device)
            gen_ids = base_model.generate(input_enc['input_ids'], attention_mask=input_enc['attention_mask'],
                                         max_length=config.max_target_len)
            base_preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            base_refs.extend(batch['target_texts'])

    base_metrics = compute_metrics(base_preds, base_refs)

    if accelerator.is_main_process:
        # Create evaluations directory
        eval_dir = Path("evaluations")
        eval_dir.mkdir(exist_ok=True)
        
        # Save table with metrics comparison
        table_file = eval_dir / "table_bart_finetuned.txt"
        with open(table_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON: BART (Finetuned on Chunks) vs BART Base Pretrained\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Metric':<20} {'Finetuned':>15} {'Base':>15} {'Improvement':>15}\n")
            f.write("-" * 70 + "\n")
            for metric in ['rouge1', 'rouge2', 'rougeL', 'bert_f1']:
                ft_val = ft_metrics[metric]
                base_val = base_metrics[metric]
                improvement = ((ft_val - base_val) / base_val * 100) if base_val else 0
                f.write(f"{metric:<20} {ft_val:>15.4f} {base_val:>15.4f} {improvement:>14.1f}%\n")
        
        # Save samples (25 samples with reference and generated summaries)
        samples_file = eval_dir / "samples_bart_finetuned.txt"
        with open(samples_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SAMPLE OUTPUTS: BART Finetuned Model\n")
            f.write("="*80 + "\n\n")
            
            num_samples = min(25, len(ft_preds))
            for i in range(num_samples):
                f.write(f"Sample {i+1}:\n")
                f.write(f"Reference: {ft_refs[i]}\n")
                f.write(f"Generated: {ft_preds[i]}\n")
                f.write("-" * 80 + "\n\n")
        
        logger.info(f"Table saved to {table_file}")
        logger.info(f"Samples saved to {samples_file}")
        logger.info("\nComparison Results:")
        logger.info(f"{'Metric':<15} {'Finetuned':>12} {'Base':>12} {'Improvement':>12}")
        logger.info("-" * 55)
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bert_f1']:
            ft_val = ft_metrics[metric]
            base_val = base_metrics[metric]
            improvement = ((ft_val - base_val) / base_val * 100) if base_val else 0
            logger.info(f"{metric:<15} {ft_val:>12.4f} {base_val:>12.4f} {improvement:>11.1f}%")


def main():
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
        # Setup logging
        log_file = setup_logging(config.output_dir)
        logger.info("=" * 60)
        logger.info("BART Finetuning with BGE-M3 Embeddings")
        logger.info(f"Log file: {log_file}")
        logger.info("=" * 60)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.compare_base:
        compare_base(config, accelerator)
        return

    # Training
    # tokenizer: do not pass use_safetensors for tokenizer
    tokenizer = BartTokenizer.from_pretrained(config.bart_model)

    train_path = Path(config.encoded_data_dir) / "train_chunks_encoded.pkl"
    val_path = Path(config.encoded_data_dir) / "validation_chunks_encoded.pkl"

    if not train_path.exists() or not val_path.exists():
        logger.error("Training or validation data not found")
        return

    train_ds = ChunkDataset(train_path, tokenizer, config.max_chunks, config.max_target_len)
    val_ds = ChunkDataset(val_path, tokenizer, config.max_chunks, config.max_target_len)

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    model = BartChunkModel(
        bart_model=config.bart_model,
        embed_dim=config.embedding_dim,
        freeze_decoder=config.freeze_decoder,
        config=config
    )

    if accelerator.is_main_process:
        logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        logger.info(f"Decoder freezing: {'enabled' if config.freeze_decoder else 'disabled'}")
        logger.info("Starting training...")

    train(model, train_dl, val_dl, tokenizer, config, accelerator)

    # Final evaluation
    if accelerator.is_main_process:
        logger.info("Training completed. Running final evaluation...")


if __name__ == "__main__":
    main()
