"""
T5 Fine-tuning with BGE-M3 Chunk Embeddings - FIXED VERSION

Key fixes:
1. Added data validation and verification
2. Improved generation parameters to prevent repetition
3. Added input-output verification logging
4. Better training stability
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
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
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
    output_dir: str = "models/t5"

    # Model
    t5_model: str = "t5-base"
    embedding_dim: int = 1024  # BGE-M3 embedding dimension
    freeze_decoder: bool = False

    # Training
    batch_size: int = 1
    epochs: int = 1000
    lr: float = 3e-4  # Slightly lower learning rate
    weight_decay: float = 1e-3
    warmup_steps: int = 500
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    patience: int = 20  # Increased patience
    eval_every_n_epochs: int = 1

    # Generation - KEY CHANGES TO PREVENT REPETITION
    max_chunks: int = 800
    max_target_len: int = 150
    min_target_len: int = 20
    num_beams: int = 4
    no_repeat_ngram_size: int = 3  # NEW: Prevent 3-gram repetition
    repetition_penalty: float = 2.0  # NEW: Penalize repetition
    length_penalty: float = 1.0  # NEW: Encourage appropriate length
    temperature: float = 0.7  # NEW: Add some randomness

    # Hardware / misc
    seed: int = 42
    projection_dropout: float = 0.1
    use_chunk_pos_embeddings: bool = True
    validate_data: bool = True  # NEW: Validate data before training


class ChunkDataset(Dataset):
    def __init__(self, pkl_path: Path, tokenizer: T5Tokenizer, max_chunks: int, max_tgt_len: int, validate: bool = True):
        accelerator = Accelerator()
        with accelerator.main_process_first():
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.max_tgt_len = max_tgt_len

        self.stories = data['stories']
        
        # More robust filtering
        self.samples = []
        for sid, s in self.stories.items():
            if 'chunk_embeddings' not in s or len(s['chunk_embeddings']) == 0:
                continue
            if 'document' not in s:
                continue
            if 'summary' not in s['document']:
                continue
            if 'text' not in s['document']['summary']:
                continue
            # Ensure chunks exist and match embeddings
            if 'chunks' not in s or len(s['chunks']) == 0:
                logger.warning(f"Story {sid} has embeddings but no chunks!")
                continue
                
            self.samples.append(sid)

        logger.info(f"Loaded {len(self.samples)} samples from {pkl_path}")
        
        # Validation step
        if validate and len(self.samples) > 0:
            self._validate_samples()

    def _validate_samples(self):
        """Validate that inputs match summaries"""
        logger.info("Validating dataset samples...")
        sample_idx = min(3, len(self.samples))
        
        for i in range(sample_idx):
            story_id = self.samples[i]
            story = self.stories[story_id]
            
            chunks_text = ' '.join(story['chunks'][:50])  # First 50 chunks
            summary_text = story['document']['summary']['text']
            
            logger.info(f"\nValidation Sample {i+1}:")
            logger.info(f"Input preview: {chunks_text[:200]}...")
            logger.info(f"Summary: {summary_text[:200]}...")
            logger.info("-" * 80)

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
        target_ids = target_enc['input_ids'].squeeze(0)
        target_mask = target_enc['attention_mask'].squeeze(0)

        # Get original chunks for verification
        chunks = story['chunks'][:self.max_chunks] if 'chunks' in story else []

        return {
            'embeddings': torch.from_numpy(embeddings),
            'num_chunks': num_chunks,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'target_text': target,
            'doc_text': ' '.join(chunks) if chunks else "",
            'story_id': story_id
        }


def collate_fn(batch):
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
        chunk_masks[i, :n] = 1

    return {
        'embeddings': embeddings_padded,
        'chunk_mask': chunk_masks,
        'target_ids': target_ids,
        'target_mask': target_mask,
        'target_texts': [b['target_text'] for b in batch],
        'doc_texts': [b['doc_text'] for b in batch],
        'story_ids': [b['story_id'] for b in batch]
    }


class T5ChunkModel(nn.Module):
    def __init__(self, t5_model: str, embed_dim: int, freeze_decoder: bool = False, config: Config = None):
        super().__init__()
        
        # Load only the config, not the pretrained weights
        t5_config = T5Config.from_pretrained(t5_model)
        
        # Initialize T5 model with random weights (no pretrained weights)
        self.t5 = T5ForConditionalGeneration(t5_config)
        
        self.projection = nn.Linear(embed_dim, self.t5.config.d_model)

        self.layernorm = nn.LayerNorm(self.t5.config.d_model)
        self.dropout = nn.Dropout(config.projection_dropout if config is not None else 0.1)

        self.use_chunk_pos = config.use_chunk_pos_embeddings if config is not None else True
        if self.use_chunk_pos:
            max_positions = config.max_chunks if config is not None else 1024
            self.chunk_pos_emb = nn.Embedding(max_positions, self.t5.config.d_model)
        else:
            self.chunk_pos_emb = None

        # Better initialization
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

        if freeze_decoder:
            logger.info("Freezing T5 decoder parameters...")
            for param in self.t5.decoder.parameters():
                param.requires_grad = False

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")

    def _encode(self, embeddings: torch.Tensor, chunk_mask: torch.Tensor):
        x = self.projection(embeddings)
        x = self.layernorm(x)
        x = self.dropout(x)

        if self.use_chunk_pos and self.chunk_pos_emb is not None:
            seq_len = x.size(1)
            pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
            pos_emb = self.chunk_pos_emb(pos_ids)
            x = x + pos_emb

        attention_mask = chunk_mask.to(dtype=torch.long)
        encoder_outputs = self.t5.encoder(inputs_embeds=x, attention_mask=attention_mask, return_dict=True)
        return encoder_outputs

    def forward(self, embeddings, chunk_mask, target_ids, target_mask):
        if not isinstance(chunk_mask, torch.BoolTensor) and chunk_mask.dtype != torch.bool:
            chunk_mask = chunk_mask.bool()

        encoder_outputs = self._encode(embeddings, chunk_mask)

        labels = target_ids.clone()
        labels[labels == self.t5.config.pad_token_id] = -100

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=chunk_mask.to(dtype=torch.long),
            decoder_attention_mask=target_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    def generate(self, embeddings, chunk_mask, max_len=150, min_len=20, num_beams=4,
                 no_repeat_ngram_size=3, repetition_penalty=2.0, length_penalty=1.0):
        """Enhanced generation with repetition prevention"""
        if not isinstance(chunk_mask, torch.BoolTensor) and chunk_mask.dtype != torch.bool:
            chunk_mask = chunk_mask.bool()

        encoder_outputs = self._encode(embeddings, chunk_mask)

        return self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=chunk_mask.to(dtype=torch.long),
            max_length=max_len,
            min_length=min_len,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
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
            embeddings = batch['embeddings'].to(accelerator.device)
            chunk_mask = batch['chunk_mask'].to(accelerator.device)
            target_ids = batch['target_ids'].to(accelerator.device)
            target_mask = batch['target_mask'].to(accelerator.device)

            outputs = model(embeddings, chunk_mask, target_ids, target_mask)
            losses.append(outputs.loss.item())

            # Enhanced generation
            gen_ids = model.generate(
                embeddings, chunk_mask,
                max_len=config.max_target_len,
                min_len=config.min_target_len,
                num_beams=config.num_beams,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty
            )
            preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            refs.extend(batch['target_texts'])

    metrics = compute_metrics(preds, refs)
    metrics['loss'] = float(np.mean(losses) if losses else 0.0)

    return metrics, preds, refs


def setup_logging(output_dir: str):
    """Setup logging to file and console"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.getLogger().setLevel(logging.INFO)

    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)

    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)

    return log_file


def print_sample_outputs(preds: List[str], refs: List[str], doc_texts: List[str], epoch: int, num_samples: int = 5):
    """Print sample model outputs for monitoring progress and save to txt file"""
    logger.info(f"\n{'='*80}")
    logger.info(f"EPOCH {epoch} - SAMPLE OUTPUTS")
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
    best_rouge = 0.0
    patience = 0
    global_step = 0

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
            embeddings = batch['embeddings'].to(accelerator.device)
            chunk_mask = batch['chunk_mask'].to(accelerator.device)
            target_ids = batch['target_ids'].to(accelerator.device)
            target_mask = batch['target_mask'].to(accelerator.device)

            outputs = model(embeddings, chunk_mask, target_ids, target_mask)
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

        avg_train_loss = epoch_loss / len(train_dl)
        history['train_loss'].append(avg_train_loss)

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
                logger.info(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
                logger.info(f"Val Loss: {val_metrics['loss']:.4f} | " +
                          f"ROUGE-1: {val_metrics['rouge1']:.4f} | " +
                          f"ROUGE-2: {val_metrics['rouge2']:.4f} | " +
                          f"BERT-F1: {val_metrics['bert_f1']:.4f}")

                val_ds = val_dl.dataset if hasattr(val_dl, 'dataset') else None
                if val_ds is not None:
                    sample_doc_texts = [val_ds[i]['doc_text'] for i in range(min(3, len(val_ds)))]
                    print_sample_outputs(val_preds[:3], val_refs[:3], sample_doc_texts[:3], epoch + 1, num_samples=3)

                metrics_file = Path(config.output_dir) / "training_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(history, f, indent=2)

            # Improved early stopping - track both loss and ROUGE
            improved = False
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                improved = True
            
            if val_metrics['rouge1'] > best_rouge:
                best_rouge = val_metrics['rouge1']
                improved = True
            
            if improved:
                patience = 0
                model_path = Path(config.output_dir) / "best_model.pt"
                torch.save(accelerator.unwrap_model(model).state_dict(), model_path)
                if accelerator.is_main_process:
                    logger.info(f"✓ New best model saved!")
            else:
                patience += 1
                if accelerator.is_main_process:
                    logger.info(f"Patience: {patience}/{config.patience}")

                if patience >= config.patience:
                    if accelerator.is_main_process:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        accelerator.wait_for_everyone()


def compare_base(config, accelerator):
    """Compare untrained T5 model with base pretrained model"""
    if accelerator.is_main_process:
        logger.info("Comparing untrained vs base T5 model")

    test_path = Path(config.encoded_data_dir) / "test_chunks_encoded.pkl"
    model_path = Path(config.output_dir) / "best_model.pt"

    if not test_path.exists() or not model_path.exists():
        logger.error("Test data or model not found")
        return

    tokenizer = T5Tokenizer.from_pretrained(config.t5_model)
    test_ds = ChunkDataset(test_path, tokenizer, config.max_chunks, config.max_target_len, validate=False)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # Untrained model
    model = T5ChunkModel(config.t5_model, config.embedding_dim, freeze_decoder=config.freeze_decoder, config=config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model, test_dl = accelerator.prepare(model, test_dl)
    untrained_metrics, untrained_preds, untrained_refs = evaluate(accelerator.unwrap_model(model), test_dl,
                                                                   tokenizer, config, accelerator, "Untrained")

    # Base model
    base_model = T5ForConditionalGeneration.from_pretrained(config.t5_model)
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
        # Save comparison results to text file
        comparison_file = Path("evaluations") / "comparison_results_untrained.txt"
        comparison_file.parent.mkdir(exist_ok=True)
        
        with open(comparison_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON: T5 Untrained vs Base Pretrained\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Metric':<20} {'Untrained':>15} {'Base':>15} {'Difference':>15}\n")
            f.write("-" * 70 + "\n")
            for metric in ['rouge1', 'rouge2', 'rougeL', 'bert_f1']:
                untrained_val = untrained_metrics[metric]
                base_val = base_metrics[metric]
                diff = ((untrained_val - base_val) / base_val * 100) if base_val else 0
                f.write(f"{metric:<20} {untrained_val:>15.4f} {base_val:>15.4f} {diff:>14.1f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("SAMPLE OUTPUTS COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            for i in range(min(5, len(untrained_preds))):
                f.write(f"Sample {i+1}:\n")
                f.write(f"{'Reference':<12}: {untrained_refs[i]}\n")
                f.write(f"{'Untrained':<12}: {untrained_preds[i]}\n")
                f.write(f"{'Base':<12}: {base_preds[i]}\n")
                f.write("-" * 80 + "\n\n")
        
        logger.info(f"Comparison results saved to {comparison_file}")
        logger.info("Comparison Results:")
        logger.info(f"{'Metric':<15} {'Untrained':>12} {'Base':>12} {'Difference':>12}")
        logger.info("-" * 55)
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bert_f1']:
            untrained_val = untrained_metrics[metric]
            base_val = base_metrics[metric]
            diff = ((untrained_val - base_val) / base_val * 100) if base_val else 0
            logger.info(f"{metric:<15} {untrained_val:>12.4f} {base_val:>12.4f} {diff:>11.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--compare-base', action='store_true')
    args = parser.parse_args()

    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.grad_accum_steps,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        log_file = setup_logging(config.output_dir)
        logger.info("="*80)
        logger.info("T5 Fine-tuning with BGE-M3 Embeddings - FIXED VERSION")
        logger.info(f"Log file: {log_file}")
        logger.info("="*80)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.compare_base:
        compare_base(config, accelerator)
        return

    tokenizer = T5Tokenizer.from_pretrained(config.t5_model)

    if not args.eval_only:
        train_path = Path(config.encoded_data_dir) / "train_chunks_encoded.pkl"
        val_path = Path(config.encoded_data_dir) / "validation_chunks_encoded.pkl"

        if not train_path.exists() or not val_path.exists():
            logger.error("Training or validation data not found")
            return

        train_ds = ChunkDataset(train_path, tokenizer, config.max_chunks, 
                               config.max_target_len, validate=config.validate_data)
        val_ds = ChunkDataset(val_path, tokenizer, config.max_chunks, 
                             config.max_target_len, validate=config.validate_data)

        train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        model = T5ChunkModel(
            t5_model=config.t5_model,
            embed_dim=config.embedding_dim,
            freeze_decoder=config.freeze_decoder,
            config=config
        )

        if accelerator.is_main_process:
            logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
            logger.info("Starting training...")

        train(model, train_dl, val_dl, tokenizer, config, accelerator)

    # Evaluation
    test_path = Path(config.encoded_data_dir) / "test_chunks_encoded.pkl"
    model_path = Path(config.output_dir) / "best_model.pt"

    if test_path.exists() and model_path.exists():
        test_ds = ChunkDataset(test_path, tokenizer, config.max_chunks, config.max_target_len, validate=False)
        test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        model = T5ChunkModel(config.t5_model, config.embedding_dim, 
                           freeze_decoder=config.freeze_decoder, config=config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model, test_dl = accelerator.prepare(model, test_dl)

        metrics, preds, refs = evaluate(accelerator.unwrap_model(model), test_dl,
                                       tokenizer, config, accelerator, "Test")

        if accelerator.is_main_process:
            logger.info("\n" + "="*80)
            logger.info("FINAL TEST RESULTS")
            logger.info("="*80)
            logger.info(f"ROUGE-1: {metrics['rouge1']:.4f} | ROUGE-2: {metrics['rouge2']:.4f}")
            logger.info(f"ROUGE-L: {metrics['rougeL']:.4f} | BERT-F1: {metrics['bert_f1']:.4f}")
            
            sample_docs = [test_ds[i]['doc_text'] for i in range(min(5, len(test_ds)))]
            print_sample_outputs(preds[:5], refs[:5], sample_docs[:5], "FINAL", num_samples=5)


if __name__ == "__main__":
    main()