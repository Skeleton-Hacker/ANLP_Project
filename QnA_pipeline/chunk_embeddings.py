"""
BGE-M3 Chunk Embedding Generation Pipeline (Accelerate + Multi-GPU Version)

This module generates high-quality embeddings for text chunks using BGE-M3 model,
accelerated across multiple GPUs using Hugging Face Accelerate.

Run using:
    accelerate launch --num_processes 4 --num_machines 1 bge_m3_embeddings_accelerate.py
"""

import logging
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from accelerate import Accelerator, DistributedType

logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class EmbeddingConfig:
    chunked_data_dir: str = "chunked_data"
    output_dir: str = "chunked_data"
    # splits: List[str] = field(default_factory=lambda: ["train", "validation", "test"])
    splits: List[str] = field(default_factory=lambda: ["test"])

    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    batch_size: int = 256
    seed: int = 42


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def generate_chunk_embeddings_accelerated(
    chunks: List[str],
    model_name: str,
    accelerator: Accelerator,
    batch_size: int = 2048,
) -> tuple:
    """Generate embeddings for chunks using accelerate. Returns (embeddings, chunk_indices, local_rank)."""
    local_rank = accelerator.process_index
    device = accelerator.device

    if accelerator.is_local_main_process:
        logger.info(f"[Rank {local_rank}] Loading model {model_name} on device {device}")

    model = SentenceTransformer(model_name, device=device)

    # Each process handles a subset of data
    num_chunks = len(chunks)
    chunks_per_proc = (num_chunks + accelerator.num_processes - 1) // accelerator.num_processes
    start = local_rank * chunks_per_proc
    end = min(start + chunks_per_proc, num_chunks)
    local_chunks = chunks[start:end]
    chunk_indices = list(range(start, end))

    if accelerator.is_local_main_process:
        logger.info(f"Total chunks: {num_chunks} | Each process handles: {len(local_chunks)}")

    # Generate embeddings on GPU
    local_embeddings = model.encode(
        local_chunks,
        convert_to_tensor=True,
        show_progress_bar=(accelerator.is_local_main_process),
        batch_size=batch_size,
        normalize_embeddings=True,
        device=device,
    )

    if accelerator.is_local_main_process:
        logger.info(f"Local embeddings shape: {local_embeddings.shape}, device: {local_embeddings.device}")

    # Convert to CPU to free GPU memory immediately
    local_embeddings = local_embeddings.cpu()
    
    if accelerator.is_local_main_process:
        logger.info(f"Moved embeddings to CPU - shape: {local_embeddings.shape}")

    # Clean up model and intermediate tensors
    del model
    del local_chunks
    torch.cuda.empty_cache()

    # Return embeddings with chunk indices and rank for later reconstruction
    return local_embeddings, chunk_indices, local_rank


def process_and_save_split(split_name: str, config: EmbeddingConfig, accelerator: Accelerator):
    """Process a split (train/val/test) and save embeddings."""
    if accelerator.is_local_main_process:
        logger.info(f"\nProcessing split: {split_name}")

    input_path = Path(config.chunked_data_dir) / f"{split_name}_chunks.pkl"
    if not input_path.exists():
        if accelerator.is_local_main_process:
            logger.error(f"Input file not found: {input_path}")
        return None

    with open(input_path, "rb") as f:
        stories = pickle.load(f)

    chunks = []
    chunk_to_story = []

    for story_id, story_data in stories.items():
        story_chunks = story_data.get("chunks", [])
        for chunk_idx, chunk in enumerate(story_chunks):
            chunks.append(chunk)
            chunk_to_story.append((story_id, chunk_idx))


    if accelerator.is_local_main_process:
        logger.info(f"Loaded {len(chunks)} chunks from {len(stories)} stories")
    
    # Generate embeddings - each process handles its own
    embeddings, chunk_indices, local_rank = generate_chunk_embeddings_accelerated(
        chunks,
        config.embedding_model_name,
        accelerator,
        config.batch_size,
    )

    # Save per-process embeddings with indices
    accelerator.wait_for_everyone()
    output_path = Path(config.output_dir) / f"{split_name}_chunks_encoded_rank{local_rank}.pkl"
    
    logger.info(f"[Rank {local_rank}] Saving {len(chunk_indices)} embeddings to {output_path}...")
    embeddings_np = embeddings.numpy() if hasattr(embeddings, 'numpy') else embeddings
    
    rank_data = {
        "chunk_indices": chunk_indices,
        "embeddings": embeddings_np,
        "metadata": {
            "rank": local_rank,
            "num_embeddings": len(chunk_indices),
            "embedding_dim": embeddings_np.shape[1],
        }
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(rank_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"✓ [Rank {local_rank}] Saved embeddings - File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")

    # Only main process merges and reconstructs
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        logger.info("Main process: Merging embeddings from all ranks...")
        
        # Load embeddings from all ranks
        all_embeddings_dict = {}
        for rank in range(accelerator.num_processes):
            rank_path = Path(config.output_dir) / f"{split_name}_chunks_encoded_rank{rank}.pkl"
            if rank_path.exists():
                with open(rank_path, "rb") as f:
                    rank_data = pickle.load(f)
                    for idx, chunk_idx in enumerate(rank_data["chunk_indices"]):
                        all_embeddings_dict[chunk_idx] = rank_data["embeddings"][idx]
        
        # Reconstruct in order
        embeddings_list = [all_embeddings_dict[i] for i in range(len(chunks))]
        embeddings_np = np.array(embeddings_list)
        logger.info(f"Merged embeddings shape: {embeddings_np.shape}")
        
        embedded_data = {
            "stories": {},
            "metadata": {
                "split": split_name,
                "num_stories": len(stories),
                "num_chunks": len(chunks),
                "embedding_model": config.embedding_model_name,
                "embedding_dim": embeddings_np.shape[1],
                "timestamp": datetime.now().isoformat(),
            },
        }

        logger.info("Reconstructing story structure with embeddings...")
        for idx, (story_id, chunk_idx) in enumerate(chunk_to_story):
            if story_id not in embedded_data["stories"]:
                embedded_data["stories"][story_id] = {
                    "document": stories[story_id]["document"],
                    "questions": stories[story_id]["questions"],
                    "answers": stories[story_id]["answers"],
                    "chunks": [],
                    "chunk_embeddings": [],
                }
            embedded_data["stories"][story_id]["chunks"].append(chunks[idx])
            embedded_data["stories"][story_id]["chunk_embeddings"].append(embeddings_np[idx])

        for story_id in embedded_data["stories"]:
            embedded_data["stories"][story_id]["chunk_embeddings"] = np.array(
                embedded_data["stories"][story_id]["chunk_embeddings"]
            )

        output_path = Path(config.output_dir) / f"{split_name}_chunks_encoded.pkl"
        logger.info(f"Saving merged data to {output_path}...")
        with open(output_path, "wb") as f:
            pickle.dump(embedded_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"✓ Saved embedded data to {output_path}")
        logger.info(f"✓ File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
        
        # Clean up rank files
        logger.info("Cleaning up per-rank files...")
        for rank in range(accelerator.num_processes):
            rank_path = Path(config.output_dir) / f"{split_name}_chunks_encoded_rank{rank}.pkl"
            if rank_path.exists():
                rank_path.unlink()
                logger.info(f"Deleted {rank_path.name}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    accelerator = Accelerator()
    config = EmbeddingConfig()
    set_seed(config.seed)

    if accelerator.is_local_main_process:
        logger.info("=" * 80)
        logger.info("Accelerated BGE-M3 Embedding Pipeline")
        logger.info("=" * 80)
        logger.info(f"Num processes: {accelerator.num_processes}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Distributed type: {accelerator.distributed_type}")
        logger.info(f"Embedding model: {config.embedding_model_name}")
        logger.info(f"Batch size: {config.batch_size}")
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()

    for split in config.splits:
        process_and_save_split(split, config, accelerator)
        accelerator.wait_for_everyone()
        
        # Clear GPU cache after each split
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if accelerator.is_local_main_process:
            logger.info(f"GPU cache cleared after processing {split} split")

    if accelerator.is_local_main_process:
        logger.info("\n" + "=" * 80)
        logger.info("✓ All splits processed successfully!")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()