"""
BGE-large Chunk Embedding Generation Pipeline

This module generates high-quality embeddings for text chunks using BGE-large-en-v1.5 model.
It includes:
- Loading chunked stories from pickle files
- Generating BGE-large embeddings (1024-dim) for all chunks
- Saving embeddings in a structured format for downstream use

BGE-large-en-v1.5 produces 1024-dimensional embeddings which will be projected to BART's
embedding space in the fine-tuning pipeline.
"""

import logging
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class EmbeddingConfig:
    """Configuration for BGE-large embedding generation."""
    # Data settings
    chunked_data_dir: str = "chunked_data"
    output_dir: str = "chunked_data"
    splits: List[str] = field(default_factory=lambda: ["train", "validation", "test"])
    
    # Embedding settings
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"  # BGE-large for high-quality embeddings
    embedding_dim: int = 1024  # BGE-large produces 1024-dim embeddings
    
    # Processing settings
    batch_size: int = 32  # Smaller batch size for larger model
    num_workers: int = 4
    
    # Random seed
    seed: int = 42


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def generate_chunk_embeddings(
    chunks: List[str],
    model_name: str,
    device: str = "cuda",
    batch_size: int = 32
) -> np.ndarray:
    """Generate BGE-large embeddings for chunks.
    
    Args:
        chunks: List of chunk strings.
        model_name: Name of the sentence transformer model (BGE-large).
        device: Device to use for computation ('cuda' or 'cpu').
        batch_size: Batch size for embedding generation.
        
    Returns:
        Numpy array of shape (num_chunks, 1024).
    """
    logger.info(f"Generating BGE-large embeddings for {len(chunks)} chunks...")
    
    # Load the model using safetensors to avoid torch.load vulnerability issues
    # on older torch versions. The model is loaded into memory on the CPU first.
    model = SentenceTransformer(model_name, device='cpu', use_auth_token=False)
    model.to(device)  # Move model to the target device (e.g., 'cuda:0')
    
    embeddings = model.encode(
        chunks,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True,
        batch_size=batch_size,
        normalize_embeddings=True  # Normalize embeddings
    )
    
    # Convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    
    logger.info(f"Generated embeddings with shape: {embeddings_np.shape}")
    
    return embeddings_np


def process_and_save_split(
    split_name: str,
    config: EmbeddingConfig,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Process a split and save BGE-large embeddings.
    
    Args:
        split_name: Name of the split ('train', 'validation', 'test').
        config: Configuration object.
        device: Device to use for computation ('cuda' or 'cpu').
        
    Returns:
        Dictionary with embedded data.
    """
    logger.info(f"\nProcessing {split_name} split...")
    
    # Load chunks
    input_path = Path(config.chunked_data_dir) / f"{split_name}_chunks.pkl"
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return None
    
    with open(input_path, 'rb') as f:
        stories = pickle.load(f)
    
    # Extract chunks with story mapping
    chunks = []
    chunk_to_story = []  # (story_id, chunk_index) for each chunk
    
    story_items = list(stories.items())
    for story_id, story_data in tqdm(
        story_items, 
        desc="Extracting chunks with mapping"
    ):
        story_chunks = story_data.get('chunks', [])
        for chunk_idx, chunk in enumerate(story_chunks):
            chunks.append(chunk)
            chunk_to_story.append((story_id, chunk_idx))
    
    logger.info(f"Loaded {len(chunks)} chunks from {len(stories)} stories")
    
    # Generate BGE-large embeddings
    embeddings = generate_chunk_embeddings(
        chunks,
        config.embedding_model_name,
        device,
        config.batch_size
    )
    
    # Create embedded data structure
    embedded_data = {
        'stories': {},
        'metadata': {
            'split': split_name,
            'num_stories': len(stories),
            'num_chunks': len(chunks),
            'embedding_model': config.embedding_model_name,
            'embedding_dim': embeddings.shape[1],
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Reconstruct story structure with embeddings
    logger.info("Reconstructing story structure with embeddings...")
    for idx, (story_id, chunk_idx) in tqdm(
        enumerate(chunk_to_story), 
        desc="Reconstructing stories", 
        total=len(chunk_to_story)
    ):
        if story_id not in embedded_data['stories']:
            # Copy original story data
            embedded_data['stories'][story_id] = {
                'document': stories[story_id]['document'],
                'questions': stories[story_id]['questions'],
                'answers': stories[story_id]['answers'],
                'chunks': [],
                'chunk_embeddings': []
            }
        
        embedded_data['stories'][story_id]['chunks'].append(chunks[idx])
        embedded_data['stories'][story_id]['chunk_embeddings'].append(
            embeddings[idx]
        )
    
    # Convert embedding lists to numpy arrays
    logger.info("Converting embedding lists to numpy arrays...")
    story_ids = list(embedded_data['stories'].keys())
    for story_id in tqdm(story_ids, desc="Converting to numpy arrays"):
        embedded_data['stories'][story_id]['chunk_embeddings'] = np.array(
            embedded_data['stories'][story_id]['chunk_embeddings']
        )
    
    # Save embedded data
    output_path = Path(config.output_dir) / f"{split_name}_chunks_encoded.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(embedded_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Saved embedded data to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    return embedded_data


def main():
    """Main BGE-large embedding generation pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = EmbeddingConfig()
    
    # Set seed
    set_seed(config.seed)
    
    # Detailed device detection
    logger.info("=" * 80)
    logger.info("Device and CUDA Diagnostics")
    logger.info("=" * 80)
    
    cuda_available = torch.cuda.is_available()
    logger.info(f"PyTorch reports CUDA available: {cuda_available}")
    
    if cuda_available:
        device = "cuda"
        logger.info(f"CUDA version detected by PyTorch: {torch.version.cuda}")
        logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device}")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(current_device)}")
    else:
        device = "cpu"
        logger.warning("CUDA not available. Falling back to CPU.")
        logger.warning("Processing will be significantly slower.")

    logger.info("=" * 80)
    logger.info("BGE-large Chunk Embedding Generation Pipeline")
    logger.info("=" * 80)
    logger.info(f"Using device: {device}")
    logger.info(f"Embedding model: {config.embedding_model_name}")
    logger.info(f"Embedding dimension: {config.embedding_dim}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info("=" * 80)
    
    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Verify chunked data exists
    train_path = Path(config.chunked_data_dir) / "train_chunks.pkl"
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Please run semantic_chunking.py first to generate chunked data.")
        return
    
    # Process each split
    logger.info("\n" + "=" * 80)
    logger.info("Generating BGE-large Embeddings for All Splits")
    logger.info("=" * 80)
    
    for split in config.splits:
        try:
            process_and_save_split(split, config, device)
        except Exception as e:
            logger.error(f"Error processing {split} split: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Completed Successfully!")
    logger.info("=" * 80)
    logger.info(f"Embedded data saved in: {config.output_dir}")
    
    # List output files
    output_files = list(Path(config.output_dir).glob("*_encoded.pkl"))
    if output_files:
        logger.info("\nEmbedded files:")
        for f in sorted(output_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  - {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
