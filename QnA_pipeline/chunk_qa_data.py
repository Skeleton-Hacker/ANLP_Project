"""
Process QA datasets with semantic chunking and embeddings.
Combines document chunks with question embeddings for the QA task.
"""
import logging
import pickle
import os
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from accelerate import Accelerator
from dataclasses import dataclass
from typing import Dict, List, Optional

from load_dataset import load_train, load_validation, load_test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class QAChunkConfig:
    dataset_name: str = "deepmind/narrativeqa"
    embedding_model: str = "BAAI/bge-large-en-v1.5"  # Changed to match chunking model
    output_dir: str = "chunked_data"
    batch_size: int = 32
    max_samples: Optional[int] = None
    chunked_data_dir: str = "chunked_data"


def encode_questions_batch(
    questions: List[str],
    model: SentenceTransformer,
    batch_size: int = 32
) -> np.ndarray:
    """Encode multiple questions in batches using BGE model."""
    embeddings = model.encode(
        questions, 
        convert_to_numpy=True, 
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False
    )
    return embeddings


def process_qa_split(
    config: QAChunkConfig,
    split: str,
    stories_dict: Dict,
    chunked_documents: Dict,
    accelerator: Accelerator
):
    """
    Process QA dataset split with pre-chunked documents.
    
    Args:
        config: Configuration
        split: Dataset split name
        stories_dict: Dictionary of stories from load_qa_dataset
        chunked_documents: Pre-chunked and embedded documents
        accelerator: Accelerator instance
    """
    logger.info(f"Processing {split} split...")
    
    # Load embedding model
    model = SentenceTransformer(config.embedding_model)
    model = model.to(accelerator.device)
    logger.info(f"Loaded embedding model: {config.embedding_model}")
    
    processed_samples = {}
    sample_idx = 0
    
    matched_stories = 0
    unmatched_stories = 0
    total_questions = 0
    
    for story_id, story_data in tqdm(stories_dict.items(), 
                                     desc=f"Processing {split}", 
                                     disable=not accelerator.is_main_process):
        # Get chunked document
        if story_id not in chunked_documents:
            logger.debug(f"Story {story_id} not found in chunked data, skipping")
            unmatched_stories += 1
            continue
        
        matched_stories += 1
        doc_chunks = chunked_documents[story_id]['chunks']
        chunk_embeddings = chunked_documents[story_id]['chunk_embeddings']
        
        # Process each question-answer pair for this story
        questions = story_data['questions']
        answers = story_data['answers']
        
        # Batch encode all questions for this story
        question_embeddings = encode_questions_batch(questions, model, batch_size=config.batch_size)
        
        for q, a, q_emb in zip(questions, answers, question_embeddings):
            # Store processed sample
            sample_id = f"{split}_{sample_idx}"
            processed_samples[sample_id] = {
                'story_id': story_id,
                'question': q,
                'question_embedding': q_emb,
                'answer': a,
                'chunks': doc_chunks,
                'chunk_embeddings': chunk_embeddings,
                'document': story_data['document']
            }
            sample_idx += 1
            total_questions += 1
    
    logger.info(f"{split} split processing complete:")
    logger.info(f"  - Matched stories: {matched_stories}")
    logger.info(f"  - Unmatched stories: {unmatched_stories}")
    logger.info(f"  - Total QA pairs: {total_questions}")
    
    return processed_samples


def main():
    config = QAChunkConfig()
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        logger.info("="*80)
        logger.info("QA Dataset Processing with Existing Chunks")
        logger.info(f"Dataset: {config.dataset_name}")
        logger.info(f"Chunked data directory: {config.chunked_data_dir}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info("="*80)
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load chunked documents
    chunked_data_path = Path(config.chunked_data_dir)
    
    if not chunked_data_path.exists():
        logger.error(f"Chunked data directory not found: {chunked_data_path}")
        logger.error("Please ensure chunked .pkl files are in chunked_data/")
        return
    
    # Process each split
    for split_name, load_fn in [("train", load_train), 
                                 ("validation", load_validation), 
                                 ("test", load_test)]:
        if accelerator.is_main_process:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {split_name} split")
            logger.info(f"{'='*80}")
        
        # Load chunked documents for this split
        chunk_file = chunked_data_path / f"{split_name}_chunks_encoded.pkl"
        
        if not chunk_file.exists():
            logger.warning(f"Chunked data file not found: {chunk_file}")
            logger.warning(f"Skipping {split_name} split")
            continue
        
        logger.info(f"Loading chunked data from: {chunk_file}")
        with open(chunk_file, 'rb') as f:
            chunked_data = pickle.load(f)
        
        # Extract stories from the chunked data structure
        if 'stories' in chunked_data:
            chunked_documents = chunked_data['stories']
        else:
            # If it's already in the right format
            chunked_documents = chunked_data
        
        logger.info(f"Loaded {len(chunked_documents)} chunked stories for {split_name}")
        
        # Load QA data
        logger.info(f"Loading QA data for {split_name}...")
        stories_dict = load_fn(
            dataset_name=config.dataset_name,
            max_samples=config.max_samples,
            group_by_story=True
        )
        
        logger.info(f"Loaded {len(stories_dict)} stories from {split_name}")
        
        # Process QA pairs
        processed = process_qa_split(
            config, 
            split_name, 
            stories_dict, 
            chunked_documents, 
            accelerator
        )
        
        if accelerator.is_main_process:
            # Save processed data
            output_file = Path(config.output_dir) / f"{split_name}_qa_encoded.pkl"
            logger.info(f"Saving {len(processed)} QA samples to {output_file}...")
            
            with open(output_file, 'wb') as f:
                pickle.dump({
                    'samples': processed,
                    'metadata': {
                        'dataset': config.dataset_name,
                        'split': split_name,
                        'num_samples': len(processed),
                        'embedding_model': config.embedding_model,
                        'created': datetime.now().isoformat()
                    }
                }, f)
            
            logger.info(f"✓ Saved {split_name}_qa_encoded.pkl ({output_file.stat().st_size / (1024*1024):.2f} MB)")
    
    if accelerator.is_main_process:
        logger.info("\n" + "="*80)
        logger.info("Processing complete!")
        logger.info(f"QA encoded files saved to: {config.output_dir}")
        logger.info("="*80)
        
        # List created files
        logger.info("\nCreated files:")
        for f in sorted(Path(config.output_dir).glob("*_qa_encoded.pkl")):
            logger.info(f"  - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()