import logging
import re
import pickle
import os
# Prevent huggingface/tokenizers parallelism warnings when using forks.
# Set before any tokenizers usage to avoid the "forked after parallelism" warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from load_dataset import load_train, load_validation, load_test
from tqdm import tqdm
from accelerate import Accelerator
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


@dataclass
class Config:
    dataset_name: str = "deepmind/narrativeqa"
    model_name: str = "BAAI/bge-large-en-v1.5"
    threshold: float = 0.5
    max_samples: Optional[int] = None
    output_dir: str = "chunked_data"
    splits: List[str] = field(default_factory=lambda: ["train", "validation", "test"])
    batch_size: int = 8
    num_workers: int = 4
    dataloader_workers: int = 4


class SemanticSentenceChunker:
    """Simple sentence splitter for preprocessing."""
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class DocumentDataset(Dataset):
    """Dataset for batch processing of documents."""
    
    def __init__(self, story_items: List[Tuple[str, Dict[str, Any]]]):
        """Initialize dataset with story items.
        
        Args:
            story_items: List of (story_id, story_data) tuples.
        """
        self.story_items = story_items
        self.sentence_splitter = SemanticSentenceChunker()
    
    def __len__(self):
        return len(self.story_items)
    
    def __getitem__(self, idx):
        story_id, story_data = self.story_items[idx]
        doc_text = story_data['document'].get('text', '')
        sentences = self.sentence_splitter.split_into_sentences(doc_text) if doc_text else []
        
        return {
            'story_id': story_id,
            'sentences': sentences,
            'num_sentences': len(sentences)
        }


class GraphBasedChunker:
    """Graph-based semantic chunker using similarity threshold."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        threshold: float = 0.5,
        accelerator: Optional[Accelerator] = None
    ):
        """Initialize the chunker with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model.
            device: Device to run the model on ("cpu" or "cuda").
            threshold: Similarity threshold for chunking (0.0 to 1.0).
            accelerator: Accelerator instance for distributed processing.
        """
        self.accelerator = accelerator
        # Load model with safetensors
        self.model = SentenceTransformer(model_name, device='cpu', use_auth_token=False)
        self.model.to(device)
        self.threshold = threshold
        self.sentence_splitter = SemanticSentenceChunker()
        
        if accelerator and accelerator.is_main_process:
            logger.info(f"Initialized GraphBasedChunker with model={model_name}, device={device}, threshold={threshold}")
        elif not accelerator:
            logger.info(f"Initialized GraphBasedChunker with model={model_name}, device={device}, threshold={threshold}")
    
    def chunk_sentences(self, sentences: List[str]) -> List[str]:
        """Chunk pre-split sentences using graph-based semantic similarity.
        
        Args:
            sentences: List of sentences to chunk.
            
        Returns:
            List of chunk strings, where each chunk is concatenated sentences.
        """
        n = len(sentences)
        
        if n == 0:
            return []
        
        if n == 1:
            return [sentences[0]]
        
        # Compute sentence embeddings
        embeddings = self.model.encode(
            sentences,
            convert_to_tensor=True,
            device=self.model.device,
            show_progress_bar=False,
            batch_size=32  # Process in batches for better GPU utilization
        )
        
        # Compute pairwise cosine similarity
        sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        
        # Graph-based chunking using BFS
        visited = [False] * n
        chunks = []
        
        for i in range(n):
            if not visited[i]:
                chunk_indices = []
                queue = [i]
                visited[i] = True
                
                while queue:
                    u = queue.pop(0)
                    chunk_indices.append(u)
                    
                    for v in range(n):
                        if sim_matrix[u, v] >= self.threshold and not visited[v]:
                            visited[v] = True
                            queue.append(v)
                
                # Sort indices to maintain original sentence order within chunk
                chunk_indices.sort()
                chunk_text = " ".join([sentences[idx] for idx in chunk_indices])
                chunks.append(chunk_text)
        
        return chunks
    
    def chunk_document(self, doc_text: str) -> List[str]:
        """Chunk a document using graph-based semantic similarity.
        
        Args:
            doc_text: The full document text to chunk.
            
        Returns:
            List of chunk strings, where each chunk is concatenated sentences.
        """
        sentences = self.sentence_splitter.split_into_sentences(doc_text)
        return self.chunk_sentences(sentences)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return batch


def _process_batch_multithreaded(batch_data: List[Dict], chunker: GraphBasedChunker, num_workers: int = 4) -> Dict[str, List[str]]:
    """Process a batch of documents using multithreading for parallel GPU utilization.
    
    Args:
        batch_data: List of dictionaries with 'story_id' and 'sentences'.
        chunker: GraphBasedChunker instance.
        num_workers: Number of worker threads for parallel processing.
        
    Returns:
        Dictionary mapping story_id to chunks.
    """
    results = {}
    
    def process_single(item):
        story_id = item['story_id']
        sentences = item['sentences']
        if sentences:
            chunks = chunker.chunk_sentences(sentences)
        else:
            chunks = []
        return story_id, chunks
    
    # Use ThreadPoolExecutor to process multiple documents in parallel on GPU
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single, item): item for item in batch_data}
        
        for future in as_completed(futures):
            try:
                story_id, chunks = future.result()
                results[story_id] = chunks
            except Exception as exc:
                item = futures[future]
                logger.error(f"Story {item['story_id']} generated an exception: {exc}")
                results[item['story_id']] = []
    
    return results


def _add_chunks_to_stories(
    stories: Dict[str, Any],
    chunker: GraphBasedChunker,
    accelerator: Optional[Accelerator] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    dataloader_workers: int = 2
) -> Dict[str, Any]:
    """Add chunks to each story using DataLoader and multithreading for better GPU utilization.
    
    Args:
        stories: Dictionary of stories with structure {story_id: {document, questions, answers}}.
        chunker: Initialized GraphBasedChunker instance.
        accelerator: Accelerator instance for distributed processing.
        batch_size: Number of documents to process in a batch.
        num_workers: Number of threads for parallel GPU processing within each batch.
        dataloader_workers: Number of workers for DataLoader.
        
    Returns:
        Updated stories dictionary with 'chunks' key added to each story.
    """
    total_stories = len(stories)
    story_items = list(stories.items())
    
    if accelerator:
        # Split stories across processes
        stories_per_process = len(story_items) // accelerator.num_processes
        start_idx = accelerator.process_index * stories_per_process
        
        # Handle remaining stories
        if accelerator.process_index == accelerator.num_processes - 1:
            end_idx = len(story_items)
        else:
            end_idx = start_idx + stories_per_process
        
        local_story_items = story_items[start_idx:end_idx]
        
        if accelerator.is_main_process:
            logger.info(f"Processing {total_stories} stories across {accelerator.num_processes} processes")
            logger.info(f"Process {accelerator.process_index}: handling {len(local_story_items)} stories ({start_idx} to {end_idx})")
            logger.info(f"Using batch_size={batch_size}, num_workers={num_workers} for multithreading")
    else:
        local_story_items = story_items
        logger.info(f"Processing {total_stories} stories (single process)")
        logger.info(f"Using batch_size={batch_size}, num_workers={num_workers} for multithreading")
    
    # Create dataset and dataloader
    dataset = DocumentDataset(local_story_items)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Process batches with multithreading
    local_results = {}
    progress_bar = tqdm(
        dataloader,
        desc=f"Processing batches (Process {accelerator.process_index if accelerator else 0})",
        disable=not (accelerator is None or accelerator.is_main_process),
        total=len(dataloader)
    )
    
    for batch in progress_bar:
        try:
            batch_results = _process_batch_multithreaded(batch, chunker, num_workers=num_workers)
            local_results.update(batch_results)
        except Exception as exc:
            if accelerator is None or accelerator.is_main_process:
                logger.error(f"Batch processing generated an exception: {exc}")
            # Add empty chunks for failed batch items
            for item in batch:
                if item['story_id'] not in local_results:
                    local_results[item['story_id']] = []
    
    # Gather results from all processes
    if accelerator:
        # Some versions of `Accelerator` do not provide gather_object.
        # Fall back to a simple file-based gather: each process writes its
        # local_results to a temp file, then the main process reads and merges them.
        import tempfile as _tempfile
        import pickle as _pickle

        tmp_dir = Path(_tempfile.gettempdir())
        tmp_file = tmp_dir / f"semantic_chunking_results_{accelerator.process_index}.pkl"
        try:
            with open(tmp_file, "wb") as tf:
                _pickle.dump(local_results, tf, protocol=_pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to write temp results for process {accelerator.process_index}: {e}")

        # Ensure all processes have written their files
        accelerator.wait_for_everyone()

        # Main process merges files
        if accelerator.is_main_process:
            merged_results = {}
            for i in range(accelerator.num_processes):
                pfile = tmp_dir / f"semantic_chunking_results_{i}.pkl"
                if pfile.exists():
                    try:
                        with open(pfile, "rb") as rf:
                            data = _pickle.load(rf)
                            if isinstance(data, dict):
                                merged_results.update(data)
                    except Exception as e:
                        logger.error(f"Failed to read temp results from {pfile}: {e}")
                    try:
                        pfile.unlink()
                    except Exception:
                        pass

            # Add chunks back to original stories dict
            for story_id, chunks in merged_results.items():
                stories[story_id]['chunks'] = chunks

            logger.info(f"Completed chunking for all {total_stories} stories")
        else:
            # Non-main processes return None
            return None
    else:
        # Single process: just add chunks directly
        for story_id, chunks in local_results.items():
            stories[story_id]['chunks'] = chunks
        
        logger.info(f"Completed chunking for all {total_stories} stories")
    
    return stories


def process_and_save_split(
    split_name: str,
    dataset_name: str,
    max_samples: Optional[int],
    model_name: str,
    device: str,
    threshold: float,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    dataloader_workers: int,
    accelerator: Optional[Accelerator] = None
) -> Optional[Dict[str, Any]]:
    """Process a single data split and save to pickle file.
    
    Args:
        split_name: Name of the split ('train', 'validation', or 'test').
        dataset_name: Name of the dataset to load.
        max_samples: Maximum number of samples to load (None for all).
        model_name: Sentence transformer model name.
        device: Device to run the model on.
        threshold: Similarity threshold for chunking.
        output_dir: Directory to save the pickle file.
        batch_size: Number of documents to process in a batch.
        num_workers: Number of threads for parallel GPU processing.
        dataloader_workers: Number of workers for DataLoader.
        accelerator: Accelerator instance for distributed processing.
        
    Returns:
        Dictionary with stories containing chunks (None for non-main processes).
    """
    if accelerator is None or accelerator.is_main_process:
        logger.info(f"Loading {split_name} data with chunks")
    
    # Load appropriate split
    if split_name == "train":
        stories = load_train(
            dataset_name=dataset_name,
            max_samples=max_samples,
            trust_remote_code=True,
            group_by_story=True
        )
    elif split_name == "validation":
        stories = load_validation(
            dataset_name=dataset_name,
            max_samples=max_samples,
            trust_remote_code=True,
            group_by_story=True
        )
    elif split_name == "test":
        stories = load_test(
            dataset_name=dataset_name,
            max_samples=max_samples,
            trust_remote_code=True,
            group_by_story=True
        )
    else:
        raise ValueError(f"Unknown split: {split_name}")
    
    # Create chunker and process
    chunker = GraphBasedChunker(
        model_name=model_name,
        device=device,
        threshold=threshold,
        accelerator=accelerator
    )
    
    stories_with_chunks = _add_chunks_to_stories(
        stories, 
        chunker, 
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=num_workers,
        dataloader_workers=dataloader_workers
    )
    
    # Save to pickle file (only on main process)
    if accelerator is None or accelerator.is_main_process:
        if stories_with_chunks is not None:
            output_path = output_dir / f"{split_name}_chunks.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(stories_with_chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved {split_name} chunks to {output_path}")
            logger.info(f"  - Total stories: {len(stories_with_chunks)}")
            
            # Calculate statistics
            total_chunks = sum(len(story['chunks']) for story in stories_with_chunks.values())
            logger.info(f"  - Total chunks: {total_chunks}")
            
            return stories_with_chunks
    
    return None


__all__ = [
    "GraphBasedChunker",
    "SemanticSentenceChunker",
    "DocumentDataset",
    "process_and_save_split"
]


# Note: argparse-based CLI removed. Use the `Config` dataclass above to
# configure behavior programmatically when running this module as a script.


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Use Config dataclass instead of argparse
    config = Config()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Set device based on accelerator
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info("=" * 80)
        logger.info("Semantic Chunking Pipeline with Accelerate")
        logger.info("=" * 80)
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Device: {device}")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Threshold: {config.threshold}")
        logger.info(f"Dataset: {config.dataset_name}")
        logger.info(f"Max samples per split: {config.max_samples if config.max_samples else 'All'}")
        logger.info(f"Splits to process: {config.splits}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Num workers (threads per batch): {config.num_workers}")
        logger.info(f"DataLoader workers: {config.dataloader_workers}")
        logger.info("=" * 80)

    # Create output directory
    output_dir = Path(config.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created: {output_dir.absolute()}")
    
    # Wait for main process to create directory
    accelerator.wait_for_everyone()
    
    # Process each split
    for split in config.splits:
        if accelerator.is_main_process:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {split.upper()} split")
            logger.info(f"{'='*80}")
        
        try:
            start_time = datetime.now()
            
            result = process_and_save_split(
                split_name=split,
                dataset_name=config.dataset_name,
                max_samples=config.max_samples,
                model_name=config.model_name,
                device=str(device),
                threshold=config.threshold,
                output_dir=output_dir,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                dataloader_workers=config.dataloader_workers,
                accelerator=accelerator
            )
            
            if accelerator.is_main_process:
                end_time = datetime.now()
                elapsed = (end_time - start_time).total_seconds()
                logger.info(f"Completed {split} split in {elapsed:.2f} seconds")
        
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"Error processing {split} split: {e}", exc_info=True)
        
        # Synchronize processes between splits
        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info(f"\n{'='*80}")
        logger.info("All splits processed successfully!")
        logger.info(f"Chunked data saved to: {output_dir.absolute()}")
        logger.info("=" * 80)
        
        # List saved files
        saved_files = list(output_dir.glob("*.pkl"))
        if saved_files:
            logger.info(f"\nSaved files:")
            for f in sorted(saved_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"  - {f.name} ({size_mb:.2f} MB)")
