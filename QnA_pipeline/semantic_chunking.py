import logging
import re
import pickle
import os
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from load_dataset import load_train, load_validation, load_test
from tqdm import tqdm
from accelerate import Accelerator
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
import gc
import html
# Prevent huggingface/tokenizers parallelism warnings when using forks.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)


@dataclass
class Config:
    dataset_name: str = "deepmind/narrativeqa"
    model_name: str = "BAAI/bge-large-en-v1.5"
    threshold: float = 0.7
    max_samples: Optional[int] = None
    output_dir: str = "/ssd_scratch/yr_chunked_data"
    splits: List[str] = field(default_factory=lambda: ["train", "validation", "test"])
    batch_size: int =  4 # Document batch size (how many docs to process at once)
    embedding_batch_size: int = 256  # Batch size for encoding sentences
    max_sentences_per_doc: int = 10000  # Split very large documents
    dataloader_workers: int = 4  # Num workers for data loading
    use_distributed: bool = True  # Use multi-GPU inference
    memory_threshold_gb: float = 12.0  # GPU memory threshold for safety


class SemanticSentenceChunker:
    """Simple sentence splitter for preprocessing."""
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Remove HTML comments
        cleaned = re.sub(r'<!--.*?-->', ' ', text, flags=re.S)
        # Remove HTML tags
        cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
        # Unescape HTML entities (&amp;, &nbsp;, ...)
        cleaned = html.unescape(cleaned)
        # Replace markdown links [text](url) with text
        cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
        # Remove bare URLs
        cleaned = re.sub(r'http\S+|www\.\S+', ' ', cleaned)
        # Remove backticks and inline code markers
        cleaned = re.sub(r'`+', '', cleaned)
        # Remove repeated underscores/asterisks used for markdown emphasis (e.g. __bold__, **bold**)
        cleaned = re.sub(r'[_*]{2,}', '', cleaned)
        # Replace remaining single underscores/asterisks with a space
        cleaned = re.sub(r'[_*]', ' ', cleaned)
        # Normalize newlines and carriage returns to spaces
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
        # Collapse multiple whitespace into a single space
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Split into sentences (keep default behavior of splitting after .!?)
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
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


class OptimizedGraphChunker:
    """Memory-optimized graph-based semantic chunker with true multi-GPU support."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        threshold: float = 0.5,
        accelerator: Optional[Accelerator] = None,
        embedding_batch_size: int = 128,
        max_sentences_per_doc: int = 5000,
        memory_threshold_gb: float = 12.0
    ):
        """Initialize the chunker with proper multi-GPU setup.
        
        Args:
            model_name: Name of the sentence transformer model.
            threshold: Similarity threshold for chunking (0.0 to 1.0).
            accelerator: Accelerator instance for distributed processing.
            embedding_batch_size: Batch size for encoding sentences.
            max_sentences_per_doc: Max sentences to process at once (memory management).
            memory_threshold_gb: GPU memory threshold in GB for safety.
        """
        self.accelerator = accelerator
        self.threshold = threshold
        self.sentence_splitter = SemanticSentenceChunker()
        self.embedding_batch_size = embedding_batch_size
        self.max_sentences_per_doc = max_sentences_per_doc
        self.memory_threshold_gb = memory_threshold_gb
        
        # Initialize model - DO NOT wrap with accelerator.prepare()
        # SentenceTransformer handles its own batching and doesn't need DDP
        self.model = SentenceTransformer(model_name)
        
        # Move model to the correct device for this process
        if self.accelerator is not None:
            device = self.accelerator.device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self.model.to(device)
        self.device = device
        
        # Log initialization info
        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"Initialized OptimizedGraphChunker with:")
            logger.info(f"  - Model: {model_name}")
            logger.info(f"  - Device: {device}")
            logger.info(f"  - Threshold: {threshold}")
            logger.info(f"  - Embedding batch size: {embedding_batch_size}")
            logger.info(f"  - Max sentences per doc: {max_sentences_per_doc}")
            if self.accelerator:
                logger.info(f"  - Processes: {self.accelerator.num_processes}")
                logger.info(f"  - Process index: {self.accelerator.process_index}")
    
    def _compute_similarity_matrix_efficient(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity matrix with memory optimization."""
        n = embeddings.shape[0]
        
        # Check if we need to compute in chunks due to memory constraints
        embedding_dim = embeddings.shape[1]
        matrix_size_mb = (n * n * 4) / (1024 * 1024)  # float32 = 4 bytes
        
        if matrix_size_mb > self.memory_threshold_gb * 1024:  # Convert GB to MB
            logger.warning(f"Similarity matrix would be {matrix_size_mb:.1f}MB, computing in chunks")
            return self._compute_similarity_in_chunks(embeddings)
        
        # Standard computation for smaller matrices
        with torch.no_grad():
            sim_matrix = util.cos_sim(embeddings, embeddings)
        return sim_matrix
    
    def _compute_similarity_in_chunks(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity matrix in chunks to save memory."""
        n = embeddings.shape[0]
        chunk_size = min(int(np.sqrt((self.memory_threshold_gb * 1024 * 1024) / 4)), n)
        
        sim_matrix = torch.zeros(n, n, device=embeddings.device, dtype=torch.float32)
        
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            chunk_i = embeddings[i:end_i]
            
            # Compute similarities for this chunk with all embeddings
            with torch.no_grad():
                similarities = util.cos_sim(chunk_i, embeddings)
                sim_matrix[i:end_i, :] = similarities
        
        return sim_matrix
    
    def _optimized_chunking(self, sim_matrix: torch.Tensor, sentences: List[str]) -> List[str]:
        """Optimized chunking with better memory management."""
        n = sim_matrix.shape[0]
        device = sim_matrix.device
        
        # Use PyTorch operations where possible for speed
        visited = torch.zeros(n, dtype=torch.bool, device=device)
        chunks = []
        
        for i in range(n):
            if not visited[i]:
                # Start new chunk with current sentence
                chunk_indices = [i]
                visited[i] = True
                
                # Use a queue for BFS (converted to list for simplicity)
                queue = [i]
                
                while queue:
                    current = queue.pop(0)
                    
                    # Find all connected sentences efficiently
                    if current < n:
                        # Check similarity with remaining sentences
                        remaining_indices = torch.where(~visited)[0]
                        if len(remaining_indices) > 0:
                            similarities = sim_matrix[current, remaining_indices]
                            connected_indices = remaining_indices[similarities >= self.threshold]
                            
                            if len(connected_indices) > 0:
                                chunk_indices.extend(connected_indices.tolist())
                                visited[connected_indices] = True
                                queue.extend(connected_indices.tolist())
                
                # Sort indices to maintain order
                chunk_indices.sort()
                chunk_text = " ".join([sentences[idx] for idx in chunk_indices])
                chunks.append(chunk_text)
        
        return chunks
    
    def _chunk_sentences_batch(self, sentences: List[str]) -> List[str]:
        """Chunk a batch of sentences with memory management and multi-GPU support."""
        n = len(sentences)
        
        if n == 0:
            return []
        
        if n == 1:
            return [sentences[0]]
        
        try:
            # Compute sentence embeddings - model.encode() is now directly accessible
            with torch.no_grad():
                embeddings = self.model.encode(
                    sentences,
                    convert_to_tensor=True,
                    batch_size=self.embedding_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    device=self.device  # Explicitly set device
                )
            
            # Ensure embeddings are on the correct device
            if embeddings.device != self.device:
                embeddings = embeddings.to(self.device)
            
            # Compute similarity matrix efficiently
            sim_matrix = self._compute_similarity_matrix_efficient(embeddings)
            
            # Perform optimized chunking
            chunks = self._optimized_chunking(sim_matrix, sentences)
            
            # Cleanup
            del embeddings, sim_matrix
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in _chunk_sentences_batch with {n} sentences: {e}")
            # Fallback: return all sentences as separate chunks
            return sentences
    
    def chunk_sentences(self, sentences: List[str]) -> List[str]:
        """Chunk pre-split sentences using graph-based semantic similarity."""
        n = len(sentences)
        
        if n == 0:
            return []
        
        if n == 1:
            return [sentences[0]]
        
        # Split very large documents into manageable batches
        if n > self.max_sentences_per_doc:
            logger.warning(f"Document has {n} sentences, splitting into batches of {self.max_sentences_per_doc}")
            all_chunks = []
            
            for i in range(0, n, self.max_sentences_per_doc):
                batch_sentences = sentences[i:i + self.max_sentences_per_doc]
                batch_chunks = self._chunk_sentences_batch(batch_sentences)
                all_chunks.extend(batch_chunks)
                
                # Cleanup between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            return all_chunks
        else:
            return self._chunk_sentences_batch(sentences)
    
    def chunk_document(self, doc_text: str) -> List[str]:
        """Chunk a document using graph-based semantic similarity."""
        sentences = self.sentence_splitter.split_into_sentences(doc_text)
        return self.chunk_sentences(sentences)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return batch


def _process_batch_optimized(batch: List[Dict], chunker: OptimizedGraphChunker) -> Dict[str, List[str]]:
    """Process a batch of documents with optimized memory management."""
    results = {}
    
    for item in batch:
        story_id = item['story_id']
        sentences = item['sentences']
        
        if not sentences:
            results[story_id] = []
            continue
        
        try:
            chunks = chunker.chunk_sentences(sentences)
            results[story_id] = chunks
        except Exception as e:
            logger.error(f"Error processing story {story_id}: {e}")
            results[story_id] = []
    
    return results


def _add_chunks_distributed(
    stories: Dict[str, Any],
    chunker: OptimizedGraphChunker,
    accelerator: Accelerator,
    batch_size: int,
    dataloader_workers: int
) -> Optional[Dict[str, Any]]:
    """Add chunks using true distributed processing across all GPUs."""
    
    total_stories = len(stories)
    story_items = list(stories.items())
    
    # Each process handles a subset of stories
    stories_per_process = len(story_items) // accelerator.num_processes
    start_idx = accelerator.process_index * stories_per_process
    
    if accelerator.process_index == accelerator.num_processes - 1:
        end_idx = len(story_items)
    else:
        end_idx = start_idx + stories_per_process
    
    local_story_items = story_items[start_idx:end_idx]
    
    if accelerator.is_main_process:
        logger.info(f"Distributed processing setup:")
        logger.info(f"  - Total stories: {total_stories}")
        logger.info(f"  - Processes: {accelerator.num_processes}")
        logger.info(f"  - Stories per process: ~{stories_per_process}")
    
    logger.info(f"Process {accelerator.process_index}: Processing {len(local_story_items)} stories")
    
    # Create dataset and dataloader
    dataset = DocumentDataset(local_story_items)
    
    # DataLoader without acceleration (since we're not using DDP on the model)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    # Process batches with distributed inference
    local_results = {}
    progress_bar = tqdm(
        dataloader,
        desc=f"GPU {accelerator.process_index} - Processing",
        disable=not accelerator.is_main_process,
        total=len(dataloader)
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Process batch with distributed inference
            batch_results = _process_batch_optimized(batch, chunker)
            local_results.update(batch_results)
            
            # Progress updates
            if accelerator.is_main_process and batch_idx % 10 == 0:
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(accelerator.device) / 1024**3
                    progress_bar.set_postfix({
                        'processed': len(local_results),
                        'memory': f"{memory_allocated:.1f}GB"
                    })
                else:
                    progress_bar.set_postfix({'processed': len(local_results)})
            
        except Exception as e:
            logger.error(f"Batch processing error on GPU {accelerator.process_index}: {e}")
            # Add empty chunks for failed batch
            for item in batch:
                if item['story_id'] not in local_results:
                    local_results[item['story_id']] = []
        
        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Save this process's results to a temporary file
    tmp_file = Path(tempfile.gettempdir()) / f"chunking_results_{accelerator.process_index}.pkl"
    with open(tmp_file, "wb") as f:
        pickle.dump(local_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Process {accelerator.process_index}: Saved {len(local_results)} results to {tmp_file}")
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    # Main process collects all results
    if accelerator.is_main_process:
        logger.info("Gathering results from all processes...")
        merged_results = {}
        
        # Collect results from all processes
        for process_idx in range(accelerator.num_processes):
            tmp_file = Path(tempfile.gettempdir()) / f"chunking_results_{process_idx}.pkl"
            
            if tmp_file.exists():
                try:
                    with open(tmp_file, "rb") as f:
                        process_results = pickle.load(f)
                        merged_results.update(process_results)
                        logger.info(f"  Loaded {len(process_results)} results from process {process_idx}")
                    tmp_file.unlink()  # Clean up
                except Exception as e:
                    logger.error(f"Error loading results from process {process_idx}: {e}")
        
        # Add chunks back to original stories
        for story_id, chunks in merged_results.items():
            if story_id in stories:
                stories[story_id]['chunks'] = chunks
        
        logger.info(f"Completed chunking for {len(merged_results)}/{total_stories} stories")
        return stories
    else:
        return None


def _add_chunks_single_process(
    stories: Dict[str, Any],
    chunker: OptimizedGraphChunker,
    batch_size: int,
    dataloader_workers: int
) -> Dict[str, Any]:
    """Single-process fallback."""
    story_items = list(stories.items())
    
    dataset = DocumentDataset(story_items)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    local_results = {}
    progress_bar = tqdm(dataloader, desc="Processing (single GPU)")
    
    for batch in progress_bar:
        batch_results = _process_batch_optimized(batch, chunker)
        local_results.update(batch_results)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    for story_id, chunks in local_results.items():
        stories[story_id]['chunks'] = chunks
    
    return stories


def process_and_save_split(
    split_name: str,
    dataset_name: str,
    max_samples: Optional[int],
    model_name: str,
    threshold: float,
    output_dir: Path,
    batch_size: int,
    embedding_batch_size: int,
    max_sentences_per_doc: int,
    dataloader_workers: int,
    use_distributed: bool,
    memory_threshold_gb: float,
    accelerator: Optional[Accelerator] = None
) -> Optional[Dict[str, Any]]:
    """Process a single data split with true multi-GPU support."""
    
    if accelerator is None or accelerator.is_main_process:
        logger.info(f"Loading {split_name} data...")
    
    # Load dataset (only on main process to avoid redundant loading)
    if accelerator is None or accelerator.is_main_process:
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
    else:
        stories = None
    
    # Broadcast stories to all processes if using distributed
    if use_distributed and accelerator is not None:
        # Save to temp file and load on all processes
        if accelerator.is_main_process:
            temp_stories_file = Path(tempfile.gettempdir()) / f"stories_{split_name}.pkl"
            with open(temp_stories_file, 'wb') as f:
                pickle.dump(stories, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        accelerator.wait_for_everyone()
        
        if not accelerator.is_main_process:
            temp_stories_file = Path(tempfile.gettempdir()) / f"stories_{split_name}.pkl"
            with open(temp_stories_file, 'rb') as f:
                stories = pickle.load(f)
        
        accelerator.wait_for_everyone()
        
        # Clean up temp file
        if accelerator.is_main_process:
            temp_stories_file.unlink()
    
    # Initialize chunker with distributed setup
    chunker = OptimizedGraphChunker(
        model_name=model_name,
        threshold=threshold,
        accelerator=accelerator,
        embedding_batch_size=embedding_batch_size,
        max_sentences_per_doc=max_sentences_per_doc,
        memory_threshold_gb=memory_threshold_gb
    )
    
    # Process with distributed or single-process setup
    if use_distributed and accelerator is not None:
        stories_with_chunks = _add_chunks_distributed(
            stories, chunker, accelerator, batch_size, dataloader_workers
        )
    else:
        stories_with_chunks = _add_chunks_single_process(
            stories, chunker, batch_size, dataloader_workers
        )
    
    # Save results
    if accelerator is None or accelerator.is_main_process:
        if stories_with_chunks is not None:
            output_path = output_dir / f"{split_name}_chunks.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(stories_with_chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved {split_name} chunks to {output_path}")
            logger.info(f"  - Total stories: {len(stories_with_chunks)}")
            
            total_chunks = sum(len(story.get('chunks', [])) for story in stories_with_chunks.values())
            logger.info(f"  - Total chunks: {total_chunks}")
            if len(stories_with_chunks) > 0:
                logger.info(f"  - Avg chunks per story: {total_chunks / len(stories_with_chunks):.2f}")
            
            return stories_with_chunks
    
    return None


def main():
    """Main execution function with proper multi-GPU setup."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize configuration
    config = Config()
    
    # Initialize Accelerator (this is key for multi-GPU)
    accelerator = Accelerator()
    
    # Set up device mapping for better GPU utilization
    if torch.cuda.is_available():
        # Log GPU information
        gpu_count = torch.cuda.device_count()
        current_device = accelerator.device
        
        if accelerator.is_main_process:
            logger.info(f"GPU Information:")
            logger.info(f"  - Available GPUs: {gpu_count}")
            logger.info(f"  - Accelerator device: {current_device}")
            if current_device.type == 'cuda':
                logger.info(f"  - Device name: {torch.cuda.get_device_name(current_device)}")
                logger.info(f"  - Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.1f} GB")
    
    if accelerator.is_main_process:
        logger.info("=" * 80)
        logger.info("OPTIMIZED Multi-GPU Semantic Chunking Pipeline")
        logger.info("=" * 80)
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Use distributed: {config.use_distributed}")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Threshold: {config.threshold}")
        logger.info(f"Dataset: {config.dataset_name}")
        logger.info(f"Max samples per split: {config.max_samples if config.max_samples else 'All'}")
        logger.info(f"Splits to process: {config.splits}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Document batch size: {config.batch_size}")
        logger.info(f"Embedding batch size: {config.embedding_batch_size}")
        logger.info(f"Max sentences per doc: {config.max_sentences_per_doc}")
        logger.info(f"DataLoader workers: {config.dataloader_workers}")
        logger.info(f"Memory threshold: {config.memory_threshold_gb} GB")
        logger.info("=" * 80)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created: {output_dir.absolute()}")
    
    accelerator.wait_for_everyone()
    
    # Process each split
    for split_name in config.splits:
        if accelerator.is_main_process:
            logger.info(f"{'='*80}")
            logger.info(f"Processing {split_name.upper()} split")
            logger.info(f"{'='*80}")
        
        try:
            start_time = datetime.now()
            
            result = process_and_save_split(
                split_name=split_name,
                dataset_name=config.dataset_name,
                max_samples=config.max_samples,
                model_name=config.model_name,
                threshold=config.threshold,
                output_dir=output_dir,
                batch_size=config.batch_size,
                embedding_batch_size=config.embedding_batch_size,
                max_sentences_per_doc=config.max_sentences_per_doc,
                dataloader_workers=config.dataloader_workers,
                use_distributed=config.use_distributed,
                memory_threshold_gb=config.memory_threshold_gb,
                accelerator=accelerator if config.use_distributed else None
            )
            
            if accelerator.is_main_process:
                end_time = datetime.now()
                elapsed = (end_time - start_time).total_seconds()
                logger.info(f"Completed {split_name} split in {elapsed:.2f} seconds")
        
        except Exception as e:
            if accelerator is None or accelerator.is_main_process:
                logger.error(f"Error processing {split_name} split: {e}", exc_info=True)
        
        # Synchronize and cleanup
        if accelerator is not None:
            accelerator.wait_for_everyone()
        
        # Cleanup between splits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    if accelerator is None or accelerator.is_main_process:
        logger.info(f"{'='*80}")
        logger.info("All splits processed successfully!")
        logger.info(f"Chunked data saved to: {output_dir.absolute()}")
        logger.info("=" * 80)
        
        # List saved files
        saved_files = list(output_dir.glob("*.pkl"))
        if saved_files:
            logger.info(f"Saved files:")
            total_size = 0
            for f in sorted(saved_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                total_size += size_mb
                logger.info(f"  - {f.name} ({size_mb:.2f} MB)")
            logger.info(f"Total size: {total_size:.2f} MB")


if __name__ == "__main__":
    main()