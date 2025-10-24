import logging
import re
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from load_dataset import load_train, load_validation, load_test
from tqdm import tqdm


logger = logging.getLogger(__name__)


class SemanticSentenceChunker:
    """Simple sentence splitter for preprocessing."""
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class GraphBasedChunker:
    """Graph-based semantic chunker using similarity threshold."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        threshold: float = 0.5
    ):
        """Initialize the chunker with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model.
            device: Device to run the model on ("cpu" or "cuda").
            threshold: Similarity threshold for chunking (0.0 to 1.0).
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.threshold = threshold
        self.sentence_splitter = SemanticSentenceChunker()
        logger.info(f"Initialized GraphBasedChunker with model={model_name}, device={device}, threshold={threshold}")
    
    def chunk_document(self, doc_text: str) -> List[str]:
        """Chunk a document using graph-based semantic similarity.
        
        Args:
            doc_text: The full document text to chunk.
            
        Returns:
            List of chunk strings, where each chunk is concatenated sentences.
        """
        # Step 1: Split into sentences
        sentences = self.sentence_splitter.split_into_sentences(doc_text)
        n = len(sentences)
        
        if n == 0:
            return []
        
        if n == 1:
            return [sentences[0]]
        
        # Step 2: Compute sentence embeddings
        logger.info(f"Encoding {n} sentences for document")
        embeddings = self.model.encode(
            sentences,
            convert_to_tensor=True,
            device=self.model.device,
            show_progress_bar=False  # We'll use our own progress tracking
        )
        
        # Step 3: Compute pairwise cosine similarity
        logger.info("Computing similarity matrix")
        sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        
        # Step 4: Graph-based chunking using BFS
        logger.info("Performing graph-based chunking")
        visited = [False] * n
        chunks = []
        
        for i in tqdm(range(n), desc="Processing sentences", unit="sentence", leave=False):
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
        
        logger.info(f"Created {len(chunks)} chunks from {n} sentences")
        return chunks


def _add_chunks_to_stories(
    stories: Dict[str, Any],
    chunker: GraphBasedChunker
) -> Dict[str, Any]:
    """Add chunks to each story in the dataset.
    
    Args:
        stories: Dictionary of stories with structure {story_id: {document, questions, answers}}.
        chunker: Initialized GraphBasedChunker instance.
        
    Returns:
        Updated stories dictionary with 'chunks' key added to each story.
    """
    total_stories = len(stories)
    logger.info(f"Processing {total_stories} stories for chunking")
    
    for idx, (story_id, story_data) in enumerate(tqdm(stories.items(), desc="Processing stories", unit="story"), 1):
        doc_text = story_data['document'].get('text', '')
        
        if doc_text:
            chunks = chunker.chunk_document(doc_text)
            story_data['chunks'] = chunks
        else:
            story_data['chunks'] = []
            logger.warning(f"Story {story_id} has no text content")
    
    logger.info(f"Completed chunking for all {total_stories} stories")
    return stories


def get_train_chunks(
    dataset_name: str = "deepmind/narrativeqa",
    max_samples: Optional[int] = None,
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cuda",
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Load training data and add semantic chunks to each story.
    
    Args:
        dataset_name: Name of the dataset to load.
        max_samples: Maximum number of samples to load (None for all).
        model_name: Sentence transformer model name.
        device: Device to run the model on.
        threshold: Similarity threshold for chunking.
        
    Returns:
        Dictionary with stories containing chunks.
    """
    logger.info("Loading training data with chunks")
    stories = load_train(
        dataset_name=dataset_name,
        max_samples=max_samples,
        trust_remote_code=True,
        group_by_story=True
    )
    
    chunker = GraphBasedChunker(model_name=model_name, device=device, threshold=threshold)
    return _add_chunks_to_stories(stories, chunker)


def get_validation_chunks(
    dataset_name: str = "deepmind/narrativeqa",
    max_samples: Optional[int] = None,
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cuda",
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Load validation data and add semantic chunks to each story.
    
    Args:
        dataset_name: Name of the dataset to load.
        max_samples: Maximum number of samples to load (None for all).
        model_name: Sentence transformer model name.
        device: Device to run the model on.
        threshold: Similarity threshold for chunking.
        
    Returns:
        Dictionary with stories containing chunks.
    """
    logger.info("Loading validation data with chunks")
    stories = load_validation(
        dataset_name=dataset_name,
        max_samples=max_samples,
        trust_remote_code=True,
        group_by_story=True
    )
    
    chunker = GraphBasedChunker(model_name=model_name, device=device, threshold=threshold)
    return _add_chunks_to_stories(stories, chunker)


def get_test_chunks(
    dataset_name: str = "deepmind/narrativeqa",
    max_samples: Optional[int] = None,
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cuda",
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Load test data and add semantic chunks to each story.
    
    Args:
        dataset_name: Name of the dataset to load.
        max_samples: Maximum number of samples to load (None for all).
        model_name: Sentence transformer model name.
        device: Device to run the model on.
        threshold: Similarity threshold for chunking.
        
    Returns:
        Dictionary with stories containing chunks.
    """
    logger.info("Loading test data with chunks")
    stories = load_test(
        dataset_name=dataset_name,
        max_samples=max_samples,
        trust_remote_code=True,
        group_by_story=True
    )
    
    chunker = GraphBasedChunker(model_name=model_name, device=device, threshold=threshold)
    return _add_chunks_to_stories(stories, chunker)


__all__ = [
    "GraphBasedChunker",
    "SemanticSentenceChunker",
    "get_train_chunks",
    "get_validation_chunks",
    "get_test_chunks"
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage: Load a small sample with chunks
    print("Loading validation data with chunks (max 2 samples)...")
    validation_stories = get_validation_chunks(max_samples=2, device="cuda")
    
    # Display results
    for story_id, story_data in validation_stories.items():
        print(f"\nStory ID: {story_id}")
        print(f"Number of chunks: {len(story_data['chunks'])}")
        print(f"Number of questions: {len(story_data['questions'])}")
        if story_data['chunks']:
            print(f"Third chunk preview: {story_data['chunks'][2][:100]}...")
            print(f"Associated questions:")
            for question in story_data['questions']:
                print(f" - {question}")
            print("Associated answers:")
            for answer in story_data['answers']:
                print(f" - {answer}")
        
        print("-" * 80)
