import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import nltk
import numpy as np
import json
import re
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SemanticSentenceChunker:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.7, device: str = 'cuda:3'):
        """
        Initialize the semantic sentence chunker.
        
        Args:
            model_name: SBERT model name to use for embeddings
            similarity_threshold: Threshold for grouping sentences into chunks
            device: Device to run the model on (e.g., 'cuda:3')
        """
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.similarity_threshold = similarity_threshold
        print(f"Loaded SBERT model: {model_name} on device: {device}")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK sentence tokenizer.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Clean the text first
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Use NLTK to split into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Filter out very short sentences (less than 3 words)
        filtered_sentences = []
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) >= 3:  # Keep sentences with at least 3 words
                filtered_sentences.append(sentence.strip())
        
        return filtered_sentences
    
    def compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Compute SBERT embeddings for all sentences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Computing embeddings for {len(sentences)} sentences on {self.device}...")
        embeddings = self.model.encode(sentences, convert_to_tensor=False, show_progress_bar=True, device=self.device)
        return np.array(embeddings)
    
    def compute_similarity_scores(self, embeddings: np.ndarray) -> List[float]:
        """
        Compute cosine similarity scores between adjacent sentences.
        
        Args:
            embeddings: Sentence embeddings array
            
        Returns:
            List of similarity scores between adjacent sentences
        """
        similarity_scores = []
        
        for i in range(len(embeddings) - 1):
            # Compute cosine similarity between current and next sentence
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarity_scores.append(float(sim))
        
        return similarity_scores
    
    def create_chunks(self, sentences: List[str], similarity_scores: List[float]) -> List[Dict]:
        """
        Create semantic chunks based on similarity scores.
        
        Args:
            sentences: List of sentences
            similarity_scores: Similarity scores between adjacent sentences
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_chunk = []
        current_chunk_sentences = []
        
        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)
            
            # If this is the last sentence, add it and finalize the chunk
            if i == len(sentences) - 1:
                chunks.append({
                    'chunk_id': len(chunks),
                    'sentences': current_chunk_sentences.copy(),
                    'sentence_count': len(current_chunk_sentences),
                    'text': ' '.join(current_chunk_sentences)
                })
                break
            
            # Check similarity with next sentence
            similarity = similarity_scores[i]
            
            # If similarity is below threshold, finalize current chunk and start new one
            if similarity < self.similarity_threshold:
                chunks.append({
                    'chunk_id': len(chunks),
                    'sentences': current_chunk_sentences.copy(),
                    'sentence_count': len(current_chunk_sentences),
                    'text': ' '.join(current_chunk_sentences),
                    'boundary_similarity': similarity
                })
                current_chunk_sentences = []
        
        return chunks
    
    def chunk_document(self, document_text: str) -> Tuple[List[Dict], Dict]:
        """
        Process a document and create semantic chunks.
        
        Args:
            document_text: The document text to chunk
            
        Returns:
            Tuple of (chunks, statistics)
        """
        print("Starting semantic chunking process...")
        
        # Step 1: Split into sentences
        sentences = self.split_into_sentences(document_text)
        print(f"Split document into {len(sentences)} sentences")
        
        if len(sentences) < 2:
            print("Not enough sentences for chunking")
            return [], {}
        
        # Step 2: Compute embeddings
        embeddings = self.compute_sentence_embeddings(sentences)
        
        # Step 3: Compute similarity scores
        similarity_scores = self.compute_similarity_scores(embeddings)
        
        # Step 4: Create chunks
        chunks = self.create_chunks(sentences, similarity_scores)
        
        # Step 5: Compute statistics
        statistics = self.compute_statistics(chunks, similarity_scores, sentences)
        
        return chunks, statistics
    
    def compute_statistics(self, chunks: List[Dict], similarity_scores: List[float], sentences: List[str]) -> Dict:
        """
        Compute statistics about the chunking results.
        
        Args:
            chunks: List of chunk dictionaries
            similarity_scores: Similarity scores between sentences
            sentences: Original sentences
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {}
        
        chunk_sizes = [chunk['sentence_count'] for chunk in chunks]
        
        statistics = {
            'total_sentences': len(sentences),
            'total_chunks': len(chunks),
            'average_chunk_size': np.mean(chunk_sizes),
            'median_chunk_size': np.median(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'similarity_threshold_used': self.similarity_threshold,
            'average_similarity': np.mean(similarity_scores) if similarity_scores else 0,
            'min_similarity': min(similarity_scores) if similarity_scores else 0,
            'max_similarity': max(similarity_scores) if similarity_scores else 0,
            'similarities_below_threshold': sum(1 for s in similarity_scores if s < self.similarity_threshold),
            'chunk_size_distribution': {
                'size_1': sum(1 for size in chunk_sizes if size == 1),
                'size_2_5': sum(1 for size in chunk_sizes if 2 <= size <= 5),
                'size_6_10': sum(1 for size in chunk_sizes if 6 <= size <= 10),
                'size_11_plus': sum(1 for size in chunk_sizes if size > 10)
            }
        }
        
        return statistics
    
    def save_results(self, chunks: List[Dict], statistics: Dict, document_info: Dict, output_file: str = 'semantic_chunks.json'):
        """
        Save chunks and statistics to JSON file.
        
        Args:
            chunks: List of chunk dictionaries
            statistics: Statistics dictionary
            document_info: Information about the source document
            output_file: Output file path
        """
        results = {
            'document_info': document_info,
            'chunking_config': {
                'model_name': self.model.get_sentence_embedding_dimension(),
                'similarity_threshold': self.similarity_threshold
            },
            'statistics': statistics,
            'chunks': chunks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to {output_file}")
    
    def print_statistics(self, statistics: Dict):
        """Print detailed statistics about the chunking results."""
        print("\n" + "="*80)
        print("SEMANTIC CHUNKING STATISTICS")
        print("="*80)
        
        print(f"Total Sentences: {statistics['total_sentences']}")
        print(f"Total Chunks: {statistics['total_chunks']}")
        print(f"Average Chunk Size: {statistics['average_chunk_size']:.2f} sentences")
        print(f"Median Chunk Size: {statistics['median_chunk_size']:.1f} sentences")
        print(f"Min/Max Chunk Size: {statistics['min_chunk_size']} / {statistics['max_chunk_size']} sentences")
        
        print(f"\nSimilarity Threshold Used: {statistics['similarity_threshold_used']:.3f}")
        print(f"Average Similarity Score: {statistics['average_similarity']:.3f}")
        print(f"Min/Max Similarity: {statistics['min_similarity']:.3f} / {statistics['max_similarity']:.3f}")
        print(f"Boundaries Created (below threshold): {statistics['similarities_below_threshold']}")
        
        print(f"\nChunk Size Distribution:")
        dist = statistics['chunk_size_distribution']
        print(f"  1 sentence: {dist['size_1']} chunks")
        print(f"  2-5 sentences: {dist['size_2_5']} chunks")
        print(f"  6-10 sentences: {dist['size_6_10']} chunks")
        print(f"  11+ sentences: {dist['size_11_plus']} chunks")


def load_narrativeqa_document(dataset_split: str = 'validation', document_index: int = 0) -> Tuple[str, Dict]:
    """
    Load a single document from NarrativeQA dataset.
    
    Args:
        dataset_split: Which split to use ('train', 'validation', 'test')
        document_index: Index of document to load
        
    Returns:
        Tuple of (document_text, document_info)
    """
    print(f"Loading document {document_index} from NarrativeQA {dataset_split} split...")
    
    dataset = load_dataset("deepmind/narrativeqa", split=dataset_split)
    sample = dataset[document_index]
    
    document_text = sample["document"]["text"]
    document_info = {
        'dataset': 'narrativeqa',
        'split': dataset_split,
        'index': document_index,
        'document_id': sample["document"]["id"] if "id" in sample["document"] else f"{dataset_split}_{document_index}",
        'summary_available': bool(sample["document"].get("summary", {}).get("text", "")),
        'questions_available': len(sample.get("answers", [])) > 0,
        'original_length_chars': len(document_text),
        'original_length_words': len(document_text.split())
    }
    
    print(f"Loaded document: {document_info['document_id']}")
    print(f"Length: {document_info['original_length_chars']} chars, {document_info['original_length_words']} words")
    
    return document_text, document_info


def main():
    """Main function to run semantic sentence chunking."""
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"CUDA available with {gpu_count} GPUs")
        if gpu_count > 3:
            print(f"Using GPU:3 - {torch.cuda.get_device_name(3)}")
        else:
            print(f"Warning: GPU:3 not available, only {gpu_count} GPUs detected")
    else:
        print("Warning: CUDA not available, will use CPU")
    
    # Configuration
    SIMILARITY_THRESHOLD = 0.7  # Adjust this threshold as needed
    DOCUMENT_INDEX = 0  # Which document to process
    OUTPUT_FILE = 'semantic_chunks.json'
    
    # Load document
    document_text, document_info = load_narrativeqa_document(
        dataset_split='validation', 
        document_index=DOCUMENT_INDEX
    )
    
    # Initialize chunker
    chunker = SemanticSentenceChunker(
        model_name='all-MiniLM-L6-v2',  # Fast and good SBERT model
        similarity_threshold=SIMILARITY_THRESHOLD,
        device='cuda:3'  # Use GPU 3
    )
    
    # Process document
    chunks, statistics = chunker.chunk_document(document_text)
    
    # Print statistics
    chunker.print_statistics(statistics)
    
    # Print some example chunks
    print("\n" + "="*80)
    print("EXAMPLE CHUNKS")
    print("="*80)
    
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"\n--- Chunk {i+1} ({chunk['sentence_count']} sentences) ---")
        print(f"Text: {chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}")
        if 'boundary_similarity' in chunk:
            print(f"Boundary similarity: {chunk['boundary_similarity']:.3f}")
    
    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")
    
    # Save results
    chunker.save_results(chunks, statistics, document_info, OUTPUT_FILE)
    
    return chunks, statistics


if __name__ == "__main__":
    chunks, stats = main()