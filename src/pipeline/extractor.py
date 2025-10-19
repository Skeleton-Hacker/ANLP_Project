import torch
from sentence_transformers import SentenceTransformer
from typing import List

class SentenceEmbeddingExtractor:
    """Extract sentence embeddings using Sentence-BERT"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device=None):
        """
        Initialize the embedding extractor
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run the model on (cuda/cpu)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Sentence-BERT model {model_name} on {self.device}...")
        
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        print(f"Model loaded successfully. Embedding dimension: {self.get_embedding_dim()}")
    
    def extract_cls_embedding(self, sentence: str) -> torch.Tensor:
        """
        Extract sentence embedding (equivalent to CLS for sentence-transformers)
        
        Args:
            sentence: Input sentence string
            
        Returns:
            Sentence embedding as tensor
        """
        embedding = self.model.encode(
            sentence, 
            convert_to_tensor=True,
            device=self.device
        )
        return embedding
    
    def extract_batch_embeddings(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Extract embeddings for a batch of sentences
        
        Args:
            sentences: List of sentence strings
            batch_size: Batch size for processing
            
        Returns:
            Tensor of shape [num_sentences, embedding_dim]
        """
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )
        return embeddings
    
    def extract_chunk_embeddings(self, chunk: List[str]) -> torch.Tensor:
        """
        Extract embeddings for all sentences in a chunk
        
        Args:
            chunk: List of sentences in the chunk (m sentences)
            
        Returns:
            Tensor of shape [m, embedding_dim] where m = number of sentences
        """
        return self.extract_batch_embeddings(chunk)
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self.model.get_sentence_embedding_dimension()