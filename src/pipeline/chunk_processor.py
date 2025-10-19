import torch
from typing import List, Dict, Tuple, Optional
import numpy as np
from extractor import SentenceEmbeddingExtractor
from autoencoder import AutoencoderCompressor

class ChunkProcessor:
    """Process chunks by extracting embeddings and compressing them"""
    
    def __init__(
        self,
        embedding_extractor: SentenceEmbeddingExtractor,
        autoencoder_compressor: AutoencoderCompressor
    ):
        """
        Initialize the chunk processor
        
        Args:
            embedding_extractor: Instance of SentenceEmbeddingExtractor
            autoencoder_compressor: Instance of AutoencoderCompressor
        """
        self.embedding_extractor = embedding_extractor
        self.autoencoder_compressor = autoencoder_compressor
    
    def process_single_chunk(
        self,
        chunk: List[str],
        k: int
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single chunk: extract embeddings and compress
        
        Args:
            chunk: List of sentences in the chunk
            k: Number of compressed embeddings to generate
            
        Returns:
            Dictionary containing:
                - 'original_embeddings': Original m sentence embeddings
                - 'compressed_embeddings': Compressed k embeddings
                - 'indices': Indices of selected sentences
                - 'num_sentences': Number of sentences (m)
                - 'compression_ratio': m/k ratio
        """
        # Extract CLS embeddings for all sentences in the chunk
        m_embeddings = self.embedding_extractor.extract_chunk_embeddings(chunk)
        m = m_embeddings.shape[0]
        
        # Compress m embeddings to k embeddings
        k_embeddings, indices = self.autoencoder_compressor.compress_embeddings(m_embeddings, k=k)
        
        return {
            'original_embeddings': m_embeddings,
            'compressed_embeddings': k_embeddings,
            'indices': indices,
            'num_sentences': m,
            'compression_ratio': m / k if k > 0 else float('inf')
        }
    
    def process_multiple_chunks(
        self,
        chunks: List[List[str]],
        k: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Process multiple chunks
        
        Args:
            chunks: List of chunks, where each chunk is a list of sentences
            k: Number of compressed embeddings per chunk
            
        Returns:
            List of dictionaries, one per chunk
        """
        results = []
        
        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx+1}/{len(chunks)} ({len(chunk)} sentences)...")
            result = self.process_single_chunk(chunk, k=k)
            results.append(result)
        
        return results
    
    def reconstruct_chunk_embeddings(
        self,
        compressed_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct original embeddings from compressed latent embeddings
        
        Args:
            compressed_embeddings: Compressed k embeddings [k, latent_dim]
            
        Returns:
            Reconstructed embeddings [k, input_dim]
        """
        # Decode from latent space back to original embedding space
        reconstructed = self.autoencoder_compressor.decode(compressed_embeddings)
        return reconstructed
    
    def calculate_reconstruction_error(
        self,
        original_embeddings: torch.Tensor,
        compressed_embeddings: torch.Tensor,
        indices: torch.Tensor
    ) -> float:
        """
        Calculate reconstruction error (MSE) between original and reconstructed embeddings
        
        Args:
            original_embeddings: Original m embeddings [m, input_dim]
            compressed_embeddings: Compressed k embeddings [k, latent_dim]
            indices: Indices of selected sentences [k]
            
        Returns:
            Mean squared error
        """
        # Reconstruct the k selected embeddings
        reconstructed = self.reconstruct_chunk_embeddings(compressed_embeddings)
        
        # Get the original k selected embeddings
        selected_original = original_embeddings[indices]
        
        # Calculate MSE
        mse = torch.mean((selected_original - reconstructed) ** 2).item()
        return mse
    
    def get_compression_stats(
        self,
        results: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Get statistics about compression across multiple chunks
        
        Args:
            results: List of chunk processing results
            
        Returns:
            Dictionary with compression statistics
        """
        total_sentences = sum(r['num_sentences'] for r in results)
        total_compressed = sum(r['compressed_embeddings'].shape[0] for r in results)
        avg_compression_ratio = np.mean([r['compression_ratio'] for r in results])
        
        return {
            'total_sentences': total_sentences,
            'total_compressed': total_compressed,
            'overall_compression_ratio': total_sentences / total_compressed if total_compressed > 0 else float('inf'),
            'avg_compression_ratio_per_chunk': avg_compression_ratio,
            'num_chunks': len(results)
        }