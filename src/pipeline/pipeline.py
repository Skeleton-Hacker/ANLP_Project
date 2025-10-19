import torch
from typing import List, Dict, Optional, Tuple
import sys
import os
import json
from datasets import load_dataset
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking.semantic_chunking import SemanticSentenceChunker  # Fixed import path
from extractor import SentenceEmbeddingExtractor
from autoencoder import AutoencoderCompressor
from chunk_processor import ChunkProcessor

class CompressionPipeline:
    """
    Complete pipeline for chunking text, extracting embeddings, and compressing
    """
    
    def __init__(
        self,
        autoencoder_type: str = 'standard',
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',  # Fixed: use SBERT
        hidden_dim: int = 512,
        latent_dim: int = 256,
        device: Optional[str] = None,
        **autoencoder_kwargs
    ):
        """
        Initialize the complete pipeline
        
        Args:
            autoencoder_type: Type of autoencoder to use
            model_name: Name of the pretrained model for embeddings
            hidden_dim: Hidden dimension for autoencoder
            latent_dim: Latent dimension for autoencoder
            device: Device to run on
            **autoencoder_kwargs: Additional arguments for autoencoder
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name  # Store for consistency
        
        print("=" * 80)
        print("Initializing Semantic Compression Pipeline")
        print("=" * 80)
        
        # Initialize embedding extractor
        print("\n[1/3] Initializing Embedding Extractor...")
        self.embedding_extractor = SentenceEmbeddingExtractor(
            model_name=model_name,
            device=self.device
        )
        
        embedding_dim = self.embedding_extractor.get_embedding_dim()
        
        # Initialize autoencoder
        print(f"\n[2/3] Initializing {autoencoder_type} Autoencoder...")
        self.autoencoder = AutoencoderCompressor(
            autoencoder_type=autoencoder_type,
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            device=self.device,
            **autoencoder_kwargs
        )
        
        # Initialize chunk processor
        print("\n[3/3] Initializing Chunk Processor...")
        self.chunk_processor = ChunkProcessor(
            embedding_extractor=self.embedding_extractor,
            autoencoder_compressor=self.autoencoder
        )
        
        print("\n" + "=" * 80)
        print("Pipeline Initialization Complete!")
        print("=" * 80 + "\n")
    
    def load_and_chunk_narrativeqa(
        self,
        split: str = 'train',
        max_samples: Optional[int] = None,
        chunk_size: int = 512,
        similarity_threshold: float = 0.5
    ) -> List[List[str]]:
        """
        Load NarrativeQA dataset and create semantic chunks
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to process
            chunk_size: Target chunk size in tokens
            similarity_threshold: Threshold for semantic chunking
            
        Returns:
            List of chunks, where each chunk is a list of sentences
        """
        print(f"\nLoading NarrativeQA dataset (split: {split})...")
        dataset = load_dataset('narrativeqa', split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"Loaded {len(dataset)} documents")
        
        # Initialize semantic chunker with same model
        chunker = SemanticSentenceChunker(
            model_name=self.model_name,  # Fixed: use consistent model
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold
        )
        
        all_chunks = []
        
        for idx, example in enumerate(dataset):
            print(f"\nProcessing document {idx+1}/{len(dataset)}...")
            
            # Get the document text - handle different dataset structures
            try:
                if 'document' in example and 'text' in example['document']:
                    document_text = example['document']['text']
                elif 'document' in example and 'summary' in example['document']:
                    document_text = example['document']['summary']['text']
                else:
                    print(f"  Skipping: Could not find document text")
                    print(f"  Available keys: {example.keys()}")
                    continue
            except Exception as e:
                print(f"  Error accessing document: {e}")
                continue
            
            # Create semantic chunks
            chunks = chunker.chunk_text(document_text)
            
            print(f"  Created {len(chunks)} chunks")
            
            # Split chunks into sentences
            for chunk_idx, chunk in enumerate(chunks):
                # Simple sentence splitting (you can use a more sophisticated method)
                sentences = [s.strip() for s in chunk.split('.') if s.strip()]
                if sentences:
                    all_chunks.append(sentences)
                    print(f"    Chunk {chunk_idx+1}: {len(sentences)} sentences")
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks
    
    def train_on_chunks(
        self,
        chunks: List[List[str]],
        num_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ) -> Dict[str, float]:
        """
        Train the autoencoder on embeddings from all chunks
        
        Args:
            chunks: List of chunks (each chunk is a list of sentences)
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training statistics
        """
        print("\n" + "=" * 80)
        print("Training Autoencoder")
        print("=" * 80)
        
        # Extract embeddings from all sentences
        print("\nExtracting embeddings from all sentences...")
        all_sentences = []
        for chunk in chunks:
            all_sentences.extend(chunk)
        
        print(f"Total sentences: {len(all_sentences)}")
        
        # Extract embeddings in batches
        all_embeddings = self.embedding_extractor.extract_batch_embeddings(
            all_sentences,
            batch_size=32
        )
        
        print(f"Embeddings shape: {all_embeddings.shape}")
        
        # Train autoencoder
        print(f"\nTraining {self.autoencoder.autoencoder_type} autoencoder...")
        stats = self.autoencoder.train_autoencoder(
            all_embeddings,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=True
        )
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80 + "\n")
        
        return stats
    
    def compress_chunks(
        self,
        chunks: List[List[str]],
        k: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Compress all chunks to k embeddings each
        
        Args:
            chunks: List of chunks
            k: Number of compressed embeddings per chunk
            
        Returns:
            List of compression results
        """
        print("\n" + "=" * 80)
        print(f"Compressing Chunks (m >> k={k})")
        print("=" * 80 + "\n")
        
        results = self.chunk_processor.process_multiple_chunks(chunks, k=k)
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("Compression Summary")
        print("=" * 80)
        
        total_sentences = sum(r['num_sentences'] for r in results)
        avg_compression_ratio = np.mean([r['compression_ratio'] for r in results])
        
        print(f"\nTotal chunks processed: {len(results)}")
        print(f"Total sentences: {total_sentences}")
        print(f"Compressed embeddings per chunk: {k}")
        print(f"Average compression ratio: {avg_compression_ratio:.2f}x")
        
        # Calculate reconstruction errors
        print("\nCalculating reconstruction errors...")
        errors = []
        for idx, result in enumerate(results):
            error = self.chunk_processor.calculate_reconstruction_error(
                result['original_embeddings'],
                result['compressed_embeddings'],  # Fixed: added missing argument
                result['indices']
            )
            errors.append(error)
            if idx < 5:  # Print first 5
                print(f"  Chunk {idx+1}: MSE = {error:.6f}")
        
        print(f"\nAverage reconstruction MSE: {np.mean(errors):.6f}")
        print(f"Std reconstruction MSE: {np.std(errors):.6f}")
        
        return results
    
    def run_complete_pipeline(
        self,
        max_documents: int = 10,
        k: int = 8,
        num_epochs: int = 50,
        chunk_size: int = 512,
        save_results: bool = True
    ) -> Dict:
        """
        Run the complete pipeline from data loading to compression
        
        Args:
            max_documents: Maximum number of documents to process
            k: Number of compressed embeddings per chunk
            num_epochs: Training epochs
            chunk_size: Target chunk size
            save_results: Whether to save results
            
        Returns:
            Dictionary with all results
        """
        # Load and chunk data
        chunks = self.load_and_chunk_narrativeqa(
            split='train',
            max_samples=max_documents,
            chunk_size=chunk_size
        )
        
        # Train autoencoder
        train_stats = self.train_on_chunks(
            chunks,
            num_epochs=num_epochs
        )
        
        # Compress chunks
        compression_results = self.compress_chunks(chunks, k=k)
        
        # Package results
        results = {
            'chunks': chunks,
            'train_stats': train_stats,
            'compression_results': compression_results,
            'config': {
                'autoencoder_type': self.autoencoder.autoencoder_type,
                'k': k,
                'num_epochs': num_epochs,
                'max_documents': max_documents
            }
        }
        
        # Save results
        if save_results:
            self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict, output_dir: str = 'outputs'):
        """Save pipeline results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, f"{self.autoencoder.autoencoder_type}_model.pth")
        self.autoencoder.save_model(model_path)
        
        # Save configuration
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(results['config'], f, indent=2)
        
        print(f"\nResults saved to {output_dir}/")