import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import nltk
import time
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from nltk.translate.bleu_score import sentence_bleu

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SentenceAutoEncoder(nn.Module):
    """
    Autoencoder for sentence embeddings with configurable bottleneck size.
    """
    def __init__(self, input_dim: int, bottleneck_dim: int, hidden_dim: int = 256):
        super(SentenceAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, bottleneck_dim),
            nn.Tanh()  # Normalize bottleneck representation
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, encoded):
        return self.decoder(encoded)

class FixedChunker:
    """
    Creates fixed-size chunks of sentences from documents.
    """
    def __init__(self, chunk_size: int = 5):
        self.chunk_size = chunk_size
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK sentence tokenizer."""
        # Clean the text first
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Use NLTK to split into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Filter out very short sentences (less than 3 words)
        filtered_sentences = []
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) >= 1:  # Keep sentences with at least 1 word
                filtered_sentences.append(sentence.strip())
        
        return filtered_sentences
    
    def create_fixed_chunks(self, sentences: List[str]) -> List[Dict]:
        """Create fixed-size chunks from sentences."""
        chunks = []
        
        for i in range(0, len(sentences), self.chunk_size):
            chunk_sentences = sentences[i:i + self.chunk_size]
            
            # Only include chunks that have the full chunk_size
            if len(chunk_sentences) == self.chunk_size:
                chunks.append({
                    'chunk_id': len(chunks),
                    'sentences': chunk_sentences,
                    'sentence_count': len(chunk_sentences),
                    'text': ' '.join(chunk_sentences),
                    'start_idx': i,
                    'end_idx': i + len(chunk_sentences) - 1
                })
        
        return chunks

class AutoEncoderChunkAnalyzer:
    """
    Analyzes how well chunks can be represented using autoencoders.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda:0"):
        self.device = device
        self.sentence_model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
        print(f"Loaded sentence model: {model_name} on {device}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def load_document(self, dataset_split: str = 'validation', document_index: int = 0):
        """Load a single document from NarrativeQA dataset."""
        dataset = load_dataset("deepmind/narrativeqa", split=dataset_split)
        sample = dataset[document_index]
        doc_text = sample["document"]["text"]
        return doc_text, {
            "dataset": "narrativeqa",
            "split": dataset_split,
            "index": document_index,
            "document_id": sample["document"].get("id", f"{dataset_split}_{document_index}"),
        }
    
    def train_autoencoder(self, train_embeddings: torch.Tensor, val_embeddings: torch.Tensor, bottleneck_dim: int, epochs: int = 100, lr: float = 0.001, patience: int = 10) -> Tuple[SentenceAutoEncoder, List[float], List[float]]:
        """Train an autoencoder on the given embeddings with early stopping."""
        model = SentenceAutoEncoder(
            input_dim=self.embedding_dim,
            bottleneck_dim=bottleneck_dim
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        train_embeddings = train_embeddings.clone().to(self.device)
        val_embeddings = val_embeddings.clone().to(self.device)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            reconstructed, _ = model(train_embeddings)
            train_loss = criterion(reconstructed, train_embeddings)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_reconstructed, _ = model(val_embeddings)
                val_loss = criterion(val_reconstructed, val_embeddings)
                val_losses.append(val_loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
                
                # Print a sample reconstruction similarity
                with torch.no_grad():
                    sample_original = val_embeddings[0].unsqueeze(0)
                    sample_reconstructed, _ = model(sample_original)
                    similarity = util.cos_sim(sample_original, sample_reconstructed).item()
                    print(f"  Sample reconstruction similarity: {similarity:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Load the best model state
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        return model, train_losses, val_losses
    
    def evaluate_reconstruction(self, original_embeddings: torch.Tensor, reconstructed_embeddings: torch.Tensor) -> Dict:
        """Evaluate the quality of reconstruction."""
        original_np = original_embeddings.cpu().numpy()
        reconstructed_np = reconstructed_embeddings.cpu().numpy()
        
        # Mean Squared Error
        mse = mean_squared_error(original_np.flatten(), reconstructed_np.flatten())
        
        # Cosine similarity for each embedding
        cosine_sims = []
        for i in range(len(original_np)):
            sim = cosine_similarity([original_np[i]], [reconstructed_np[i]])[0][0]
            cosine_sims.append(sim)
        
        # Calculate semantic similarity using sentence transformer
        semantic_sims = util.cos_sim(original_embeddings, reconstructed_embeddings).diag().cpu().numpy()
        
        return {
            'mse': float(mse),
            'avg_cosine_similarity': float(np.mean(cosine_sims)),
            'min_cosine_similarity': float(np.min(cosine_sims)),
            'max_cosine_similarity': float(np.max(cosine_sims)),
            'std_cosine_similarity': float(np.std(cosine_sims)),
            'avg_semantic_similarity': float(np.mean(semantic_sims)),
            'min_semantic_similarity': float(np.min(semantic_sims)),
            'max_semantic_similarity': float(np.max(semantic_sims)),
            'std_semantic_similarity': float(np.std(semantic_sims))
        }
    
    def analyze_single_bottleneck(self, doc_text: str, bottleneck_dim: int, chunk_size: int) -> Dict:
        """Analyze a single bottleneck dimension for a given chunk size."""
        print(f"\nAnalyzing bottleneck: {bottleneck_dim}, chunk size: {chunk_size}")
        
        # Step 1: Create chunks
        chunker = FixedChunker(chunk_size=chunk_size)
        sentences = chunker.split_into_sentences(doc_text)
        chunks = chunker.create_fixed_chunks(sentences)
        
        if len(chunks) < 10: # Ensure enough chunks for splitting
            return {"error": "Not enough complete chunks found to perform train/val/test split."}
        
        print(f"Created {len(chunks)} chunks of {chunk_size} sentences each")
        
        # Step 2: Generate embeddings for each chunk
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = self.sentence_model.encode(chunk_texts, convert_to_tensor=True, device=self.device)
        
        # Step 3: Split data into train, validation, and test sets (80/10/10)
        num_chunks = chunk_embeddings.size(0)
        indices = torch.randperm(num_chunks)
        train_end = int(num_chunks * 0.8)
        val_end = int(num_chunks * 0.9)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_embeddings = chunk_embeddings[train_indices]
        val_embeddings = chunk_embeddings[val_indices]
        test_embeddings = chunk_embeddings[test_indices]
        
        print(f"Split into {len(train_embeddings)} train, {len(val_embeddings)} val, {len(test_embeddings)} test samples.")

        # Step 4: Train autoencoder
        start_time = time.time()
        autoencoder, train_losses, val_losses = self.train_autoencoder(train_embeddings, val_embeddings, bottleneck_dim)
        training_time = time.time() - start_time
        
        # Step 5: Evaluate reconstruction on the test set
        autoencoder.eval()
        with torch.no_grad():
            reconstructed_embeddings, encoded_representations = autoencoder(test_embeddings)
        
        # Step 6: Calculate metrics on the test set
        metrics = self.evaluate_reconstruction(test_embeddings, reconstructed_embeddings)
        
        # Step 7: Compression ratio
        compression_ratio = self.embedding_dim / bottleneck_dim
        
        result = {
            'chunk_size': chunk_size,
            'bottleneck_dim': bottleneck_dim,
            'compression_ratio': compression_ratio,
            'num_chunks': len(chunks),
            'num_sentences': len(sentences),
            'training_time': training_time,
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_val_loss': val_losses[-1] if val_losses else 0,
            **metrics
        }
        
        return result
    
    def parameter_sweep(self, doc_text: str, bottleneck_dims: List[int], chunk_sizes: List[int]) -> List[Dict]:
        """Perform parameter sweep across different bottleneck dimensions and chunk sizes."""
        results = []
        
        for chunk_size in chunk_sizes:
            print("\n" + "="*50)
            print(f"PROCESSING CHUNK SIZE: {chunk_size}")
            print("="*50)
            
            for bottleneck_dim in bottleneck_dims:
                try:
                    result = self.analyze_single_bottleneck(doc_text, bottleneck_dim, chunk_size)
                    result['timestamp'] = datetime.now().isoformat()
                    results.append(result)
                    
                    # Print current result
                    if 'error' not in result:
                        print(f"Chunk Size {chunk_size}, Bottleneck {bottleneck_dim}: MSE={result['mse']:.6f}, "
                              f"Avg Cosine Sim={result['avg_cosine_similarity']:.4f}, "
                              f"Compression Ratio={result['compression_ratio']:.1f}x")
                    else:
                        print(f"Chunk Size {chunk_size}, Bottleneck {bottleneck_dim}: {result['error']}")
                        
                except Exception as e:
                    print(f"Error with chunk size {chunk_size} and bottleneck dimension {bottleneck_dim}: {str(e)}")
                    results.append({
                        'chunk_size': chunk_size,
                        'bottleneck_dim': bottleneck_dim,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
        
        return results
    
    def print_results_summary(self, results: List[Dict]):
        """Print a summary of all results."""
        print("\n" + "="*80)
        print("AUTOENCODER CHUNK ANALYSIS SUMMARY")
        print("="*80)
        
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid results found!")
            return
        
        print(f"Total experiments: {len(results)}")
        print(f"Successful experiments: {len(valid_results)}")
        print(f"Embedding dimension: {self.embedding_dim}")
        
        print("\nPer-configuration results:")
        print("-" * 130)
        print(f"{'Chunk Size':<12} {'Bottleneck':<12} {'Compression':<12} {'MSE':<12} {'Avg Cos Sim':<12} {'Avg Sem Sim':<12} {'Training Time':<15}")
        print("-" * 130)
        
        for result in sorted(valid_results, key=lambda x: (x['chunk_size'], x['bottleneck_dim'])):
            print(f"{result['chunk_size']:<12} "
                  f"{result['bottleneck_dim']:<12} "
                  f"{result['compression_ratio']:<12.1f} "
                  f"{result['mse']:<12.6f} "
                  f"{result['avg_cosine_similarity']:<12.4f} "
                  f"{result['avg_semantic_similarity']:<12.4f} "
                  f"{result['training_time']:<15.2f}")
        
        # Find best configurations
        best_mse = min(valid_results, key=lambda x: x['mse'])
        best_cosine = max(valid_results, key=lambda x: x['avg_cosine_similarity'])
        best_semantic = max(valid_results, key=lambda x: x['avg_semantic_similarity'])
        
        print(f"\nBest configurations:")
        print(f"Lowest MSE: Chunk Size {best_mse['chunk_size']}, Bottleneck {best_mse['bottleneck_dim']} (MSE: {best_mse['mse']:.6f})")
        print(f"Highest Cosine Similarity: Chunk Size {best_cosine['chunk_size']}, Bottleneck {best_cosine['bottleneck_dim']} (Sim: {best_cosine['avg_cosine_similarity']:.4f})")
        print(f"Highest Semantic Similarity: Chunk Size {best_semantic['chunk_size']}, Bottleneck {best_semantic['bottleneck_dim']} (Sim: {best_semantic['avg_semantic_similarity']:.4f})")

def run_autoencoder_analysis():
    """Main function to run the autoencoder analysis."""
    print("Starting Autoencoder Chunk Analysis")
    print("="*50)
    
    # Configuration
    CHUNK_SIZES = [3, 5, 7, 10,50,100] # Different chunk sizes
    BOTTLENECK_DIMS = [16, 32, 64, 128, 256]  # Different compression levels
    DOCUMENT_INDEX = 0
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Initialize analyzer
    analyzer = AutoEncoderChunkAnalyzer(device=DEVICE)
    
    # Load document
    doc_text, doc_info = analyzer.load_document(document_index=DOCUMENT_INDEX)
    print(f"Loaded document: {doc_info['document_id']}")
    print(f"Document length: {len(doc_text)} characters")
    
    # Run parameter sweep
    results = analyzer.parameter_sweep(doc_text, BOTTLENECK_DIMS, CHUNK_SIZES)
    
    # Print summary
    analyzer.print_results_summary(results)
    
    # Save to CSV
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"autoencoder_chunk_analysis_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Results saved to: {csv_filename}")
    
    # Save detailed results to JSON
    detailed_results = {
        'document_info': doc_info,
        'configuration': {
            'chunk_sizes': CHUNK_SIZES,
            'embedding_model': 'all-MiniLM-L6-v2',
            'bottleneck_dimensions': BOTTLENECK_DIMS,
            'device': DEVICE
        },
        'results': results
    }
    
    json_filename = f"autoencoder_chunk_analysis_detailed_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"✅ Detailed results saved to: {json_filename}")
    
    return results, df

if __name__ == "__main__":
    results, df = run_autoencoder_analysis()
