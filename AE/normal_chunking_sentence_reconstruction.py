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

class Vocabulary:
    """Manages the vocabulary for converting sentences to and from numerical vectors."""
    def __init__(self):
        self.word2index = {"<PAD>": 0, "<UNK>": 1}
        self.index2word = {0: "<PAD>", 1: "<UNK>"}
        self.n_words = 2

    def build_vocab(self, sentences: List[str]):
        """Build vocabulary from a list of sentences."""
        words = set()
        for sentence in sentences:
            words.update(sentence.lower().split())
        
        for word in sorted(list(words)):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1
        print(f"Built vocabulary with {self.n_words} unique words.")

    def sentence_to_vector(self, sentence: str) -> torch.Tensor:
        """Convert a sentence to a bag-of-words vector."""
        vector = torch.zeros(self.n_words)
        for word in sentence.lower().split():
            if word in self.word2index:
                vector[self.word2index[word]] += 1
        return vector

    def vector_to_sentence(self, vector: torch.Tensor) -> str:
        """Convert a bag-of-words vector back to a sentence."""
        # Get the indices of the words present in the vector
        indices = torch.where(vector > 0.5)[0] # Use a threshold to identify words
        if len(indices) == 0:
            # If no word is above threshold, take the one with the highest score
            indices = [torch.argmax(vector)]
            
        words = [self.index2word.get(idx.item(), "<UNK>") for idx in indices]
        return ' '.join(words)

class SentenceAutoEncoder(nn.Module):
    """
    Autoencoder for sentence vectors (bag-of-words).
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
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.vocab = Vocabulary()
        print(f"Using device: {device}")
    
    def load_document(self, dataset_split: str = 'validation', document_index: int = 0):
        """Load a single document from NarrativeQA dataset."""
        dataset = load_dataset("deepmind/narrativeqa", split=dataset_split)
        sample = dataset[document_index]
        doc_text = sample["document"]["text"]
        
        # Build vocabulary from the document
        chunker = FixedChunker()
        sentences = chunker.split_into_sentences(doc_text)
        self.vocab.build_vocab(sentences)
        
        return doc_text, {
            "dataset": "narrativeqa",
            "split": dataset_split,
            "index": document_index,
            "document_id": sample["document"].get("id", f"{dataset_split}_{document_index}"),
        }
    
    def train_autoencoder(self, train_vectors: torch.Tensor, val_vectors: torch.Tensor, bottleneck_dim: int, epochs: int = 100, lr: float = 0.001, patience: int = 10) -> Tuple[SentenceAutoEncoder, List[float], List[float]]:
        """Train an autoencoder on the given sentence vectors with early stopping."""
        model = SentenceAutoEncoder(
            input_dim=self.vocab.n_words,
            bottleneck_dim=bottleneck_dim
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        train_vectors = train_vectors.clone().to(self.device)
        val_vectors = val_vectors.clone().to(self.device)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            reconstructed, _ = model(train_vectors)
            train_loss = criterion(reconstructed, train_vectors)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_reconstructed, _ = model(val_vectors)
                val_loss = criterion(val_reconstructed, val_vectors)
                val_losses.append(val_loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

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
    
    def evaluate_reconstruction(self, original_sentences: List[str], reconstructed_vectors: torch.Tensor) -> Dict:
        """Evaluate the quality of reconstruction using BLEU score."""
        
        reconstructed_sentences = [self.vocab.vector_to_sentence(vec) for vec in reconstructed_vectors]
        
        bleu_scores = []
        for original, reconstructed in zip(original_sentences, reconstructed_sentences):
            original_tokens = original.lower().split()
            reconstructed_tokens = reconstructed.lower().split()
            
            # Calculate BLEU score
            score = sentence_bleu([original_tokens], reconstructed_tokens, weights=(0.5, 0.5)) # BLEU-2
            bleu_scores.append(score)
            
        return {
            'avg_bleu_score': float(np.mean(bleu_scores)),
            'min_bleu_score': float(np.min(bleu_scores)),
            'max_bleu_score': float(np.max(bleu_scores)),
            'std_bleu_score': float(np.std(bleu_scores)),
            'sample_reconstructions': list(zip(original_sentences[:5], reconstructed_sentences[:5])) # Show 5 samples
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
        
        # Step 2: Convert each sentence in each chunk to a vector
        chunk_sentence_vectors = []
        original_chunk_sentences = []
        for chunk in chunks:
            # For simplicity, we'll train the autoencoder on individual sentences within chunks
            for sentence in chunk['sentences']:
                chunk_sentence_vectors.append(self.vocab.sentence_to_vector(sentence))
                original_chunk_sentences.append(sentence)

        sentence_vectors = torch.stack(chunk_sentence_vectors)
        
        # Step 3: Split data into train, validation, and test sets (80/10/10)
        num_sentences = sentence_vectors.size(0)
        indices = torch.randperm(num_sentences)
        train_end = int(num_sentences * 0.8)
        val_end = int(num_sentences * 0.9)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_vectors = sentence_vectors[train_indices]
        val_vectors = sentence_vectors[val_indices]
        test_vectors = sentence_vectors[test_indices]
        
        test_original_sentences = [original_chunk_sentences[i] for i in test_indices]
        
        print(f"Split into {len(train_vectors)} train, {len(val_vectors)} val, {len(test_vectors)} test sentences.")

        # Step 4: Train autoencoder
        start_time = time.time()
        autoencoder, train_losses, val_losses = self.train_autoencoder(train_vectors, val_vectors, bottleneck_dim)
        training_time = time.time() - start_time
        
        # Step 5: Evaluate reconstruction on the test set
        autoencoder.eval()
        with torch.no_grad():
            reconstructed_vectors, _ = autoencoder(test_vectors.to(self.device))
        
        # Step 6: Calculate metrics on the test set
        metrics = self.evaluate_reconstruction(test_original_sentences, reconstructed_vectors.cpu())
        
        # Step 7: Compression ratio
        compression_ratio = self.vocab.n_words / bottleneck_dim
        
        result = {
            'chunk_size': chunk_size,
            'bottleneck_dim': bottleneck_dim,
            'compression_ratio': compression_ratio,
            'num_chunks': len(chunks),
            'num_sentences': len(sentences),
            'vocab_size': self.vocab.n_words,
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
                        print(f"Chunk Size {chunk_size}, Bottleneck {bottleneck_dim}: "
                              f"Avg BLEU={result['avg_bleu_score']:.4f}, "
                              f"Compression Ratio={result['compression_ratio']:.1f}x")
                        # Also print a sample reconstruction
                        if result['sample_reconstructions']:
                            original, reconstructed = result['sample_reconstructions'][0]
                            print(f"  Sample -> Original: '{original}'")
                            print(f"  Sample -> Reconstructed: '{reconstructed}'")
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
        if valid_results:
            print(f"Vocabulary size: {valid_results[0]['vocab_size']}")
        
        print("\nPer-configuration results:")
        print("-" * 100)
        print(f"{'Chunk Size':<12} {'Bottleneck':<12} {'Compression':<12} {'Avg BLEU':<12} {'Training Time':<15}")
        print("-" * 100)
        
        for result in sorted(valid_results, key=lambda x: (x['chunk_size'], x['bottleneck_dim'])):
            print(f"{result['chunk_size']:<12} "
                  f"{result['bottleneck_dim']:<12} "
                  f"{result['compression_ratio']:<12.1f} "
                  f"{result['avg_bleu_score']:<12.4f} "
                  f"{result['training_time']:<15.2f}")
        
        # Find best configurations
        best_bleu = max(valid_results, key=lambda x: x['avg_bleu_score'])
        
        print(f"\nBest configurations:")
        print(f"Highest BLEU Score: Chunk Size {best_bleu['chunk_size']}, Bottleneck {best_bleu['bottleneck_dim']} (BLEU: {best_bleu['avg_bleu_score']:.4f})")

def run_autoencoder_analysis():
    """Main function to run the autoencoder analysis."""
    print("Starting Autoencoder Sentence Reconstruction Analysis")
    print("="*50)
    
    # Configuration
    CHUNK_SIZES = [3, 5, 7, 10] # Different chunk sizes
    BOTTLENECK_DIMS = [16, 32, 64, 128]  # Different compression levels
    DOCUMENT_INDEX = 0
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Initialize analyzer
    analyzer = AutoEncoderChunkAnalyzer(device=DEVICE)
    
    # Load document and build vocabulary
    doc_text, doc_info = analyzer.load_document(document_index=DOCUMENT_INDEX)
    print(f"Loaded document: {doc_info['document_id']}")
    print(f"Document length: {len(doc_text)} characters")
    
    # Run parameter sweep
    results = analyzer.parameter_sweep(doc_text, BOTTLENECK_DIMS, CHUNK_SIZES)
    
    # Print summary
    analyzer.print_results_summary(results)
    
    # Save to CSV
    # We need to handle the list of tuples in 'sample_reconstructions' for CSV conversion
    results_for_df = []
    for res in results:
        res_copy = res.copy()
        if 'sample_reconstructions' in res_copy:
            res_copy['sample_reconstructions'] = str(res_copy['sample_reconstructions'])
        results_for_df.append(res_copy)
        
    df = pd.DataFrame(results_for_df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"autoencoder_sentence_reconstruction_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Results saved to: {csv_filename}")
    
    # Save detailed results to JSON
    detailed_results = {
        'document_info': doc_info,
        'configuration': {
            'chunk_sizes': CHUNK_SIZES,
            'bottleneck_dimensions': BOTTLENECK_DIMS,
            'device': DEVICE
        },
        'results': results
    }
    
    json_filename = f"autoencoder_sentence_reconstruction_detailed_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"✅ Detailed results saved to: {json_filename}")
    
    return results, df

if __name__ == "__main__":
    results, df = run_autoencoder_analysis()
