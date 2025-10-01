
import torch
import numpy as np
from semantic_chunking import SemanticSentenceChunker, load_narrativeqa_document
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple

class RollingWindowChunker:
    """
    Implements a semantic chunking strategy based on a rolling window of context.

    This method enhances the simple adjacent-sentence comparison by providing a more
    stable and context-aware boundary detection. Instead of comparing a candidate
    sentence (i) only with the immediately preceding sentence (i-1), it compares
    the candidate sentence's embedding with the *average embedding* of a "window"
    of the last 'k' sentences in the current chunk.

    Core Logic:
    1.  The document is split into sentences and embeddings are computed for all.
    2.  A chunk is grown by adding one sentence at a time.
    3.  For each new candidate sentence, a context window of the last 'k' sentences
        of the current chunk is selected.
    4.  The embeddings of the sentences in this window are averaged to create a
        single, stable "context vector" representing the chunk's recent topic.
    5.  The cosine similarity between the candidate sentence and this context
        vector is calculated.
    6.  If the similarity is above the defined threshold, the sentence is added to
        the chunk. Otherwise, the current chunk is finalized, and a new one begins.

    This approach is more robust against single sentences that might be slightly
    off-topic but don't represent a true shift in the narrative, resulting in
    more coherent and consistently themed chunks. It remains computationally
    efficient with a time complexity of O(n).
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cuda:0'):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.sentence_splitter = SemanticSentenceChunker()
        print(f"Loaded SBERT model: {model_name} on device: {device}")

    def chunk_document(self, doc_text: str, threshold: float, window_size: int = 3) -> Tuple[List[Dict], Dict]:
        """
        Processes a document to create semantic chunks using the rolling window method.

        Args:
            doc_text: The document text to chunk.
            threshold: The similarity score threshold for creating a boundary.
            window_size: The number of recent sentences to average for context.

        Returns:
            A tuple containing (list_of_chunks, statistics_dictionary).
        """
        sentences = self.sentence_splitter.split_into_sentences(text=doc_text)
        if len(sentences) < 2:
            return [], {}

        embeddings = self.model.encode(sentences, convert_to_tensor=True, device=self.device)
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embeddings = [embeddings[0]]

        for i in range(1, len(sentences)):
            candidate_embedding = embeddings[i]

            # Determine the context window: the last `window_size` embeddings
            start_index = max(0, len(current_chunk_embeddings) - window_size)
            window_embeddings = torch.stack(current_chunk_embeddings[start_index:])
            
            # Calculate the average context vector
            context_vector = torch.mean(window_embeddings, dim=0, keepdim=True)

            # Calculate similarity between the candidate and the context
            similarity = util.cos_sim(context_vector, candidate_embedding.unsqueeze(0)).item()

            if similarity >= threshold:
                # Add sentence to the current chunk
                current_chunk_sentences.append(sentences[i])
                current_chunk_embeddings.append(candidate_embedding)
            else:
                # Finalize the current chunk and start a new one
                chunks.append({
                    'chunk_id': len(chunks),
                    'sentences': current_chunk_sentences.copy(),
                    'sentence_count': len(current_chunk_sentences),
                    'text': ' '.join(current_chunk_sentences),
                    'boundary_similarity': similarity
                })
                current_chunk_sentences = [sentences[i]]
                current_chunk_embeddings = [candidate_embedding]

        # Add the final remaining chunk
        chunks.append({
            'chunk_id': len(chunks),
            'sentences': current_chunk_sentences.copy(),
            'sentence_count': len(current_chunk_sentences),
            'text': ' '.join(current_chunk_sentences)
        })
        
        # We pass dummy similarity scores for statistics calculation
        dummy_scores = [c.get('boundary_similarity', threshold) for c in chunks]
        statistics = self.sentence_splitter.compute_statistics(chunks, dummy_scores, sentences)
        statistics['chunking_strategy'] = 'rolling_window'
        statistics['window_size'] = window_size
        return chunks, statistics

if __name__ == "__main__":
    # --- Configuration ---
    SIMILARITY_THRESHOLD = 0.6  # Optimal threshold may differ from other methods
    WINDOW_SIZE = 3             # Number of sentences to average for context
    DOCUMENT_INDEX = 0

    print("🚀 Initializing Rolling Window Chunker...")
    chunker = RollingWindowChunker(device='cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"📚 Loading document {DOCUMENT_INDEX} from NarrativeQA...")
    document_text, document_info = load_narrativeqa_document(document_index=DOCUMENT_INDEX)

    print("\nProcessing document...")
    chunks, stats = chunker.chunk_document(document_text, SIMILARITY_THRESHOLD, WINDOW_SIZE)

    # --- Print Results ---
    print("\n" + "="*80)
    print("ROLLING WINDOW CHUNKING STATISTICS")
    print("="*80)
    chunker.sentence_splitter.print_statistics(stats)
    print(f"Strategy Specifics: Rolling Window (size={stats['window_size']})")
    
    print("\n" + "="*80)
    print("EXAMPLE CHUNKS")
    print("="*80)
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ({chunk['sentence_count']} sentences) ---")
        print(f"Text: {chunk['text'][:250].strip()}{'...' if len(chunk['text']) > 250 else ''}")
        if 'boundary_similarity' in chunk:
            print(f"Boundary Similarity (how different it was from the next chunk): {chunk['boundary_similarity']:.3f}")

