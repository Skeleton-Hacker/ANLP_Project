
import nltk
from nltk.tokenize import texttiling
import numpy as np
from semantic_chunking import SemanticSentenceChunker, load_narrativeqa_document
from typing import List, Dict, Tuple

# Download the TextTiling tokenizer models if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextTilingChunker:
    """
    Implements a classic, lexical-based chunking strategy called TextTiling.

    TextTiling works by partitioning a document into topically coherent segments.
    It does *not* use semantic embeddings. Instead, its decisions are based on the
    patterns of vocabulary usage and word frequency.

    Core Logic:
    1.  The text is tokenized into lowercase "pseudo-sentences" of a fixed
        size (the `w` parameter).
    2.  The algorithm creates blocks of a fixed number of these pseudo-sentences
        (the `k` parameter).
    3.  It then moves a gap between adjacent blocks and calculates a "cohesion score"
        at each gap. This score measures how similar the vocabulary is in the
        block before the gap versus the block after the gap.
    4.  Sharp drops in the cohesion score signal a topic shift and are identified
        as boundaries.
    5.  The original text between these identified boundaries forms the final chunks.

    This method serves as a powerful non-semantic baseline. By comparing its
    output to embedding-based methods, we can validate whether dense semantic
    representations truly offer superior performance for this task over classic
    lexical cohesion techniques.
    """
    def __init__(self):
        self.sentence_splitter = SemanticSentenceChunker()
        self.tiling_tokenizer = texttiling.TextTilingTokenizer()
        print("Initialized TextTiling Chunker (Lexical Method)")

    def chunk_document(self, doc_text: str, w: int = 20, k: int = 10) -> Tuple[List[Dict], Dict]:
        """
        Processes a document to create chunks using the TextTiling algorithm.

        Args:
            doc_text: The document text to chunk.
            w: The size of the pseudo-sentence (in tokens).
            k: The size of the block (in pseudo-sentences) used for scoring.

        Returns:
            A tuple containing (list_of_chunks, statistics_dictionary).
        """
        if not doc_text.strip():
            return [], {}
        
        # NLTK's TextTilingTokenizer does all the work
        tiled_text = self.tiling_tokenizer.tokenize(doc_text, w=w, k=k)
        
        original_sentences = self.sentence_splitter.split_into_sentences(doc_text)
        
        # Format the output to be consistent with other chunkers
        chunks = []
        for i, chunk_text in enumerate(tiled_text):
            # We can approximate the sentence count for statistics
            chunk_sentences = self.sentence_splitter.split_into_sentences(chunk_text)
            chunks.append({
                'chunk_id': i,
                'sentences': chunk_sentences,
                'sentence_count': len(chunk_sentences),
                'text': chunk_text.replace("\n\n", " ").strip()
            })
        
        # We pass dummy similarity scores for statistics calculation
        dummy_scores = [0.5] * (len(chunks) -1)
        statistics = self.sentence_splitter.compute_statistics(chunks, dummy_scores, original_sentences)
        statistics['chunking_strategy'] = 'text_tiling'
        statistics['w'] = w
        statistics['k'] = k
        return chunks, statistics

if __name__ == "__main__":
    # --- Configuration ---
    W = 20  # Pseudo-sentence size
    K = 10  # Block size
    DOCUMENT_INDEX = 0

    print("🚀 Initializing TextTiling Chunker...")
    chunker = TextTilingChunker()

    print(f"📚 Loading document {DOCUMENT_INDEX} from NarrativeQA...")
    document_text, document_info = load_narrativeqa_document(document_index=DOCUMENT_INDEX)

    print("\nProcessing document...")
    chunks, stats = chunker.chunk_document(document_text, w=W, k=K)

    # --- Print Results ---
    print("\n" + "="*80)
    print("TEXTTILING CHUNKING STATISTICS")
    print("="*80)
    chunker.sentence_splitter.print_statistics(stats)
    print(f"Strategy Specifics: TextTiling (w={stats['w']}, k={stats['k']})")
    
    print("\n" + "="*80)
    print("EXAMPLE CHUNKS")
    print("="*80)
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ({chunk['sentence_count']} sentences) ---")
        print(f"Text: {chunk['text'][:250].strip()}{'...' if len(chunk['text']) > 250 else ''}")
