import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict
from datasets import load_dataset
from semantic_chunking import SemanticSentenceChunker
from sentence_transformers import SentenceTransformer, util
import time

def load_single_document(dataset_split: str = 'validation', document_index: int = 0):
    dataset = load_dataset("deepmind/narrativeqa", split=dataset_split)
    sample = dataset[document_index]
    doc_text = sample["document"]["text"]
    return doc_text, {
        "dataset": "narrativeqa",
        "split": dataset_split,
        "index": document_index,
        "document_id": sample["document"].get("id", f"{dataset_split}_{document_index}"),
    }

class ONChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def chunk_document(self, doc_text: str, threshold: float) -> Dict:
        # Step 1: sentence split
        sentences = SemanticSentenceChunker().split_into_sentences(text=doc_text)
        n = len(sentences)
        if n == 0:
            return {"total_chunks": 0}
        if n == 1:
            return {"total_chunks": 1, "average_chunk_size": 1, "median_chunk_size": 1, "max_chunk_size": 1, "min_chunk_size": 1}

        # Step 2: embeddings (compute once, reuse)
        embeddings = self.model.encode(sentences, convert_to_tensor=True, device=self.model.device)

        # Step 3: O(n) neighbor comparison and chunking
        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_embeddings = [embeddings[0]]
        
        for i in range(1, n):
            # Compare current sentence with the last sentence in current chunk
            prev_embedding = current_chunk_embeddings[-1]
            curr_embedding = embeddings[i]
            
            similarity = util.cos_sim(prev_embedding.unsqueeze(0), curr_embedding.unsqueeze(0)).item()
            
            if similarity >= threshold:
                # Merge with current chunk
                current_chunk.append(sentences[i])
                current_chunk_embeddings.append(curr_embedding)
            else:
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = [sentences[i]]
                current_chunk_embeddings = [curr_embedding]
        
        # Don't forget the last chunk
        chunks.append(current_chunk)

        # Collect stats
        chunk_sizes = [len(c) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "average_chunk_size": np.mean(chunk_sizes),
            "median_chunk_size": np.median(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
        }

def run_one_document_sweep(thresholds: List[float], device="cpu", doc_idx=0):
    doc_text, doc_info = load_single_document(document_index=doc_idx)
    print(f"Loaded document {doc_info['document_id']}")

    chunker = ONChunker(device=device)

    results = []
    for threshold in thresholds:
        start = time.time()
        stats = chunker.chunk_document(doc_text, threshold)
        stats.update({
            "threshold": threshold,
            "processing_time": time.time() - start,
            "document_id": doc_info["document_id"],
            "timestamp": datetime.now().isoformat(),
        })
        results.append(stats)
        print(f"Threshold {threshold:.2f}: {stats}")

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    df = run_one_document_sweep(THRESHOLDS, device="cuda:1", doc_idx=0)
    print("\nFinal Sweep Results:\n", df)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"parameter_sweep_n_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")
