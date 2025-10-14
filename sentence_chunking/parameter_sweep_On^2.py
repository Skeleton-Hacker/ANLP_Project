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

class ONSquareChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def chunk_document(self, doc_text: str, threshold: float) -> Dict:
        # Step 1: sentence split
        sentences = SemanticSentenceChunker().split_into_sentences(text=doc_text)
        n = len(sentences)
        if n == 0:
            return {"total_chunks": 0}

        # Step 2: embeddings
        embeddings = self.model.encode(sentences, convert_to_tensor=True, device=self.model.device)

        # Step 3: pairwise similarity (O(n²))
        sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

        # Step 4: Graph-based chunking
        visited = [False] * n
        chunks = []
        for i in range(n):
            if not visited[i]:
                chunk_indices = []
                q = [i]
                visited[i] = True
                while q:
                    u = q.pop(0)
                    chunk_indices.append(u)
                    for v in range(n):
                        if sim_matrix[u, v] >= threshold and not visited[v]:
                            visited[v] = True
                            q.append(v)
                
                chunk_sentences = [sentences[idx] for idx in sorted(chunk_indices)]
                chunks.append(chunk_sentences)

        # Collect stats
        if not chunks:
            return {
                "total_chunks": 0,
                "average_chunk_size": 0,
                "median_chunk_size": 0,
                "max_chunk_size": 0,
                "min_chunk_size": 0,
            }
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

    chunker = ONSquareChunker(device=device)

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
    df = run_one_document_sweep(THRESHOLDS, device="cuda:0", doc_idx=0)
    print("\nFinal Sweep Results:\n", df)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"parameter_sweep_results_n2_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")
