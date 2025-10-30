import logging
import re
import pickle
import tempfile
from pathlib import Path
from typing import List
from datetime import datetime
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from load_dataset import load_train
from accelerate import Accelerator
import html

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticSentenceChunker:
    """Simple sentence splitter."""
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Remove HTML comments
        cleaned = re.sub(r'<!--.*?-->', ' ', text, flags=re.S)
        # Remove HTML tags
        cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
        # Unescape HTML entities (&amp;, &nbsp;, ...)
        cleaned = html.unescape(cleaned)
        # Replace markdown links [text](url) with text
        cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
        # Remove bare URLs
        cleaned = re.sub(r'http\S+|www\.\S+', ' ', cleaned)
        # Remove backticks and inline code markers
        cleaned = re.sub(r'`+', '', cleaned)
        # Remove repeated underscores/asterisks used for markdown emphasis (e.g. __bold__, **bold**)
        cleaned = re.sub(r'[_*]{2,}', '', cleaned)
        # Replace remaining single underscores/asterisks with a space
        cleaned = re.sub(r'[_*]', ' ', cleaned)
        # Normalize newlines and carriage returns to spaces
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
        # Collapse multiple whitespace into a single space
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Split into sentences (keep default behavior of splitting after .!?)
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        return [s.strip() for s in sentences if s.strip()]


class GraphChunker:
    """Graph-based semantic chunker."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", threshold: float = 0.5, accelerator: Accelerator = None):
        self.threshold = threshold
        self.sentence_splitter = SemanticSentenceChunker()
        self.accelerator = accelerator
        
        # Initialize model
        self.model = SentenceTransformer(model_name)
        
        # Move to appropriate device
        if self.accelerator is not None:
            self.device = self.accelerator.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self.model.to(self.device)
        
        if self.accelerator is None or self.accelerator.is_main_process:
            logger.info(f"Model loaded on {self.device}")
    
    def chunk_sentences(self, sentences: List[str]) -> List[str]:
        """Chunk sentences using graph-based semantic similarity."""
        n = len(sentences)
        
        if n == 0:
            return []
        if n == 1:
            return [sentences[0]]
        
        # Compute embeddings
        with torch.no_grad():
            embeddings = self.model.encode(
                sentences,
                convert_to_tensor=True,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self.device
            )
        
        # Compute similarity matrix
        with torch.no_grad():
            sim_matrix = util.cos_sim(embeddings, embeddings)
        
        # Graph-based chunking
        visited = torch.zeros(n, dtype=torch.bool, device=self.device)
        chunks = []
        
        for i in range(n):
            if not visited[i]:
                chunk_indices = [i]
                visited[i] = True
                queue = [i]
                
                while queue:
                    current = queue.pop(0)
                    remaining_indices = torch.where(~visited)[0]
                    
                    if len(remaining_indices) > 0:
                        similarities = sim_matrix[current, remaining_indices]
                        connected_indices = remaining_indices[similarities >= self.threshold]
                        
                        if len(connected_indices) > 0:
                            chunk_indices.extend(connected_indices.tolist())
                            visited[connected_indices] = True
                            queue.extend(connected_indices.tolist())
                
                chunk_indices.sort()
                chunk_text = " ".join([sentences[idx] for idx in chunk_indices])
                chunks.append(chunk_text)
        
        return chunks


def main():
    """Run threshold analysis on one document with multi-GPU support."""
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        logger.info(f"Running with {accelerator.num_processes} GPU(s)")
    
    # Load first 4 stories from train (only on main process)
    num_docs_to_process = 4
    if accelerator.is_main_process:
        logger.info(f"Loading first {num_docs_to_process} documents from train set...")
        stories = load_train(
            dataset_name="deepmind/narrativeqa",
            max_samples=num_docs_to_process,
            trust_remote_code=False,
            group_by_story=True
        )

        if not stories:
            logger.error("No stories loaded!")
            return

        # Collect the first N stories and their sentence lists
        docs_sentences = {}
        for i, (story_id, story) in enumerate(stories.items()):
            if i >= num_docs_to_process:
                break
            doc_text = story.get('document', {}).get('text', '')
            if not doc_text:
                logger.warning(f"Story {story_id} has no document text; skipping")
                continue
            sents = SemanticSentenceChunker.split_into_sentences(doc_text)
            logger.info(f"Loaded story {story_id} with {len(sents)} sentences")
            docs_sentences[story_id] = sents

        if not docs_sentences:
            logger.error("No valid documents with text found")
            return

        # Save sentences dict to temp file for sharing
        temp_file = Path(tempfile.gettempdir()) / "sentences_temp.pkl"
        with open(temp_file, 'wb') as f:
            pickle.dump(docs_sentences, f)
    
    # Wait for main process to finish loading
    accelerator.wait_for_everyone()
    
    # All processes load the sentences dict
    temp_file = Path(tempfile.gettempdir()) / "sentences_temp.pkl"
    with open(temp_file, 'rb') as f:
        docs_sentences = pickle.load(f)
    
    # Clean up temp file on main process
    if accelerator.is_main_process:
        temp_file.unlink()
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.52, 0.54, 0.56,0.566,0.57,0.573,0.576, 0.58 ,0.6,0.63,0.67,0.7,0.73,0.76 ,0.8, 0.9]
    
    # Divide thresholds across GPUs
    thresholds_per_gpu = len(thresholds) // accelerator.num_processes
    start_idx = accelerator.process_index * thresholds_per_gpu
    
    if accelerator.process_index == accelerator.num_processes - 1:
        end_idx = len(thresholds)
    else:
        end_idx = start_idx + thresholds_per_gpu
    
    local_thresholds = thresholds[start_idx:end_idx]
    
    logger.info(f"GPU {accelerator.process_index}: Processing thresholds {local_thresholds}")
    
    # Initialize chunker
    chunker = GraphChunker(model_name="BAAI/bge-large-en-v1.5", accelerator=accelerator)
    
    local_results = []

    # For each local threshold, compute metrics for each document and then average across documents
    for tau in local_thresholds:
        logger.info(f"GPU {accelerator.process_index}: Testing threshold={tau}")
        chunker.threshold = tau

        per_doc_metrics = []

        for story_id, sentences in docs_sentences.items():
            start = datetime.now()
            chunks = chunker.chunk_sentences(sentences)
            end = datetime.now()
            elapsed = (end - start).total_seconds()

            # Calculate statistics per document
            chunk_lens = [len(SemanticSentenceChunker.split_into_sentences(c)) for c in chunks]
            num_chunks = len(chunks)

            avg = float(np.mean(chunk_lens)) if chunk_lens else 0.0
            med = float(np.median(chunk_lens)) if chunk_lens else 0.0
            mx = int(max(chunk_lens)) if chunk_lens else 0

            if chunk_lens:
                q1 = float(np.percentile(chunk_lens, 2))
                q3 = float(np.percentile(chunk_lens, 98))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                filtered = [x for x in chunk_lens if (x >= lower and x <= upper)]
                outliers = [x for x in chunk_lens if not (x >= lower and x <= upper)]
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(chunk_lens)) * 100 if chunk_lens else 0.0

                avg_no_out = float(np.mean(filtered)) if filtered else 0.0
                med_no_out = float(np.median(filtered)) if filtered else 0.0
                max_no_out = int(max(filtered)) if filtered else 0
            else:
                outlier_count = 0
                outlier_pct = 0.0
                avg_no_out = 0.0
                med_no_out = 0.0
                max_no_out = 0

            per_doc_metrics.append({
                'tau': tau,
                'chunks': num_chunks,
                'avg': avg,
                'med': med,
                'max': mx,
                'avg_no_out': avg_no_out,
                'med_no_out': med_no_out,
                'max_no_out': max_no_out,
                'outliers': outlier_count,
                'outlier_pct': outlier_pct,
                'time': elapsed
            })

        # Average metrics across documents (simple mean for numeric fields)
        if per_doc_metrics:
            agg = {
                'tau': tau,
            }
            numeric_keys = ['chunks', 'avg', 'med', 'max', 'avg_no_out', 'med_no_out', 'max_no_out', 'outliers', 'outlier_pct', 'time']
            for k in numeric_keys:
                vals = [d[k] for d in per_doc_metrics]
                # For integer-valued fields like chunks, max, outliers we keep floats during averaging
                agg[k] = float(np.mean(vals)) if vals else 0.0

            local_results.append(agg)
    
    # Save local results to temp files
    temp_result_file = Path(tempfile.gettempdir()) / f"results_gpu_{accelerator.process_index}.pkl"
    with open(temp_result_file, 'wb') as f:
        pickle.dump(local_results, f)
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    # Print table on main process
    if accelerator.is_main_process:
        # Gather all results
        results = []
        for i in range(accelerator.num_processes):
            temp_result_file = Path(tempfile.gettempdir()) / f"results_gpu_{i}.pkl"
            with open(temp_result_file, 'rb') as f:
                gpu_results = pickle.load(f)
                results.extend(gpu_results)
            temp_result_file.unlink()  # Clean up
        
        # Sort by tau
        results = sorted(results, key=lambda x: x['tau'])
        
        print("\n" + "="*70)
        print("THRESHOLD ANALYSIS RESULTS")
        print("="*70)
        
        headers = ["tau", "Chunks", "Avg", "Med", "Max", "Avg_no_out", "Med_no_out", "Max_no_out", "Outliers", "Outlier%", "Time"]
        fmt = "{:<6} {:>8} {:>10} {:>8} {:>8} {:>12} {:>11} {:>11} {:>9} {:>9} {:>10}"

        header_line = fmt.format(*headers)
        print(header_line)
        print("-" * 70)
        
        for row in results:
            print(fmt.format(
                f"{row['tau']:.3f}",
                f"{row['chunks']}",
                f"{row['avg']:.3f}",
                f"{row['med']:.0f}",
                f"{row['max']}",
                f"{row.get('avg_no_out', 0.0):.3f}",
                f"{row.get('med_no_out', 0.0):.0f}",
                f"{row.get('max_no_out', 0)}",
                f"{row.get('outliers', 0)}",
                f"{row.get('outlier_pct', 0.0):.1f}",
                f"{row['time']:.3f}"
            ))
        
        print("="*70)


if __name__ == "__main__":
    main()