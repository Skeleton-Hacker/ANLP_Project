import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import time
from datetime import datetime
import os
from semantic_chunking import SemanticSentenceChunker
from datasets import load_dataset



def load_narrativeqa_document_fixed(dataset_split: str = 'validation', document_index: int = 0) -> Tuple[str, Dict]:
    """
    Fixed version - Load a single document from NarrativeQA dataset.
    
    Args:
        dataset_split: Which split to use ('train', 'validation', 'test')
        document_index: Index of document to load
        
    Returns:
        Tuple of (document_text, document_info)
    """
    print(f"Loading document {document_index} from NarrativeQA {dataset_split} split...")
    
    # Load the entire dataset first to see its structure
    dataset = load_dataset("deepmind/narrativeqa", split=dataset_split)
    print(f"Dataset size: {len(dataset)}")
    
    if document_index >= len(dataset):
        raise ValueError(f"Document index {document_index} out of range. Dataset has {len(dataset)} documents.")
    
    sample = dataset[document_index]
    
    # Debug: Print the structure of the sample
    print(f"Sample keys: {list(sample.keys())}")
    if "document" in sample:
        print(f"Document keys: {list(sample['document'].keys())}")
    
    document_text = sample["document"]["text"]
    
    # Create a more unique document ID
    document_id = f"{dataset_split}_{document_index}_{sample['document'].get('id', 'unknown')}"
    
    document_info = {
        'dataset': 'narrativeqa',
        'split': dataset_split,
        'index': document_index,
        'document_id': document_id,
        'original_document_id': sample["document"].get("id", f"{dataset_split}_{document_index}"),
        'summary_available': bool(sample["document"].get("summary", {}).get("text", "")),
        'questions_available': len(sample.get("answers", [])) > 0,
        'original_length_chars': len(document_text),
        'original_length_words': len(document_text.split())
    }
    
    print(f"Loaded document: {document_info['document_id']}")
    print(f"Length: {document_info['original_length_chars']} chars, {document_info['original_length_words']} words")
    
    return document_text, document_info

class ParameterSweep:
    def __init__(self, device: str = 'cuda:3'):
        """
        Initialize parameter sweep for semantic chunking.
        
        Args:
            device: The device to run experiments on (e.g., 'cuda:3' or 'cpu')
        """
        self.device = device
        self.results = []
        
    def run_threshold_sweep(self, 
                          thresholds: List[float] = None,
                          num_documents: int = 10,
                          dataset_split: str = 'validation',
                          model_name: str = 'all-MiniLM-L6-v2') -> List[Dict]:
        """
        Run parameter sweep across multiple thresholds and documents sequentially on a single device.
        
        Args:
            thresholds: List of similarity thresholds to test
            num_documents: Number of documents to test on
            dataset_split: Which dataset split to use
            model_name: SBERT model to use
            
        Returns:
            List of results dictionaries
        """
        if thresholds is None:
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        
        print(f"Starting parameter sweep:")
        print(f"  Thresholds: {thresholds}")
        print(f"  Documents: {num_documents}")
        print(f"  Dataset split: {dataset_split}")
        print(f"  Model: {model_name}")
        print(f"  Device for execution: {self.device}")
        
        # Load documents once
        print(f"\nLoading {num_documents} documents...")
        documents = []
        
        dataset = load_dataset("deepmind/narrativeqa", split=dataset_split, trust_remote_code=True)
        
        unique_docs = {}
        for sample in dataset:
            doc_id = sample['document']['id']
            if doc_id not in unique_docs:
                unique_docs[doc_id] = sample
        
        unique_doc_list = list(unique_docs.values())
        dataset_size = len(unique_doc_list)
        print(f"Total unique documents in {dataset_split} split: {dataset_size}")
        
        num_documents = min(num_documents, dataset_size)
        print(f"Will load {num_documents} documents")
        
        for doc_idx in range(num_documents):
            try:
                sample = unique_doc_list[doc_idx]
                doc_text = sample["document"]["text"]
                doc_info = {
                    'dataset': 'narrativeqa',
                    'split': dataset_split,
                    'index': doc_idx,
                    'document_id': sample["document"].get("id", f"unique_{doc_idx}"),
                    'summary_available': bool(sample["document"].get("summary", {}).get("text", "")),
                    'questions_available': len(sample.get("answers", [])) > 0,
                    'original_length_chars': len(doc_text),
                    'original_length_words': len(doc_text.split())
                }
                documents.append((doc_text, doc_info))
                print(f"  ✅ Loaded document {doc_idx}: {doc_info['document_id']} ({doc_info['original_length_words']} words)")
            except Exception as e:
                print(f"  ❌ Error loading document {doc_idx}: {e}")
                continue
        
        print(f"\nSuccessfully loaded {len(documents)} unique documents")
        
        all_results = []
        
        for doc_idx, (doc_text, doc_info) in enumerate(documents):
            print(f"🚀 Processing document {doc_idx + 1}/{len(documents)}: {doc_info['document_id']} on {self.device}")
            for i, threshold in enumerate(thresholds):
                print(f"  -> Threshold {i+1}/{len(thresholds)} ({threshold})")
                
                try:
                    chunker = SemanticSentenceChunker(
                        model_name=model_name,
                        similarity_threshold=threshold,
                        device=self.device
                    )
                    
                    start_time = time.time()
                    chunks, statistics = chunker.chunk_document(doc_text)
                    processing_time = time.time() - start_time
                    
                    result = {
                        'threshold': threshold,
                        'document_id': doc_info['document_id'],
                        'document_index': doc_info['index'],
                        'document_length_words': doc_info['original_length_words'],
                        'document_length_chars': doc_info['original_length_chars'],
                        'processing_time': processing_time,
                        'timestamp': datetime.now().isoformat(),
                        **statistics
                    }
                    
                    if chunks:
                        chunk_lengths = [len(chunk['text']) for chunk in chunks]
                        result.update({
                            'chunk_text_lengths_mean': np.mean(chunk_lengths),
                            'chunk_text_lengths_std': np.std(chunk_lengths),
                            'chunk_text_lengths_min': min(chunk_lengths),
                            'chunk_text_lengths_max': max(chunk_lengths),
                            'compression_ratio': len(doc_text) / len(chunks) if len(chunks) > 0 else 0,
                        })
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"  ❌ Error with threshold {threshold}: {e}")
                    continue
        
        self.results = all_results
        return all_results
    
    def analyze_results(self, results: List[Dict] = None) -> Dict:
        """
        Analyze sweep results to find optimal parameters.
        
        Args:
            results: Results from parameter sweep
            
        Returns:
            Analysis dictionary with recommendations
        """
        if results is None:
            results = self.results
        
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("PARAMETER SWEEP ANALYSIS")
        print("="*80)
        
        # Group by threshold for analysis
        threshold_analysis = df.groupby('threshold').agg({
            'total_chunks': ['mean', 'std'],
            'average_chunk_size': ['mean', 'std'],
            'median_chunk_size': ['mean', 'std'],
            'max_chunk_size': ['mean', 'std'],
            'min_chunk_size': ['mean', 'std'],
            'average_similarity': ['mean', 'std'],
            'similarities_below_threshold': ['mean', 'std'],
            'processing_time': ['mean', 'std'],
            'compression_ratio': ['mean', 'std']
        }).round(3)
        
        print("\nThreshold Analysis Summary:")
        print(threshold_analysis)
        
        # Save summary to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'parameter_sweep_summary_{timestamp}.csv'
        try:
            threshold_analysis.to_csv(csv_filename)
            print(f"\n💾 Summary saved to {csv_filename}")
        except Exception as e:
            print(f"\n❌ Error saving summary to CSV: {e}")
        
        # Find optimal thresholds based on different criteria
        avg_by_threshold = df.groupby('threshold').mean(numeric_only=True)
        
        # Criteria for "optimal" threshold
        optimal_criteria = {
            'balanced_chunk_size': {
                'description': 'Balanced chunk size (5-8 sentences average)',
                'score_func': lambda row: abs(row['average_chunk_size'] - 6.5),  # Target 6.5 sentences
                'lower_is_better': True
            },
            'moderate_chunk_count': {
                'description': 'Moderate chunk count (not too many tiny chunks)',
                'score_func': lambda row: row['total_chunks'] / (row['document_length_words'] / 100),  # Chunks per 100 words
                'lower_is_better': True
            },
            'high_similarity_maintenance': {
                'description': 'Maintains high within-chunk similarity',
                'score_func': lambda row: row['average_similarity'],
                'lower_is_better': False
            },
            'processing_efficiency': {
                'description': 'Fast processing time',
                'score_func': lambda row: row['processing_time'],
                'lower_is_better': True
            },
            'good_compression': {
                'description': 'Good compression ratio (not too granular)',
                'score_func': lambda row: abs(row['compression_ratio'] - 500),  # Target ~500 chars per chunk
                'lower_is_better': True
            }
        }
        
        recommendations = {}
        
        for criterion, config in optimal_criteria.items():
            scores = avg_by_threshold.apply(config['score_func'], axis=1)
            
            if config['lower_is_better']:
                best_threshold = scores.idxmin()
                best_score = scores.min()
            else:
                best_threshold = scores.idxmax()
                best_score = scores.max()
            
            recommendations[criterion] = {
                'threshold': best_threshold,
                'score': best_score,
                'description': config['description']
            }
            
            print(f"\n{config['description']}:")
            print(f"  Recommended threshold: {best_threshold}")
            print(f"  Score: {best_score:.3f}")
        
        # Overall recommendation (weighted combination)
        print(f"\n{'='*60}")
        print("OVERALL RECOMMENDATIONS")
        print(f"{'='*60}")
        
        # Calculate composite score
        weights = {
            'balanced_chunk_size': 0.3,
            'moderate_chunk_count': 0.2,
            'high_similarity_maintenance': 0.25,
            'processing_efficiency': 0.1,
            'good_compression': 0.15
        }
        
        composite_scores = {}
        for threshold in avg_by_threshold.index:
            score = 0
            for criterion, weight in weights.items():
                criterion_score = optimal_criteria[criterion]['score_func'](avg_by_threshold.loc[threshold])
                
                # Normalize scores (simple min-max scaling)
                if optimal_criteria[criterion]['lower_is_better']:
                    criterion_score = 1 / (1 + criterion_score)  # Invert for lower-is-better
                else:
                    criterion_score = criterion_score  # Keep as-is for higher-is-better
                
                score += weight * criterion_score
            
            composite_scores[threshold] = score
        
        best_overall_threshold = max(composite_scores.keys(), key=lambda k: composite_scores[k])
        
        print(f"Best Overall Threshold: {best_overall_threshold}")
        print(f"Composite Score: {composite_scores[best_overall_threshold]:.3f}")
        
        # Print detailed stats for best threshold
        best_stats = avg_by_threshold.loc[best_overall_threshold]
        print(f"\nStats for threshold {best_overall_threshold}:")
        print(f"  Average chunks per document: {best_stats['total_chunks']:.1f}")
        print(f"  Average chunk size: {best_stats['average_chunk_size']:.1f} sentences")
        print(f"  Median chunk size: {best_stats['median_chunk_size']:.1f} sentences")
        print(f"  Average similarity: {best_stats['average_similarity']:.3f}")
        print(f"  Processing time: {best_stats['processing_time']:.2f}s")
        print(f"  Compression ratio: {best_stats['compression_ratio']:.1f} chars/chunk")
        
        analysis_result = {
            'threshold_analysis': threshold_analysis,
            'recommendations': recommendations,
            'best_overall_threshold': best_overall_threshold,
            'composite_scores': composite_scores,
            'best_threshold_stats': best_stats.to_dict()
        }
        
        return analysis_result
    
    def create_visualizations(self, results: List[Dict] = None, save_plots: bool = True):
        """
        Create a heatmap visualization of the average chunk size per document and threshold.
        """
        if results is None:
            results = self.results
        
        if not results:
            print("No results to visualize")
            return
        
        df = pd.DataFrame(results)
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(12, 8))
        
        # Chunk size distribution heatmap
        chunk_size_pivot = df.pivot_table(
            values='average_chunk_size', 
            index='threshold', 
            columns='document_index', 
            aggfunc='mean'
        )
        sns.heatmap(chunk_size_pivot, annot=True, fmt='.1f', cmap='viridis')
        plt.title('Chunk Size Heatmap\n(Threshold × Document)')
        plt.ylabel('Similarity Threshold')
        plt.xlabel('Document Index')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f'parameter_sweep_heatmap_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Heatmap saved to {plot_filename}")
        
        plt.close(fig) # Close the figure to free memory and prevent display
    
    def save_results(self, results: List[Dict] = None, analysis: Dict = None, 
                     filename: str = None):
        """
        Save sweep results and analysis to JSON file.
        """
        if results is None:
            results = self.results
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'parameter_sweep_results_{timestamp}.json'
        
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_experiments': len(results),
                'thresholds_tested': sorted(list(set(r['threshold'] for r in results))),
                'documents_tested': len(set(r['document_id'] for r in results)),
                'device_used': self.device
            },
            'results': results,
            'analysis': analysis if analysis else {}
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Results saved to {filename}")
        return filename


def main():
    """Main function to run the parameter sweep."""
    
    print("🚀 Starting Semantic Chunking Parameter Sweep")
    print("=" * 60)
    
    # Configuration
    THRESHOLDS = [0.1,0.2,0.3,0.4,0.45,0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    NUM_DOCUMENTS = 10 # Increase this to test more documents
    DEVICE = 'cuda:3' # Use GPU 3 for processing
    
    # Check GPU availability
    if 'cuda' in DEVICE and torch.cuda.is_available():
        try:
            gpu_id = int(DEVICE.split(':')[-1])
            gpu_count = torch.cuda.device_count()
            if gpu_id >= gpu_count:
                print(f"❌ Error: GPU {DEVICE} is not available. Max GPU ID is {gpu_count-1}.")
                return None, None
            print(f"✅ Verified GPU: {gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
        except (ValueError, IndexError):
            print(f"❌ Error: Invalid device format '{DEVICE}'. Should be 'cuda:X'.")
            return None, None
    elif 'cuda' in DEVICE:
        print("⚠️ Warning: CUDA not available, falling back to CPU")
        DEVICE = 'cpu'
    else:
        print(f"Using device: {DEVICE}")

    # Initialize and run sweep
    sweep = ParameterSweep(device=DEVICE)
    
    print(f"\nStarting sweep with {len(THRESHOLDS)} thresholds on {NUM_DOCUMENTS} documents...")
    print(f"Total experiments: {len(THRESHOLDS) * NUM_DOCUMENTS}")
    
    # Run the parameter sweep
    results = sweep.run_threshold_sweep(
        thresholds=THRESHOLDS,
        num_documents=NUM_DOCUMENTS,
        dataset_split='validation',
        model_name='all-MiniLM-L6-v2'
    )
    
    if not results:
        print("\n❌ No results were generated. Exiting.")
        return None, None

    print(f"\n✅ Completed {len(results)} experiments!")
    
    # Analyze results
    print("\n🔍 Analyzing results...")
    analysis = sweep.analyze_results(results)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    sweep.create_visualizations(results, save_plots=True)
    
    # Save results
    print("\n💾 Saving results...")
    filename = sweep.save_results(results, analysis)
    
    print(f"\n🎉 Parameter sweep complete!")
    print(f"   Results saved to: {filename}")
    print(f"   Best threshold: {analysis.get('best_overall_threshold', 'N/A')}")
    
    return results, analysis


if __name__ == "__main__":
    results, analysis = main()