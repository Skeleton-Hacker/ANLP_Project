import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 1. DOCUMENT CHUNKING & EMBEDDING MODULE
# =============================================================================

class DocumentChunker:
    """Handles semantic chunking of long documents."""
    
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_tokens(self, text: str) -> List[str]:
        """Split document into overlapping chunks by word tokens."""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += self.chunk_size - self.overlap
            
        return chunks if chunks else ['']
    
    def chunk_by_sentences(self, text: str, sentences_per_chunk=5) -> List[str]:
        """Split document into chunks by sentences for better semantic coherence."""
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks if chunks else ['']


class ChunkEmbedder:
    """Embeds document chunks using pretrained sentence transformers."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        self.model = SentenceTransformer(model_name)
        self.device = device
        self.model.to(device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_chunks(self, chunks: List[str], batch_size=32) -> torch.Tensor:
        """Embed a list of text chunks."""
        embeddings = self.model.encode(
            chunks,
            convert_to_tensor=True,
            device=self.device,
            batch_size=batch_size,
            show_progress_bar=False
        )
        return embeddings


# =============================================================================
# 2. TASK ROUTER MODULE
# =============================================================================

class TaskRouter(nn.Module):
    """Classifies whether input is for summarization or QA."""
    
    def __init__(self, input_dim=384, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # Binary: 0=Summarization, 1=QA
        )
    
    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Returns logits for task classification."""
        return self.classifier(query_embedding)
    
    def predict_task(self, query_embedding: torch.Tensor) -> str:
        """Returns 'summarization' or 'qa'."""
        logits = self.forward(query_embedding)
        pred = torch.argmax(logits, dim=-1).item()
        return 'summarization' if pred == 0 else 'qa'


# =============================================================================
# 3. HIERARCHICAL DOCUMENT ENCODER
# =============================================================================

class HierarchicalDocEncoder(nn.Module):
    """
    Encodes document chunks hierarchically using Transformer layers.
    Produces a document-level representation.
    """
    
    def __init__(self, embedding_dim=384, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Positional encoding for chunk sequences
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, embedding_dim) * 0.02)
        
        # Transformer encoder for processing chunk sequences
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Special tokens
        self.doc_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
    
    def forward(self, chunk_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chunk_embeddings: [batch_size, num_chunks, embedding_dim]
        Returns:
            doc_embedding: [batch_size, embedding_dim]
        """
        batch_size, num_chunks, _ = chunk_embeddings.shape
        
        # Add positional encoding
        pos_enc = self.pos_encoder[:, :num_chunks, :]
        x = chunk_embeddings + pos_enc
        
        # Prepend document token
        doc_tokens = self.doc_token.expand(batch_size, -1, -1)
        x = torch.cat([doc_tokens, x], dim=1)
        
        # Process through transformer
        output = self.transformer(x)
        
        # Return document-level embedding (first token)
        return output[:, 0, :]


# =============================================================================
# 4. RETRIEVAL MODULE (for QA)
# =============================================================================

class ChunkRetriever:
    """Retrieves most relevant chunks for a given query."""
    
    def __init__(self, top_k=3):
        self.top_k = top_k
    
    def retrieve(self, query_embedding: torch.Tensor, 
                 chunk_embeddings: torch.Tensor,
                 chunks: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        Args:
            query_embedding: [embedding_dim]
            chunk_embeddings: [num_chunks, embedding_dim]
            chunks: List of chunk texts
        Returns:
            retrieved_embeddings: [top_k, embedding_dim]
            retrieved_texts: List of top_k chunk texts
        """
        # Compute cosine similarity
        query_norm = F.normalize(query_embedding.unsqueeze(0), dim=1)
        chunk_norm = F.normalize(chunk_embeddings, dim=1)
        similarities = torch.mm(query_norm, chunk_norm.T).squeeze(0)
        
        # Get top-k indices
        top_k = min(self.top_k, len(chunks))
        top_indices = torch.topk(similarities, top_k).indices
        
        retrieved_embeddings = chunk_embeddings[top_indices]
        retrieved_texts = [chunks[i] for i in top_indices.cpu().numpy()]
        
        return retrieved_embeddings, retrieved_texts


# =============================================================================
# 5. UNIFIED GENERATION MODULE
# =============================================================================

class UnifiedGenerator:
    """Handles both summarization and QA generation using T5."""
    
    def __init__(self, model_name='t5-base', device='cuda', max_length=512):
        self.device = device
        self.max_length = max_length
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
    
    def generate_summary(self, document_text: str, max_output_length=256) -> str:
        """Generate summary from document text."""
        # Truncate if too long
        input_text = f"summarize: {document_text}"
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_output_length,
                num_beams=4,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def generate_answer(self, question: str, context: str, max_output_length=128) -> str:
        """Generate answer from question and context."""
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_output_length,
                num_beams=4,
                early_stopping=True
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


# =============================================================================
# 6. COMPLETE SYNAPSE SYSTEM
# =============================================================================

class SYNAPSE:
    """
    Complete SYNAPSE framework for long document processing.
    Handles both summarization and question answering.
    """
    
    def __init__(self, 
                 chunk_size=512,
                 overlap=50,
                 embedding_model='all-MiniLM-L6-v2',
                 generator_model='t5-base',
                 device='cuda',
                 use_router=False):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing SYNAPSE on {self.device}...")
        
        # Components
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = ChunkEmbedder(model_name=embedding_model, device=self.device)
        self.doc_encoder = HierarchicalDocEncoder(
            embedding_dim=self.embedder.embedding_dim
        ).to(self.device)
        self.retriever = ChunkRetriever(top_k=3)
        self.generator = UnifiedGenerator(model_name=generator_model, device=self.device)
        
        # Optional: Task router (or use rule-based)
        self.use_router = use_router
        if use_router:
            self.router = TaskRouter(input_dim=self.embedder.embedding_dim).to(self.device)
        
        print("SYNAPSE initialization complete!")
    
    def process_document(self, document: str) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Process document into chunks and embeddings.
        Returns: (chunk_embeddings, chunks, doc_embedding)
        """
        # Step 1: Chunk the document
        chunks = self.chunker.chunk_by_tokens(document)
        
        # Step 2: Embed chunks
        chunk_embeddings = self.embedder.embed_chunks(chunks)
        
        # Step 3: Create document-level embedding
        chunk_emb_batch = chunk_embeddings.unsqueeze(0)  # [1, num_chunks, dim]
        doc_embedding = self.doc_encoder(chunk_emb_batch).squeeze(0)  # [dim]
        
        return chunk_embeddings, chunks, doc_embedding
    
    def determine_task(self, query: str) -> str:
        """Determine if task is summarization or QA."""
        if self.use_router:
            query_embedding = self.embedder.embed_chunks([query])
            task = self.router.predict_task(query_embedding)
        else:
            # Rule-based: check for question patterns
            query_lower = query.lower()
            question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which', '?']
            if any(word in query_lower for word in question_words):
                task = 'qa'
            else:
                task = 'summarization'
        
        return task
    
    def summarize(self, document: str, use_chunks: bool = True) -> str:
        """Generate summary of document."""
        if use_chunks:
            # Process document through our pipeline
            chunk_embeddings, chunks, doc_embedding = self.process_document(document)
            # Use first few chunks or combine for summary
            summary_context = ' '.join(chunks[:5])  # Use top chunks
        else:
            summary_context = document
        
        summary = self.generator.generate_summary(summary_context)
        return summary
    
    def answer_question(self, question: str, document: str) -> Dict[str, any]:
        """Answer question about document using retrieval."""
        # Process document
        chunk_embeddings, chunks, doc_embedding = self.process_document(document)
        
        # Embed question
        query_embedding = self.embedder.embed_chunks([question]).squeeze(0)
        
        # Retrieve relevant chunks
        retrieved_embeddings, retrieved_texts = self.retriever.retrieve(
            query_embedding, chunk_embeddings, chunks
        )
        
        # Generate answer from retrieved context
        context = ' '.join(retrieved_texts)
        answer = self.generator.generate_answer(question, context)
        
        return {
            'answer': answer,
            'retrieved_chunks': retrieved_texts,
            'context': context
        }
    
    def process_query(self, query: str, document: str) -> Dict[str, any]:
        """
        Main entry point: automatically determine task and process.
        """
        task = self.determine_task(query)
        
        if task == 'summarization':
            result = {
                'task': 'summarization',
                'output': self.summarize(document)
            }
        else:  # qa
            qa_result = self.answer_question(query, document)
            result = {
                'task': 'qa',
                'output': qa_result['answer'],
                'retrieved_chunks': qa_result['retrieved_chunks'],
                'context': qa_result['context']
            }
        
        return result


# =============================================================================
# 7. NARRATIVEQA DATASET HANDLER
# =============================================================================

class NarrativeQADataset(Dataset):
    """Custom dataset for NarrativeQA with SYNAPSE."""
    
    def __init__(self, hf_dataset, mode='qa'):
        self.data = hf_dataset
        self.mode = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract document (summary or full story)
        if 'document' in item and 'text' in item['document']:
            document = item['document']['text']
        elif 'summary' in item and 'text' in item['summary']:
            document = item['summary']['text']
        else:
            document = ""
        
        # Extract question and answer
        question = item['question']['text'] if 'question' in item else ""
        answer = item['answers'][0]['text'] if 'answers' in item and len(item['answers']) > 0 else ""
        
        return {
            'document': document,
            'question': question,
            'answer': answer,
            'story_id': item.get('document_id', idx)
        }


# =============================================================================
# 8. EVALUATION METRICS
# =============================================================================

def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores (simplified version)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(score[key].fmeasure)
        
        return {k: np.mean(v) * 100 for k, v in scores.items()}
    except ImportError:
        print("rouge-score not installed. Install with: pip install rouge-score")
        return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    
    common = set(pred_tokens) & set(truth_tokens)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# =============================================================================
# 9. TRAINING & EVALUATION PIPELINE
# =============================================================================

def evaluate_synapse(synapse: SYNAPSE, eval_dataset: Dataset, 
                     num_samples: int = 100, task: str = 'qa'):
    """Evaluate SYNAPSE on NarrativeQA."""
    
    print(f"\nEvaluating SYNAPSE on {num_samples} samples (task={task})...")
    
    predictions = []
    references = []
    
    for i in tqdm(range(min(num_samples, len(eval_dataset))), desc="Evaluating"):
        sample = eval_dataset[i]
        document = sample['document']
        
        if not document or len(document.strip()) < 10:
            continue
        
        if task == 'qa':
            question = sample['question']
            result = synapse.answer_question(question, document)
            pred = result['answer']
            ref = sample['answer']
        else:  # summarization
            pred = synapse.summarize(document)
            ref = sample['answer']  # Using answer as reference summary
        
        predictions.append(pred)
        references.append(ref)
    
    # Compute metrics
    if task == 'qa':
        f1_scores = [compute_f1_score(p, r) for p, r in zip(predictions, references)]
        avg_f1 = np.mean(f1_scores) * 100
        print(f"\nQA F1 Score: {avg_f1:.2f}%")
        return {'f1': avg_f1}
    else:
        rouge_scores = compute_rouge_scores(predictions, references)
        print(f"\nSummarization ROUGE Scores:")
        for metric, score in rouge_scores.items():
            print(f"  {metric}: {score:.2f}%")
        return rouge_scores


# =============================================================================
# 10. MAIN EXECUTION
# =============================================================================

def main():
    """Main pipeline for SYNAPSE with NarrativeQA."""
    
    print("="*80)
    print("SYNAPSE: Summarization & Answering with Precision via Scalable Encoding")
    print("="*80)
    
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHUNK_SIZE = 512
    OVERLAP = 50
    NUM_EVAL_SAMPLES = 50  # Adjust based on your needs
    
    # Initialize SYNAPSE
    synapse = SYNAPSE(
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        embedding_model='all-MiniLM-L6-v2',
        generator_model='t5-base',
        device=DEVICE,
        use_router=False  # Using rule-based routing
    )
    
    # Load NarrativeQA dataset
    print("\n" + "="*80)
    print("Loading NarrativeQA Dataset")
    print("="*80)
    
    # Load validation set for evaluation
    val_dataset_hf = load_dataset("deepmind/narrativeqa", split='validation').select(range(NUM_EVAL_SAMPLES))
    val_dataset = NarrativeQADataset(val_dataset_hf, mode='qa')
    
    print(f"Loaded {len(val_dataset)} validation samples")
    
    # Demo: Single example
    print("\n" + "="*80)
    print("Demo: Processing Single Example")
    print("="*80)
    
    sample = val_dataset[0]
    print(f"\nDocument excerpt: {sample['document'][:200]}...")
    print(f"\nQuestion: {sample['question']}")
    print(f"Ground Truth Answer: {sample['answer']}")
    
    # Test QA
    result = synapse.answer_question(sample['question'], sample['document'])
    print(f"\nSYNAPSE Answer: {result['answer']}")
    print(f"\nRetrieved Context: {result['context'][:200]}...")
    
    # Test Summarization
    print("\n" + "-"*80)
    summary = synapse.summarize(sample['document'])
    print(f"SYNAPSE Summary: {summary}")
    
    # Full evaluation
    print("\n" + "="*80)
    print("Full Evaluation on Validation Set")
    print("="*80)
    
    # Evaluate QA
    qa_metrics = evaluate_synapse(synapse, val_dataset, num_samples=NUM_EVAL_SAMPLES, task='qa')
    
    # Evaluate Summarization
    # summ_metrics = evaluate_synapse(synapse, val_dataset, num_samples=NUM_EVAL_SAMPLES, task='summarization')
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    
    # Save results
    results = {
        'qa_metrics': qa_metrics,
        # 'summarization_metrics': summ_metrics,
        'config': {
            'chunk_size': CHUNK_SIZE,
            'overlap': OVERLAP,
            'num_samples': NUM_EVAL_SAMPLES
        }
    }
    
    with open('synapse_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to 'synapse_results.json'")


if __name__ == "__main__":
    main()