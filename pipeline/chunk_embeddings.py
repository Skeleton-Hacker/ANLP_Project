
import logging
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dataclasses import dataclass, field
from accelerate import Accelerator

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
@dataclass
class AEConfig:
    """Configuration for autoencoder training."""
    # Data settings
    chunked_data_dir: str = "chunked_data"
    output_dir: str = "chunked_data"
    model_save_dir: str = "models"
    splits: List[str] = field(default_factory=lambda: ["train", "validation", "test"])
    
    # Embedding settings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    # Autoencoder architecture
    input_dim: int = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    latent_dim: int = 64
    
    # Training settings
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 2
    
    # Hardware settings - removed device, accelerator will handle this
    num_workers: int = 4
    
    # Random seed
    seed: int = 42


class ChunkEmbeddingDataset(Dataset):
    """Dataset for chunk embeddings."""
    
    def __init__(self, embeddings: np.ndarray):
        """Initialize dataset with embeddings.
        
        Args:
            embeddings: Numpy array of shape (num_chunks, embedding_dim).
        """
        self.embeddings = torch.FloatTensor(embeddings)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]


class ChunkAutoencoder(nn.Module):
    """Autoencoder for chunk embeddings."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout_rate: float = 0.1
    ):
        """Initialize autoencoder.
        
        Args:
            input_dim: Dimension of input embeddings.
            hidden_dims: List of hidden layer dimensions.
            latent_dim: Dimension of latent space (bottleneck).
            dropout_rate: Dropout rate for regularization.
        """
        super(ChunkAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            
        Returns:
            Latent representation of shape (batch_size, latent_dim).
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim).
            
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            
        Returns:
            Tuple of (reconstructed, latent) tensors.
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_chunks_from_pickle(pickle_path: Path, accelerator: Optional[Accelerator] = None) -> List[str]:
    """Load chunks from a pickle file.
    
    Args:
        pickle_path: Path to the pickle file containing chunked stories.
        accelerator: Accelerator instance for distributed processing.
        
    Returns:
        List of all chunk strings.
    """
    if accelerator and accelerator.is_main_process:
        logger.info(f"Loading chunks from {pickle_path}...")
    elif not accelerator:
        logger.info(f"Loading chunks from {pickle_path}...")
    
    with open(pickle_path, 'rb') as f:
        stories = pickle.load(f)
    
    # Extract all chunks from all stories with progress bar
    all_chunks = []
    story_items = list(stories.items())
    
    disable_tqdm = accelerator is not None and not accelerator.is_main_process
    for story_id, story_data in tqdm(story_items, desc="Extracting chunks from stories", disable=disable_tqdm):
        chunks = story_data.get('chunks', [])
        all_chunks.extend(chunks)
    
    if accelerator and accelerator.is_main_process:
        logger.info(f"Loaded {len(all_chunks)} chunks from {len(stories)} stories")
    elif not accelerator:
        logger.info(f"Loaded {len(all_chunks)} chunks from {len(stories)} stories")
    
    return all_chunks


def generate_chunk_embeddings(
    chunks: List[str],
    model_name: str,
    accelerator: Accelerator,
    batch_size: int = 32
) -> np.ndarray:
    """Generate embeddings for chunks using sentence transformer.
    
    Args:
        chunks: List of chunk strings.
        model_name: Name of the sentence transformer model.
        accelerator: Accelerator instance for device management.
        batch_size: Batch size for embedding generation.
        
    Returns:
        Numpy array of shape (num_chunks, embedding_dim).
    """
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    
    model = SentenceTransformer(model_name, device=accelerator.device)
    
    embeddings = model.encode(
        chunks,
        convert_to_tensor=True,
        device=accelerator.device,
        show_progress_bar=accelerator.is_main_process,
        batch_size=batch_size,
        normalize_embeddings=True  # Normalize for better training
    )
    
    # Convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    
    if accelerator.is_main_process:
        logger.info(f"Generated embeddings with shape: {embeddings_np.shape}")
    
    return embeddings_np


def train_autoencoder(
    train_embeddings: np.ndarray,
    val_embeddings: Optional[np.ndarray],
    config: AEConfig,
    accelerator: Accelerator
) -> ChunkAutoencoder:
    """Train the autoencoder model.
    
    Args:
        train_embeddings: Training embeddings of shape (num_train, embedding_dim).
        val_embeddings: Validation embeddings of shape (num_val, embedding_dim), or None.
        config: Configuration object.
        accelerator: Accelerator instance for distributed training.
        
    Returns:
        Trained autoencoder model.
    """
    if accelerator.is_main_process:
        logger.info("=" * 80)
        logger.info("Training Autoencoder")
        logger.info("=" * 80)
    
    # Create datasets
    train_dataset = ChunkEmbeddingDataset(train_embeddings)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    if val_embeddings is not None:
        val_dataset = ChunkEmbeddingDataset(val_embeddings)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    else:
        val_loader = None
    
    # Initialize model
    model = ChunkAutoencoder(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim
    )
    
    if accelerator.is_main_process:
        logger.info(f"Model architecture:\n{model}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=accelerator.is_main_process
    )
    
    # Prepare with accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = len(train_loader)
        
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]",
            disable=not accelerator.is_main_process
        )
        
        for batch in train_pbar:
            # Forward pass
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            
            # Backward pass
            accelerator.backward(loss)
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            num_val_batches = len(val_loader)
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", disable=not accelerator.is_main_process)
            with torch.no_grad():
                for batch in val_pbar:
                    reconstructed, _ = model(batch)
                    loss = criterion(reconstructed, batch)
                    val_loss += loss.item()
                    val_pbar.set_postfix({'val_loss': loss.item()})
            
            avg_val_loss = val_loss / num_val_batches
            val_losses.append(avg_val_loss)
            
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch+1}/{config.num_epochs} - "
                           f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = accelerator.unwrap_model(model).state_dict()
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    if accelerator.is_main_process:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        else:
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {avg_train_loss:.6f}")
    
    # Load best model if validation was used
    if val_loader is not None and 'best_model_state' in locals():
        accelerator.unwrap_model(model).load_state_dict(best_model_state)
        if accelerator.is_main_process:
            logger.info(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    if accelerator.is_main_process:
        logger.info("Training completed!")
    
    # Unwrap model for return
    model = accelerator.unwrap_model(model)
    
    return model


def encode_all_chunks(
    model: ChunkAutoencoder,
    chunks: List[str],
    embedding_model_name: str,
    accelerator: Accelerator,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode all chunks using the trained autoencoder.
    
    Args:
        model: Trained autoencoder model.
        chunks: List of chunk strings.
        embedding_model_name: Name of the sentence transformer model.
        accelerator: Accelerator instance for device management.
        batch_size: Batch size for processing.
        
    Returns:
        Tuple of (original_embeddings, encoded_embeddings) as numpy arrays.
    """
    if accelerator.is_main_process:
        logger.info(f"Encoding {len(chunks)} chunks...")
    
    # Generate original embeddings
    original_embeddings = generate_chunk_embeddings(
        chunks,
        embedding_model_name,
        accelerator,
        batch_size
    )
    
    # Encode using autoencoder
    model = accelerator.unwrap_model(model)
    model.eval()
    model = model.to(accelerator.device)
    
    dataset = ChunkEmbeddingDataset(original_embeddings)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    encoded_list = []
    
    with torch.no_grad():
        encode_pbar = tqdm(loader, desc="Encoding with AE", disable=not accelerator.is_main_process)
        for batch in encode_pbar:
            batch = batch.to(accelerator.device)
            latent = model.encode(batch)
            encoded_list.append(latent.cpu().numpy())
            encode_pbar.set_postfix({'batch_size': batch.size(0)})
    
    encoded_embeddings = np.vstack(encoded_list)
    
    if accelerator.is_main_process:
        logger.info(f"Original embeddings shape: {original_embeddings.shape}")
        logger.info(f"Encoded embeddings shape: {encoded_embeddings.shape}")
    
    return original_embeddings, encoded_embeddings


def process_and_save_split(
    split_name: str,
    model: ChunkAutoencoder,
    config: AEConfig,
    accelerator: Accelerator
) -> Optional[Dict[str, Any]]:
    """Process a split and save encoded chunks.
    
    Args:
        split_name: Name of the split ('train', 'validation', 'test').
        model: Trained autoencoder model.
        config: Configuration object.
        accelerator: Accelerator instance for device management.
        
    Returns:
        Dictionary with encoded data (only on main process).
    """
    if accelerator.is_main_process:
        logger.info(f"\nProcessing {split_name} split...")
    
    # Load chunks
    input_path = Path(config.chunked_data_dir) / f"{split_name}_chunks.pkl"
    if not input_path.exists():
        if accelerator.is_main_process:
            logger.error(f"Input file not found: {input_path}")
        return None
    
    with open(input_path, 'rb') as f:
        stories = pickle.load(f)
    
    # Extract chunks with story mapping
    chunks = []
    chunk_to_story = []  # (story_id, chunk_index) for each chunk
    
    story_items = list(stories.items())
    for story_id, story_data in tqdm(story_items, desc="Extracting chunks with mapping", 
                                    disable=not accelerator.is_main_process):
        story_chunks = story_data.get('chunks', [])
        for chunk_idx, chunk in enumerate(story_chunks):
            chunks.append(chunk)
            chunk_to_story.append((story_id, chunk_idx))
    
    if accelerator.is_main_process:
        logger.info(f"Loaded {len(chunks)} chunks from {len(stories)} stories")
    
    # Encode chunks
    original_embeddings, encoded_embeddings = encode_all_chunks(
        model,
        chunks,
        config.embedding_model_name,
        accelerator,
        config.batch_size
    )
    
    # Only main process saves the data
    if accelerator.is_main_process:
        # Create encoded data structure
        encoded_data = {
            'stories': {},
            'metadata': {
                'split': split_name,
                'num_stories': len(stories),
                'num_chunks': len(chunks),
                'embedding_model': config.embedding_model_name,
                'original_dim': original_embeddings.shape[1],
                'encoded_dim': encoded_embeddings.shape[1],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Reconstruct story structure with embeddings
        logger.info("Reconstructing story structure with embeddings...")
        for idx, (story_id, chunk_idx) in tqdm(enumerate(chunk_to_story), 
                                               desc="Reconstructing stories", 
                                               total=len(chunk_to_story)):
            if story_id not in encoded_data['stories']:
                # Copy original story data
                encoded_data['stories'][story_id] = {
                    'document': stories[story_id]['document'],
                    'questions': stories[story_id]['questions'],
                    'answers': stories[story_id]['answers'],
                    'chunks': [],
                    'original_embeddings': [],
                    'encoded_embeddings': []
                }
            
            encoded_data['stories'][story_id]['chunks'].append(chunks[idx])
            encoded_data['stories'][story_id]['original_embeddings'].append(
                original_embeddings[idx]
            )
            encoded_data['stories'][story_id]['encoded_embeddings'].append(
                encoded_embeddings[idx]
            )
        
        # Convert embedding lists to numpy arrays
        logger.info("Converting embedding lists to numpy arrays...")
        story_ids = list(encoded_data['stories'].keys())
        for story_id in tqdm(story_ids, desc="Converting to numpy arrays"):
            encoded_data['stories'][story_id]['original_embeddings'] = np.array(
                encoded_data['stories'][story_id]['original_embeddings']
            )
            encoded_data['stories'][story_id]['encoded_embeddings'] = np.array(
                encoded_data['stories'][story_id]['encoded_embeddings']
            )
        
        # Save encoded data
        output_path = Path(config.output_dir) / f"{split_name}_chunks_encoded.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(encoded_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved encoded data to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        return encoded_data
    
    return None


def main():
    """Main training and encoding pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = AEConfig()
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Set seed
    set_seed(config.seed)
    
    if accelerator.is_main_process:
        logger.info("=" * 80)
        logger.info("Chunk Autoencoder Training Pipeline with Accelerate")
        logger.info("=" * 80)
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Embedding model: {config.embedding_model_name}")
        logger.info(f"Input dimension: {config.input_dim}")
        logger.info(f"Hidden dimensions: {config.hidden_dims}")
        logger.info(f"Latent dimension: {config.latent_dim}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Epochs: {config.num_epochs}")
        logger.info(f"Learning rate: {config.learning_rate}")
        logger.info("=" * 80)
    
    # Create output directories
    if accelerator.is_main_process:
        Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Wait for main process to create directories
    accelerator.wait_for_everyone()
    
    # Define pipeline phases for progress tracking
    pipeline_phases = [
        "Loading Training Data",
        "Generating Embeddings", 
        "Training Autoencoder",
        "Encoding All Splits"
    ]
    
    # Load and prepare training data
    if accelerator.is_main_process:
        logger.info("\n" + "=" * 80)
        logger.info("Loading Training Data")
        logger.info("=" * 80)
    
    train_path = Path(config.chunked_data_dir) / "train_chunks.pkl"
    val_path = Path(config.chunked_data_dir) / "validation_chunks.pkl"
    
    if not train_path.exists():
        if accelerator.is_main_process:
            logger.error(f"Training data not found: {train_path}")
            logger.error("Please run semantic_chunking.py first to generate chunked data.")
        return
    
    # Load training chunks
    train_chunks = load_chunks_from_pickle(train_path, accelerator)
    if accelerator.is_main_process:
        logger.info(f"Loaded {len(train_chunks)} training chunks")
    
    # Load validation chunks if available
    if val_path.exists():
        val_chunks = load_chunks_from_pickle(val_path, accelerator)
        if accelerator.is_main_process:
            logger.info(f"Loaded {len(val_chunks)} validation chunks")
    else:
        val_chunks = None
        if accelerator.is_main_process:
            logger.warning("Validation data not found. Training without validation.")
    
    # Generate embeddings
    if accelerator.is_main_process:
        logger.info("\n" + "=" * 80)
        logger.info("Generating Embeddings")
        logger.info("=" * 80)
    
    train_embeddings = generate_chunk_embeddings(
        train_chunks,
        config.embedding_model_name,
        accelerator,
        config.batch_size
    )
    
    if val_chunks is not None:
        val_embeddings = generate_chunk_embeddings(
            val_chunks,
            config.embedding_model_name,
            accelerator,
            config.batch_size
        )
    else:
        val_embeddings = None
    
    # Train autoencoder
    if accelerator.is_main_process:
        logger.info("\n" + "=" * 80)
        logger.info("Training Autoencoder")
        logger.info("=" * 80)
    
    model = train_autoencoder(train_embeddings, val_embeddings, config, accelerator)
    
    # Save model (only on main process)
    if accelerator.is_main_process:
        model_save_path = Path(config.model_save_dir) / "AE"
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        model_file = model_save_path / "autoencoder.pt"
        config_file = model_save_path / "config.pkl"
        
        torch.save(model.state_dict(), model_file)
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"\nModel saved to: {model_file}")
        logger.info(f"Config saved to: {config_file}")
    
    # Wait for model to be saved
    accelerator.wait_for_everyone()
    
    # Encode all splits
    if accelerator.is_main_process:
        logger.info("\n" + "=" * 80)
        logger.info("Encoding All Splits")
        logger.info("=" * 80)
    
    splits_progress = tqdm(config.splits, desc="Processing splits", disable=not accelerator.is_main_process)
    for split in splits_progress:
        try:
            process_and_save_split(split, model, config, accelerator)
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"Error processing {split} split: {e}", exc_info=True)
        
        # Synchronize processes between splits
        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Completed Successfully!")
        logger.info("=" * 80)
        logger.info(f"Model saved in: {Path(config.model_save_dir) / 'AE'}")
        logger.info(f"Encoded data saved in: {config.output_dir}")
        
        # List output files
        output_files = list(Path(config.output_dir).glob("*_encoded.pkl"))
        if output_files:
            logger.info("\nEncoded files:")
            for f in sorted(output_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"  - {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
