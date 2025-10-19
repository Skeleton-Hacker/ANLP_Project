import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoencoders.standard_ae import StandardAutoEncoder
from autoencoders.variational_ae import VariationalAutoEncoder
from autoencoders.beta_vae import BetaVAE
from autoencoders.sparse_ae import SparseAutoEncoder
from autoencoders.denoising_ae import DenoisingAutoEncoder
from autoencoders.vq_vae import VQVAE

class AutoencoderCompressor:
    """Wrapper class for compressing embeddings using various autoencoders"""
    
    AUTOENCODER_TYPES = {
        'standard': StandardAutoEncoder,
        'variational': VariationalAutoEncoder,
        'beta_vae': BetaVAE,
        'sparse': SparseAutoEncoder,
        'denoising': DenoisingAutoEncoder,
        'vq_vae': VQVAE
    }
    
    def __init__(
        self,
        autoencoder_type: str,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        device: Optional[str] = None,
        **kwargs
    ):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.autoencoder_type = autoencoder_type
        
        if autoencoder_type not in self.AUTOENCODER_TYPES:
            raise ValueError(f"Unknown autoencoder type: {autoencoder_type}")
        
        autoencoder_class = self.AUTOENCODER_TYPES[autoencoder_type]
        
        if autoencoder_type == 'beta_vae':
            beta = kwargs.get('beta', 4.0)
            self.model = autoencoder_class(input_dim, hidden_dim, latent_dim, beta=beta)
        elif autoencoder_type == 'sparse':
            sparsity_weight = kwargs.get('sparsity_weight', 0.1)
            target_sparsity = kwargs.get('target_sparsity', 0.05)
            self.model = autoencoder_class(
                input_dim, hidden_dim, latent_dim,
                sparsity_weight=sparsity_weight,
                target_sparsity=target_sparsity
            )
        elif autoencoder_type == 'denoising':
            noise_factor = kwargs.get('noise_factor', 0.3)
            self.model = autoencoder_class(input_dim, hidden_dim, latent_dim, noise_factor=noise_factor)
        elif autoencoder_type == 'vq_vae':
            num_embeddings = kwargs.get('num_embeddings', 512)
            commitment_cost = kwargs.get('commitment_cost', 0.25)
            self.model = autoencoder_class(
                input_dim, hidden_dim, latent_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost
            )
        else:
            self.model = autoencoder_class(input_dim, hidden_dim, latent_dim)
        
        self.model.to(self.device)
        self.is_trained = False
        
        print(f"Initialized {autoencoder_type} autoencoder on {self.device}")
    
    def train_autoencoder(
        self,
        train_data: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Train the autoencoder on the given data"""
        train_data = train_data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        train_losses = []
        
        for epoch in range(num_epochs):
            indices = torch.randperm(train_data.size(0))
            shuffled_data = train_data[indices]
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, train_data.size(0), batch_size):
                batch = shuffled_data[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Use compute_loss() method that exists in all autoencoders
                loss = self.model.compute_loss(batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
        
        self.is_trained = True
        self.model.eval()
        
        return {
            'final_loss': train_losses[-1],
            'train_losses': train_losses
        }
    
    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Encode embeddings to latent space"""
        embeddings = embeddings.to(self.device)
        with torch.no_grad():
            encoded = self.model.encode(embeddings)
        return encoded
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        latent = latent.to(self.device)
        with torch.no_grad():
            decoded = self.model.decode(latent)
        return decoded
    
    def compress_embeddings(
        self,
        embeddings: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress m embeddings to k embeddings
        
        Args:
            embeddings: Input embeddings [m, input_dim]
            k: Number of compressed embeddings
            
        Returns:
            Tuple of (compressed_embeddings [k, latent_dim], indices [k])
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before compression. Call train_autoencoder() first.")
        
        embeddings = embeddings.to(self.device)
        m = embeddings.shape[0]
        
        with torch.no_grad():
            # Encode all m embeddings to latent space
            latent = self.model.encode(embeddings)  # [m, latent_dim]
            
            # Select k representative embeddings using uniform sampling
            if k >= m:
                return latent, torch.arange(m)
            
            # Uniform sampling indices
            indices = torch.linspace(0, m-1, k).long()
            compressed = latent[indices]  # [k, latent_dim]
        
        return compressed, indices
    
    def get_codebook(self) -> Optional[torch.Tensor]:
        """Get the learned codebook (only for VQ-VAE)"""
        if self.autoencoder_type != 'vq_vae':
            return None
        
        if not self.is_trained:
            raise ValueError("Model must be trained first.")
        
        return self.model.vq_layer.embedding.weight.data
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'autoencoder_type': self.autoencoder_type,
            'is_trained': self.is_trained
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.autoencoder_type = checkpoint['autoencoder_type']
        self.is_trained = checkpoint['is_trained']
        self.model.eval()
        print(f"Model loaded from {path}")