from .base import BaseAutoEncoder
from .standard_ae import StandardAutoEncoder
from .denoising_ae import DenoisingAutoEncoder
from .sparse_ae import SparseAutoEncoder
from .variational_ae import VariationalAutoEncoder
from .beta_vae import BetaVAE
from .vq_vae import VQVAE  # Add the new VQ-VAE

__all__ = [
    'BaseAutoEncoder',
    'StandardAutoEncoder',
    'VariationalAutoEncoder',
    'BetaVAE',
    'DenoisingAutoEncoder',
    'SparseAutoEncoder',
    'VQVAE'  # Include VQ-VAE in the module's public API
]

def get_autoencoder(ae_type, input_dim, hidden_dim, latent_dim, **kwargs):
    """
    Factory function to get the appropriate autoencoder model.
    
    Args:
        ae_type (str): Type of autoencoder ('ae', 'vae', 'bvae', 'dae', 'sae')
        input_dim (int): Dimension of input
        hidden_dim (int): Dimension of hidden layer
        latent_dim (int): Dimension of latent space
        **kwargs: Additional arguments for specific autoencoder types
        
    Returns:
        An autoencoder model
    """
    if ae_type == 'ae':
        return StandardAutoEncoder(input_dim, hidden_dim, latent_dim)
    elif ae_type == 'vae':
        return VariationalAutoEncoder(input_dim, hidden_dim, latent_dim)
    elif ae_type == 'bvae':
        beta = kwargs.get('beta', 4.0)
        return BetaVAE(input_dim, hidden_dim, latent_dim, beta=beta)
    elif ae_type == 'dae':
        noise_factor = kwargs.get('noise_factor', 0.3)
        return DenoisingAutoEncoder(input_dim, hidden_dim, latent_dim, noise_factor=noise_factor)
    elif ae_type == 'sae':
        sparsity_weight = kwargs.get('sparsity_weight', 0.1)
        target_sparsity = kwargs.get('target_sparsity', 0.05)
        return SparseAutoEncoder(input_dim, hidden_dim, latent_dim, 
                                sparsity_weight=sparsity_weight,
                                target_sparsity=target_sparsity)
    elif ae_type == 'vq_vae':
        return VQVAE(input_dim, hidden_dim, latent_dim)
    else:
        raise ValueError(f"Unknown autoencoder type: {ae_type}")