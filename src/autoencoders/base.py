import torch
import torch.nn as nn

class BaseAutoEncoder(nn.Module):
    """Base class for all autoencoder implementations"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(BaseAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Will be implemented by subclasses
        self.encoder = None
        self.decoder = None
        
    def encode(self, x):
        """Encode input to latent representation"""
        raise NotImplementedError("Subclasses must implement encode method")
    
    def decode(self, z):
        """Decode latent representation to reconstructed input"""
        raise NotImplementedError("Subclasses must implement decode method")
    
    def forward(self, x):
        """Forward pass through the autoencoder"""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_loss_function(self):
        """Return the appropriate loss function for this autoencoder"""
        raise NotImplementedError("Subclasses must implement get_loss_function method")
    
    def compress(self, x):
        """Compress input to latent representation (for inference)"""
        with torch.no_grad():
            return self.encode(x)
    
    def reconstruct(self, x):
        """Reconstruct input (for evaluation)"""
        with torch.no_grad():
            latent = self.encode(x)
            return self.decode(latent)