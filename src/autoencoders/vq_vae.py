import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAutoEncoder

class VectorQuantizer(nn.Module):
    """
    Vector Quantization module that maps continuous embeddings to discrete codes
    from a learned codebook.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings  # k: size of the codebook
        self.embedding_dim = embedding_dim    # dimensionality of each embedding
        self.commitment_cost = commitment_cost
        
        # Initialize the embedding table (codebook)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, inputs):
        """
        Inputs: Continuous encodings from the encoder [batch_size, embedding_dim]
        Outputs:
            - quantized: The quantized version of the input
            - perplexity: Perplexity of the encodings
            - encodings: One-hot encodings of the quantized indices
            - encoding_indices: Indices of the quantized embeddings
            - commitment_loss: Commitment loss
        """
        # Calculate distances between inputs and embedding vectors
        distances = torch.sum(inputs ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight ** 2, dim=1) - \
                   2 * torch.matmul(inputs, self.embedding.weight.t())
        
        # Find closest embedding vectors
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Convert to one-hot encodings
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize the inputs
        quantized = torch.matmul(encodings, self.embedding.weight)
        
        # Calculate commitment loss: make encoder commit to its output
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        
        # Calculate codebook loss: make codebook vectors match encoder output
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        
        # Use straight-through estimator for gradients
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate perplexity (usage of codebook)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Total loss
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        return quantized, perplexity, encodings, encoding_indices, loss

class VQEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VQEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        latent = self.fc2(hidden)
        return latent

class VQDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VQDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, z):
        hidden = self.relu(self.fc1(z))
        output = self.fc2(hidden)
        return output

class VQVAELoss(nn.Module):
    def __init__(self):
        super(VQVAELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        
    def forward(self, x_reconstructed, x, vq_loss):
        # Reconstruction loss
        recon_loss = self.mse_loss(x_reconstructed, x)
        
        # Total loss: reconstruction loss + VQ loss
        return recon_loss + vq_loss

class VQVAE(BaseAutoEncoder):
    """
    Vector Quantized Variational Autoencoder implementation
    Converts continuous embeddings to k discrete codes from a codebook
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim, num_embeddings=64, commitment_cost=0.25): 
        """
        Args:
            input_dim: Dimensionality of input
            hidden_dim: Dimensionality of hidden layers
            latent_dim: Dimensionality of latent representation
            num_embeddings: Size of the codebook (k)
            commitment_cost: Weight for commitment loss
        """
        super(VQVAE, self).__init__(input_dim, hidden_dim, latent_dim)
        
        self.encoder = VQEncoder(input_dim, hidden_dim, latent_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = VQDecoder(latent_dim, hidden_dim, input_dim)
    
    def encode(self, x):
        z_e = self.encoder(x)
        z_q, perplexity, encodings, indices, vq_loss = self.vq_layer(z_e)
        return z_q, z_e, vq_loss, indices
    
    def decode(self, z_q):
        return self.decoder(z_q)
    
    def forward(self, x):
        z_q, z_e, vq_loss, indices = self.encode(x)
        reconstructed = self.decode(z_q)
        return reconstructed, z_q, vq_loss, indices
    
    def get_loss_function(self):
        return VQVAELoss()
    
    def compress(self, x):
        with torch.no_grad():
            z_e = self.encoder(x)
            _, _, _, indices, _ = self.vq_layer(z_e)
            return indices
    
    def get_codebook(self):
        """Return the embedding codebook"""
        return self.vq_layer.embedding.weight.detach()
    
    def reconstruct(self, x):
        """Reconstruct input (for evaluation)"""
        with torch.no_grad():
            z_q, _, _, _ = self.encode(x)
            return self.decode(z_q)
    
    def reconstruct_from_indices(self, indices):
        """Reconstruct from codebook indices"""
        with torch.no_grad():
            z_q = self.vq_layer.embedding(indices)
            return self.decode(z_q)
    
    def compress_n_to_k(self, embeddings, k=None):
        """
        Map n embeddings to k codebook vectors
        
        Args:
            embeddings: Input embeddings of shape [n, input_dim]
            k: Not used, since codebook size is fixed at initialization
            
        Returns:
            - codebook: The codebook embeddings of shape [k, input_dim]
            - indices: Mapping from original embeddings to codebook indices [n]
        """
        if k is not None and k != self.vq_layer.num_embeddings:
            print(f"Warning: k={k} specified, but VQ-VAE codebook size is fixed at {self.vq_layer.num_embeddings}")
        
        device = embeddings.device
        with torch.no_grad():
            # Encode embeddings
            z_e = self.encoder(embeddings)
            
            # Get codebook indices
            _, _, _, indices, _ = self.vq_layer(z_e)
            
            # Return codebook and indices
            codebook = self.get_codebook()
            return codebook, indices.to(device)

    def decompress_k_to_n(self, indices):
        """
        Map from codebook indices back to embedding space
        
        Args:
            indices: Codebook indices [n]
            
        Returns:
            Reconstructed embeddings of shape [n, input_dim]
        """
        return self.reconstruct_from_indices(indices)