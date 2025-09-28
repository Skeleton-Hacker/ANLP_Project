import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAutoEncoder
from sklearn.cluster import KMeans

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, z):
        hidden = self.relu(self.fc1(z))
        output = self.fc2(hidden)
        return output

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        
    def forward(self, x_reconstructed, x, mu, logvar):
        # Reconstruction loss
        recon_loss = self.mse_loss(x_reconstructed, x)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size
        
        # Total loss
        return recon_loss + kl_loss

class VariationalAutoEncoder(BaseAutoEncoder):
    """Variational Autoencoder implementation"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalAutoEncoder, self).__init__(input_dim, hidden_dim, latent_dim)
        
        self.encoder = VariationalEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.kmeans = None
        self.codebook = None
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z, mu, logvar
    
    def get_loss_function(self):
        return VAELoss()
    
    def compress(self, x):
        with torch.no_grad():
            mu, _ = self.encoder(x)
            return mu  # Use mean of latent distribution as compressed representation
    
    def reconstruct(self, x):
        with torch.no_grad():
            z, _, _ = self.encode(x)
            return self.decode(z)
    
    def compress_n_to_k(self, embeddings, k=64):
        """
        Compress n embeddings to k embeddings using clustering in latent space
        
        Args:
            embeddings: Input embeddings of shape [n, input_dim]
            k: Number of embeddings to compress to (default: 64)
            
        Returns:
            - k_embeddings: The k embeddings (centroids) of shape [k, input_dim]
            - indices: Mapping from original embeddings to centroid indices [n]
        """
        device = embeddings.device
        
        with torch.no_grad():
            # Use mean vectors from encoder as deterministic embeddings
            mu, _ = self.encoder(embeddings)
            
            # Move to CPU for k-means clustering
            mu_cpu = mu.detach().cpu().numpy()
            
            # Cluster latent means to get k clusters
            kmeans = KMeans(n_clusters=k, random_state=0)
            cluster_indices = kmeans.fit_predict(mu_cpu)
            centroids = kmeans.cluster_centers_
            
            # Store KMeans model for future use
            self.kmeans = kmeans
            
            # Convert centroids back to tensor and move to same device
            centroids_tensor = torch.FloatTensor(centroids).to(device)
            
            # Decode centroids to get k embeddings
            k_embeddings = self.decode(centroids_tensor)
            
            # Store codebook on the device
            self.codebook = k_embeddings
            
            # Return indices on the same device as input
            return k_embeddings, torch.LongTensor(cluster_indices).to(device)
    
    def decompress_k_to_n(self, indices):
        """
        Decompress k-dimensional representation back to n original embeddings
        
        Args:
            indices: Indices of the k embeddings to map back to original space [n]
            
        Returns:
            Reconstructed embeddings of shape [n, input_dim]
        """
        if self.codebook is None:
            raise ValueError("Must call compress_n_to_k first to create codebook")
        
        with torch.no_grad():
            indices = indices.to(self.codebook.device)
            return self.codebook[indices]
    
    def get_codebook(self):
        """
        Return the current codebook of k embeddings
        
        Returns:
            Codebook of shape [k, input_dim]
        """
        if self.codebook is None:
            raise ValueError("Must call compress_n_to_k first to create codebook")
        return self.codebook