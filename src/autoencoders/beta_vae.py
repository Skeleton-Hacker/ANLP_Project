import torch
import torch.nn as nn
from .variational_ae import VariationalAutoEncoder, VariationalEncoder, Decoder
from sklearn.cluster import KMeans

class BetaVAELoss(nn.Module):
    def __init__(self, beta=4.0):
        super(BetaVAELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.beta = beta
        
    def forward(self, x_reconstructed, x, mu, logvar):
        # Reconstruction loss
        recon_loss = self.mse_loss(x_reconstructed, x)
        
        # KL divergence with beta weighting
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size
        
        # Total loss
        return recon_loss + self.beta * kl_loss

class BetaVAE(VariationalAutoEncoder):
    """Beta-VAE implementation with stronger KL divergence regularization"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, beta=4.0):
        super(BetaVAE, self).__init__(input_dim, hidden_dim, latent_dim)
        self.beta = beta
        self.kmeans = None
        self.codebook = None
    
    def get_loss_function(self):
        return BetaVAELoss(beta=self.beta)
    
    # Inherit compress_n_to_k, decompress_k_to_n, and get_codebook from VariationalAutoEncoder,
    # but let's implement them here explicitly for clarity
    
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