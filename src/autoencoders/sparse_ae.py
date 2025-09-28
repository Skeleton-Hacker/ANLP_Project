import torch
import torch.nn as nn
from .standard_ae import StandardAutoEncoder, Encoder, Decoder
from sklearn.cluster import KMeans

class SparsityLoss(nn.Module):
    def __init__(self, sparsity_weight=0.1, target_sparsity=0.05):
        super(SparsityLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.sparsity_weight = sparsity_weight
        self.target_sparsity = target_sparsity
        
    def kl_divergence(self, rho, rho_hat):
        """KL divergence to enforce sparsity"""
        return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    
    def forward(self, x_reconstructed, x, activations):
        # Reconstruction loss
        recon_loss = self.mse_loss(x_reconstructed, x)
        
        # Average activation of hidden units
        rho_hat = torch.mean(activations, dim=0)
        
        # Sparsity penalty
        sparsity_penalty = torch.sum(self.kl_divergence(self.target_sparsity, rho_hat))
        
        # Total loss
        return recon_loss + self.sparsity_weight * sparsity_penalty

class SparseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SparseEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        latent = self.fc2(hidden)
        activations = torch.sigmoid(latent)  # Bounded activation for sparsity
        return activations, activations  # Return activations for sparsity loss calculation

class SparseAutoEncoder(StandardAutoEncoder):
    """Sparse Autoencoder implementation"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, sparsity_weight=0.1, target_sparsity=0.05):
        super(SparseAutoEncoder, self).__init__(input_dim, hidden_dim, latent_dim)
        
        self.sparsity_weight = sparsity_weight
        self.target_sparsity = target_sparsity
        
        # Replace standard encoder with sparse encoder
        self.encoder = SparseEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
        self.kmeans = None
        self.codebook = None
    
    def encode(self, x):
        latent, activations = self.encoder(x)
        return latent, activations
    
    def forward(self, x):
        latent, activations = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent, activations
    
    def get_loss_function(self):
        return SparsityLoss(self.sparsity_weight, self.target_sparsity)
    
    def compress(self, x):
        with torch.no_grad():
            latent, _ = self.encode(x)
            return latent
    
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
            # Get latent representations (ignoring activations)
            latent_vectors, _ = self.encode(embeddings)
            
            # Move to CPU for k-means clustering
            latent_cpu = latent_vectors.detach().cpu().numpy()
            
            # Cluster in latent space to get k clusters
            kmeans = KMeans(n_clusters=k, random_state=0)
            cluster_indices = kmeans.fit_predict(latent_cpu)
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