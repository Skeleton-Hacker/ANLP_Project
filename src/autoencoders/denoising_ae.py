import torch
import torch.nn as nn
from .standard_ae import StandardAutoEncoder, Encoder, Decoder
from sklearn.cluster import KMeans

class DenoisingAutoEncoder(StandardAutoEncoder):
    """Denoising Autoencoder implementation"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, noise_factor=0.3):
        super(DenoisingAutoEncoder, self).__init__(input_dim, hidden_dim, latent_dim)
        self.noise_factor = noise_factor
        self.kmeans = None
        self.codebook = None
        
    def add_noise(self, x):
        """Add Gaussian noise to the input"""
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise
    
    def forward(self, x):
        # Add noise to the input
        x_noisy = self.add_noise(x)
        
        # Encode noisy input
        latent = self.encode(x_noisy)
        
        # Decode to reconstruct clean input
        reconstructed = self.decode(latent)
        
        return reconstructed, latent
    
    def get_loss_function(self):
        return nn.MSELoss()  # Reconstruction loss against clean input
    
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
            # For denoising, we use clean inputs for compression
            # Encode clean inputs to latent space
            latent_vectors = self.encode(embeddings)
            
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