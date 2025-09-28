import torch
import torch.nn as nn
from .base import BaseAutoEncoder
from sklearn.cluster import KMeans

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        latent = self.fc2(hidden)
        return latent

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        output = self.fc2(hidden)
        return output

class StandardAutoEncoder(BaseAutoEncoder):
    """Standard/Vanilla Autoencoder implementation"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(StandardAutoEncoder, self).__init__(input_dim, hidden_dim, latent_dim)
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.kmeans = None
        self.codebook = None
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
    
    def get_loss_function(self):
        return nn.MSELoss()
    
    def compress(self, x):
        with torch.no_grad():
            return self.encode(x)
    
    def reconstruct(self, x):
        with torch.no_grad():
            latent = self.encode(x)
            return self.decode(latent)
            
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
        device = embeddings.device  # Store the device of input embeddings
        
        with torch.no_grad():
            # Encode all embeddings to latent space
            latent_vectors = self.encode(embeddings)
            
            # Move to CPU for k-means clustering (sklearn doesn't support GPU)
            latent_cpu = latent_vectors.detach().cpu().numpy()
            
            # Cluster in latent space to get k clusters
            kmeans = KMeans(n_clusters=k, random_state=0)
            cluster_indices = kmeans.fit_predict(latent_cpu)
            centroids = kmeans.cluster_centers_
            
            # Store KMeans model for future use
            self.kmeans = kmeans
            
            # Convert centroids back to tensor and move to same device as input
            centroids_tensor = torch.FloatTensor(centroids).to(device)
            
            # Decode centroids to get k embeddings
            k_embeddings = self.decode(centroids_tensor)
            
            # Store codebook on the correct device
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
            # Ensure indices are on the same device as codebook
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