import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from autoencoders.standard_ae import StandardAutoEncoder
from autoencoders.variational_ae import VariationalAutoEncoder
from autoencoders.beta_vae import BetaVAE
from autoencoders.sparse_ae import SparseAutoEncoder
from autoencoders.denoising_ae import DenoisingAutoEncoder
from autoencoders.vq_vae import VQVAE

# Configuration
input_dim = 768  # Typical for embeddings from models like BERT
hidden_dim = 512
latent_dim = 128
batch_size = 64
num_epochs = 100
k_embeddings = 32  # Number of embeddings for compression (k < n)
learning_rate = 0.001

# Generate some synthetic embedding data
def generate_embeddings(n_samples=1000, dim=768, n_clusters=10):
    """Generate synthetic embedding data with cluster structure"""
    np.random.seed(42)
    centers = np.random.randn(n_clusters, dim) * 2
    
    # Create samples around the centers
    samples_per_cluster = n_samples // n_clusters
    data = []
    labels = []
    
    for i, center in enumerate(centers):
        cluster_samples = center + np.random.randn(samples_per_cluster, dim) * 0.5
        data.append(cluster_samples)
        labels.extend([i] * samples_per_cluster)
    
    data = np.vstack(data)
    # Normalize embeddings (common for word embeddings)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    
    return torch.FloatTensor(data), torch.LongTensor(labels)

# Evaluate autoencoder model
def evaluate_autoencoder(model, data, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = data.to(device)
    
    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = model.get_loss_function()
    
    train_losses = []
    
    model.train()
    for epoch in range(num_epochs):
        # Shuffle data
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        
        total_loss = 0.0
        for i in range(0, data.size(0), batch_size):
            # Get mini-batch
            batch = shuffled_data[i:i+batch_size]
            
            # Forward pass
            if isinstance(model, VQVAE):
                reconstructed, _, vq_loss, _ = model(batch)
                loss = loss_fn(reconstructed, batch, vq_loss)
            elif isinstance(model, VariationalAutoEncoder) or isinstance(model, BetaVAE):
                reconstructed, _, mu, logvar = model(batch)
                loss = loss_fn(reconstructed, batch, mu, logvar)
            elif isinstance(model, SparseAutoEncoder):
                reconstructed, _, activations = model(batch)
                loss = loss_fn(reconstructed, batch, activations)
            else:
                reconstructed, _ = model(batch)
                loss = loss_fn(reconstructed, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (data.size(0) // batch_size)
        train_losses.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f'{name} - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
    
    # Evaluate compression and reconstruction quality
    model.eval()
    with torch.no_grad():
        # Reconstruct data
        if isinstance(model, VQVAE):
            reconstructed = model.reconstruct(data)
        else:
            reconstructed = model.reconstruct(data)
        
        # Calculate reconstruction error
        mse = torch.mean((data - reconstructed) ** 2).item()
        
        # Compress n to k embeddings
        compressed_emb, indices = model.compress_n_to_k(data, k=k_embeddings)
        
        # Decompress k to n embeddings
        decompressed_emb = model.decompress_k_to_n(indices)
        
        # Calculate compression error
        compression_mse = torch.mean((data - decompressed_emb) ** 2).item()
        
        # Get the codebook
        codebook = model.get_codebook()
        
        print(f"\n{name} Results:")
        print(f"Original data shape: {data.shape}")
        print(f"Reconstructed data shape: {reconstructed.shape}")
        print(f"Reconstruction MSE: {mse:.6f}")
        print(f"Compressed representation shape: {compressed_emb.shape}")
        print(f"Codebook shape: {codebook.shape}")
        print(f"Compression MSE (n to k to n): {compression_mse:.6f}")
        
        return {
            "name": name,
            "model": model,
            "train_losses": train_losses,
            "reconstruction_mse": mse,
            "compression_mse": compression_mse,
            "indices": indices.cpu(),
            "codebook": codebook.cpu(),
            "data": data.cpu(),
            "reconstructed": reconstructed.cpu(),
            "decompressed": decompressed_emb.cpu()
        }

def visualize_results(results):
    # Plot training losses
    plt.figure(figsize=(12, 6))
    for result in results:
        plt.plot(result["train_losses"], label=result["name"])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('autoencoder_training_loss.png')
    plt.close()
    
    # Plot reconstruction and compression MSE
    names = [result["name"] for result in results]
    recon_mse = [result["reconstruction_mse"] for result in results]
    comp_mse = [result["compression_mse"] for result in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, recon_mse, width, label='Reconstruction MSE')
    plt.bar(x + width/2, comp_mse, width, label='Compression MSE')
    plt.xticks(x, names, rotation=45)
    plt.ylabel('MSE')
    plt.title('Reconstruction and Compression Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig('autoencoder_mse_comparison.png')
    plt.close()
    
    # Visualize original vs compressed embeddings using t-SNE (for first result)
    for result in results:
        # Sample a subset for t-SNE (for better visualization)
        sample_size = min(500, result["data"].shape[0])
        sample_indices = np.random.choice(result["data"].shape[0], sample_size, replace=False)
        
        original_data = result["data"][sample_indices].numpy()
        reconstructed_data = result["reconstructed"][sample_indices].numpy()
        decompressed_data = result["decompressed"][sample_indices].numpy()
        
        # t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        original_2d = tsne.fit_transform(original_data)
        reconstructed_2d = tsne.fit_transform(reconstructed_data)
        decompressed_2d = tsne.fit_transform(decompressed_data)
        
        # Plot the results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.scatter(original_2d[:, 0], original_2d[:, 1], alpha=0.5)
        plt.title(f'{result["name"]} - Original Embeddings')
        
        plt.subplot(1, 3, 2)
        plt.scatter(reconstructed_2d[:, 0], reconstructed_2d[:, 1], alpha=0.5)
        plt.title(f'{result["name"]} - Reconstructed Embeddings')
        
        plt.subplot(1, 3, 3)
        plt.scatter(decompressed_2d[:, 0], decompressed_2d[:, 1], alpha=0.5)
        plt.title(f'{result["name"]} - Compressed (k={k_embeddings}) Embeddings')
        
        plt.tight_layout()
        plt.savefig(f'{result["name"]}_embedding_visualization.png')
        plt.close()

def main():
    # Generate synthetic embedding data
    print("Generating synthetic embeddings...")
    data, labels = generate_embeddings(n_samples=1000)
    print(f"Generated data shape: {data.shape}")
    
    # Create autoencoder models
    models = [
        ("StandardAE", StandardAutoEncoder(input_dim, hidden_dim, latent_dim)),
        ("VariationalAE", VariationalAutoEncoder(input_dim, hidden_dim, latent_dim)),
        ("BetaVAE", BetaVAE(input_dim, hidden_dim, latent_dim, beta=4.0)),
        ("SparseAE", SparseAutoEncoder(input_dim, hidden_dim, latent_dim)),
        ("DenoisingAE", DenoisingAutoEncoder(input_dim, hidden_dim, latent_dim, noise_factor=0.3)),
        ("VQVAE", VQVAE(input_dim, hidden_dim, latent_dim, num_embeddings=k_embeddings))
    ]
    
    results = []
    
    for name, model in models:
        print(f"\nEvaluating {name}...")
        result = evaluate_autoencoder(model, data, name)
        results.append(result)
    
    # Visualize and compare results
    visualize_results(results)
    
    # Compare embedding compression
    print("\nCompression Comparison (n->k->n) MSE:")
    for result in results:
        print(f"{result['name']}: {result['compression_mse']:.6f}")
    
    # Save models (optional)
    for result in results:
        torch.save(result["model"].state_dict(), f"{result['name']}_model.pth")

if __name__ == "__main__":
    main()