import argparse
from pipeline import CompressionPipeline

def main():
    parser = argparse.ArgumentParser(description='Run semantic compression pipeline')
    parser.add_argument('--autoencoder', type=str, default='standard',
                       choices=['standard', 'variational', 'beta_vae', 'sparse', 'denoising', 'vq_vae'],
                       help='Type of autoencoder to use')
    parser.add_argument('--k', type=int, default=8,
                       help='Number of compressed embeddings per chunk')
    parser.add_argument('--max-docs', type=int, default=10,
                       help='Maximum number of documents to process')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--chunk-size', type=int, default=512,
                       help='Target chunk size in tokens')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden dimension for autoencoder')
    parser.add_argument('--latent-dim', type=int, default=256,
                       help='Latent dimension for autoencoder')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CompressionPipeline(
        autoencoder_type=args.autoencoder,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        max_documents=args.max_docs,
        k=args.k,
        num_epochs=args.epochs,
        chunk_size=args.chunk_size
    )
    
    print("\n" + "=" * 80)
    print("Pipeline Execution Complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()