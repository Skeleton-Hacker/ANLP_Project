0.  Download bitsandbytes for quantised model for llama
```bash
pip install -U bitsandbytes
```


1. First preprocess the data (multi-GPU)
```bash
accelerate launch --config_file accelerate_config.yaml semantic_chunking.py
```

2. For the normal embedding (multi-GPU)
```bash
accelerate launch --config_file accelerate_config.yaml chunk_embeddings.py
```

3. For using the llama adapter (multi-GPU)
```bash
accelerate launch --config_file accelerate_config.yaml llama_adapter.py
```

4.For using the t5 adapter (multi-GPU)
```bash
accelerate launch --config_file accelerate_config.yaml t5_adapter.py
```


Mention about the BERT score similarity between llama adapter and llama base.