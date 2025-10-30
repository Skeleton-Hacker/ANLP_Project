0.  Download bitsandbytes for quantised model for llama
```bash
pip install -U bitsandbytes
```


1. First preprocess the data
```bash
accelerate semantic_chunking.py
```

2. For the normal embedding, use
```bash
accelerate launch chunk_embeddings.py
```

3. For using the llama adapter,
```bash
accelerate launch llama_adapter.py
```

4.For using the t5 adapter
```bash
accelerate launch t5_adapter.py
```


Mention about the BERT score similarity between llama adapter and llama base.