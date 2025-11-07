
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
- The files are appropriatley named, run the files accordingly 