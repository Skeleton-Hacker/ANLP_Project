1. First preprocess the data (multi-GPU)
```bash
accelerate launch semantic_chunking.py
```

2. For the normal embedding (multi-GPU)
```bash
accelerate launch  chunk_embeddings.py
```


3.For using the code
```bash
accelerate launch <filename>.py
```


- Use `--compare-base` flag to run the test set and get the evaluation metrics in it.

- *_full_finetune means encoder and decoder is trainable. Config class freeze_decoder = True means decoder is Frozen while False means it is trainable
- *_untrained means no pretrained weights
- semantic_chunking_stopwords.py is the semantic chunking version that utilises the stopwords, lemmatisation,etc.
- semantic_chunking_sweep_table.py is to find the optimal threshold value for the chunk size
