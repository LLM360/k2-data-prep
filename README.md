# Data Preparation Code for LLM360 K2-65B

This repository contains the data preparation code for K2-65B, a 65 billion parameter
large language model from LLM360.

> [!NOTE]
> This repository is under active development. If you have suggestions or find bugs, please open a GitHub issue or reach out.

### Prepare Raw Data Chunks
```
python prepare_jsonl_chunks.py
python prepare_pile_of_law_chunks.py
python prepare_redpajama_chunks.py
python prepare_starcoder_chunks.py
```

### Tokenize
```
python tokenize_datasets.py
```

### Create FIM Data for StarCoder
```
python starcoder_fim_main.py --spm_rate 0.
python starcoder_fim_main.py --spm_rate 1.
```

### Gather into 360 chunks
```
python gather.py
```

### Shuffle
```
python shuffle.py
```

### Print Data Mix
```
python analyze.py
```
