# RobertaConfig (Original)

## Considered in optimization
- Tokenizer {1: "BPE", 2: "WordPiece", 3: "Unigram", 4: "Word"} (Tokenizer param)
- Vocab Size: 1000-46000
- Num Hidden Layers: 1-12 Default 12
- Hidden Size: 16-256 Default 768
- Hidden Act: {1: "gelu", 2: "relu", 3: "silu", 4: "gelu_new"}
- Hidden Dropout Prob: 0.2-0.5
- Intermediate Size: 32-3072
- Num Attention Heads: 1-12 Default 12
- Attention Probs Dropout Prob: 0.2-0.5
- Max Sequence Length: 256-512 - max_position_embeddings= Max Sequence Length + 2
- Position Embedding Type: {1: "absolute", 2: "relative_key", 3: "relative_key_query"}
- Learning Rate: {1: 1e-3, 2: 1e-4, 3: 5e-5} (Training param)
- Batch Size: {1: 8, 2: 16} (Training param)

## Others from docs (default)
- type_vocab_size
- initializer_range
- layer_norm_eps
- is_decoder
- use_cache
- classifier_dropout

# T5Config
## Considered in optimization
- num_layers: Hidden layers Default 6
- hidden_act: Hidden activation
- num_decoder_layers: Hidden decoder layers, default to num_layers
- d_model: Hidden size Default 512
- num_heads: Attention heads Default 8
- d_kv: Size of the key, query, value projections per attention head Default 64
- d_ff: Size of the intermediate feed forward layer Default 2048
- relative_attention_num_buckets: The number of buckets to use for each attention layer Default 32
- relative_attention_max_distance: The maximum distance of the longer sequences for the bucket separation. Default 128
- dropout_rate: Default 0.1
- feed_forward_proj: relu or gated-gelu Default relu
- Learning rate
- Batch size

Default settings will already give something smaller than t5p-220m

We cannot change vocab size and tokenizer for generation