# Architecture — Week 1 Intent Router

## High-level block view
```
[Raw utterance text]
        |
        v
[Tokenizer + truncation/padding]
        |
        v
[input_ids, attention_mask]
        |
        v
[Embedding matrix E in R^(V x d)]
        |
        v
[BiGRU contextual encoder]
        |
        v
[Masked mean pooling]
        |
        v
[Linear classifier]
        |
        v
[Softmax distribution over intents]
```

## Tensor dimensions
- `input_ids`: [B, T]
- `embeddings`: [B, T, d]
- `biGRU outputs`: [B, T, 2h]
- `pooled sentence`: [B, 2h]
- `logits`: [B, C]

## Why this architecture for Week 1
- Simple enough to derive gradients clearly.
- Stronger than bag-of-words baselines due to sequence context from BiGRU.
- Serves as conceptual bridge to Transformer encoders in Week 2.
