# Transformers (Autoregressive for Text)

## Overview
Transformers, introduced by Vaswani et al. in 2017, use self-attention for sequence modeling, excelling in text generation. Autoregressive transformers (e.g., GPT) predict the next token given previous ones, enabling coherent text. They scale to large models but require significant compute and data.

**Key Strengths**:
- Excellent for text and code generation.
- Supports zero-shot learning.
- Scales with large datasets.

**Key Weaknesses**:
- High computational cost.
- Risk of hallucinations.
- Slow for long sequences.

**Seminal Papers**:
- Vaswani et al., 2017, "Attention is All You Need" [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Radford et al., 2019, "Language Models are Unsupervised Multitask Learners" [OpenAI Blog](https://openai.com/blog/better-language-models/)
- Brown et al., 2020, "Language Models are Few-Shot Learners" [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

## Theoretical Background

Transformers represent a paradigm shift in sequence modeling, moving away from recurrent architectures to fully attention-based models. As Vaswani et al. (2017) noted, "The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution" [1].

### Self-Attention Mechanism

The core innovation of transformers is the self-attention mechanism, which allows the model to weigh the importance of different tokens in a sequence when encoding each position.

The self-attention operation can be formulated as:

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

where $Q$, $K$, and $V$ are query, key, and value matrices derived from the input, and $d_k$ is the dimension of the keys.

As Vaswani et al. explain, "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors" [1].

### Multi-Head Attention

Transformers use multi-head attention, which allows the model to jointly attend to information from different representation subspaces:

$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$

where each head is computed as:

$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

According to Vaswani et al., "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions" [1].

### Positional Encoding

Since transformers lack recurrence or convolution, they need positional encodings to capture sequence order. Vaswani et al. used sinusoidal functions:

$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})$
$PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d_{model}})$

As the authors explain, "These functions have the property that for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$" [1].

### Decoder-Only Architecture for Text Generation

Autoregressive transformers for text generation, like GPT, use a decoder-only architecture where each token can only attend to previous tokens. As Radford et al. (2019) note, "The use of a transformer decoder allows the model to generate coherent and contextually relevant text by conditioning each prediction on all previous tokens" [2].

### Scaling Laws

A key insight about transformer language models is that their performance scales predictably with model size, dataset size, and compute. According to Kaplan et al. (2020), "Performance improves smoothly as we increase the model size, dataset size, and amount of computation, following a power-law relationship" [3].

Brown et al. (2020) demonstrated with GPT-3 that "language models with sufficiently many parameters can perform well on a wide range of tasks without task-specific fine-tuning, simply by prompting the model with a few examples or instructions in natural language" [4].

## Architecture
![Transformer Architecture](../images/transformer_architecture.png)
[Download Transformer Architecture](../images/transformer_architecture.png)

## Algorithm
```pseudocode
Algorithm: Train_Transformer
Input: Text dataset T, vocabulary V, learning rate lr, epochs E
Output: Trained transformer model

1. Initialize transformer model with random weights
2. Tokenize T into sequences of tokens
3. For each epoch in E:
    a. For each batch of sequences in T:
        i. Compute output logits for next token prediction
        ii. Compute cross-entropy loss: L = CE(logits, target_tokens)
        iii. Update model weights to minimize L
4. Return transformer
```

## Training Techniques

Several techniques have been developed to improve transformer training:

1. **Masked Language Modeling**: As introduced by BERT (Devlin et al., 2019), "The model is trained to predict randomly masked tokens in the input, allowing bidirectional context conditioning" [5].

2. **Causal Language Modeling**: Used in GPT models, "The model is trained to predict the next token given all previous tokens in a sequence" [2].

3. **Rotary Position Embeddings (RoPE)**: Su et al. (2021) proposed, "A relative positional encoding method that encodes the relative position directly in the transformer's attention computation" [6].

4. **Flash Attention**: Dao et al. (2022) introduced "An IO-aware exact attention algorithm that uses tiling to reduce memory reads/writes between GPU high bandwidth memory (HBM) and on-chip SRAM" [7].

## Applications

Transformers have revolutionized natural language processing and beyond:

- **Text Generation**: GPT-3 and GPT-4 can generate human-like text for various applications.
- **Machine Translation**: Models like T5 (Raffel et al., 2020) excel at translating between languages [8].
- **Code Generation**: GitHub Copilot uses transformers to generate code from natural language descriptions.
- **Multimodal Understanding**: Models like CLIP (Radford et al., 2021) connect text and images [9].
- **Protein Structure Prediction**: AlphaFold 2 (Jumper et al., 2021) uses transformer-like architectures to predict protein structures [10].

## Code
[Download Transformer Code](../code/transformer_text.py)

This implementation generates token sequences. Run it with:
```bash
python code/transformer_text.py
```

## Additional References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). "Attention is All You Need" [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

[2] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). "Language Models are Unsupervised Multitask Learners" [OpenAI Blog](https://openai.com/blog/better-language-models/)

[3] Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., ... & Amodei, D. (2020). "Scaling Laws for Neural Language Models" [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

[4] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). "Language Models are Few-Shot Learners" [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

[5] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

[6] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding" [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

[7] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

[8] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)

[9] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). "Learning Transferable Visual Models From Natural Language Supervision" [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

[10] Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., ... & Hassabis, D. (2021). "Highly accurate protein structure prediction with AlphaFold" [Nature 596, 583–589](https://www.nature.com/articles/s41586-021-03819-2)