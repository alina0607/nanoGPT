# nanoGPT

Building a decoder-only transformer from scratch, one step at a time.

This repository is a from-first-principles implementation of a small GPT, written
as a learning exercise rather than as a wrapper around an existing library. Every
component — tokenizer, batching, self-attention, the training loop — is
implemented by hand and committed as a separate, reviewable step.

**Current status:** the dataset is in place; the implementation is being written
step by step. The roadmap below tracks progress.

---

## Roadmap

Each step is a self-contained commit with its own explanation.

### Data
- [x] Character-level dataset (tiny Shakespeare)
- [ ] Vocabulary and character-level tokenizer (`encode` / `decode`)
- [ ] Train/validation split
- [ ] Context windows (`block_size`) and batching

### Baseline
- [ ] Bigram language model
- [ ] Cross-entropy loss
- [ ] Sampling loop (`generate`)
- [ ] Training loop with periodic validation loss

### Attention
- [ ] Causal averaging via a lower-triangular matrix
- [ ] Masked softmax attention
- [ ] Token and positional embeddings
- [ ] Single-head self-attention (query / key / value)
- [ ] Scaled dot-product attention
- [ ] Multi-head attention

### Transformer block
- [ ] Position-wise feed-forward network
- [ ] Residual connections
- [ ] Layer normalisation (pre-norm)
- [ ] Dropout
- [ ] Stacked blocks and a scaled-up configuration

---

## Repository layout

```
nanoGPT/
├── data/
│   └── tinyshakespeare/
│       ├── input.txt      # raw corpus
│       └── README.md      # provenance, checksum, corpus statistics
├── LICENSE
└── README.md
```

Implementation modules are added as the roadmap progresses.

---

## Dataset

The starting corpus is **tiny Shakespeare**: a ~1.1 MB plain-text concatenation of
Shakespeare's plays, the standard benchmark for character-level language models.
Shakespeare's works are in the public domain.

| Property | Value |
| --- | --- |
| Size | 1,115,394 bytes |
| Lines | 40,000 |
| Distinct characters | 65 |
| SHA-256 | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |

See [`data/tinyshakespeare/README.md`](data/tinyshakespeare/README.md) for the
source URL and a verification command.

### Using a different corpus

The dataset directory is deliberately namespaced so corpora can be swapped without
touching the model code. To add one:

1. Create `data/<name>/` and place the raw text at `data/<name>/input.txt`.
2. Add a `README.md` recording the source, licence, size and checksum.
3. Point the training configuration at the new directory.

Any UTF-8 plain-text corpus works. Character-level modelling makes no assumption
about language or alphabet, though a larger character set increases the vocabulary
and therefore the size of the embedding and output layers.

---

## Prerequisites

- Python 3.10+
- PyTorch (added to `requirements.txt` once the first model step lands)

Apple Silicon users can train on the `mps` backend; the corpus is small enough
that CPU-only training is also viable.

---

## References

- Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out.*
  ([video](https://www.youtube.com/watch?v=kCc8FmEb1nY) ·
  [code](https://github.com/karpathy/ng-video-lecture))
- Vaswani et al. (2017), *Attention Is All You Need.* ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762))
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners.* (GPT-2)

---

## License

[MIT](LICENSE). The tiny Shakespeare corpus is public-domain text and is not
covered by this licence.
