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

The build follows a set of study notes written alongside Karpathy's lecture
(*[從零打造 GPT](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e)*,
in Traditional Chinese). Each row is one commit: read the section, implement it,
make the tests pass.

### Foundations

| # | Component | Notes |
| --- | --- | --- |
| 1 | Corpus | [§1](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch2) ✅ |
| 2 | Character-level tokenizer | [§2](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch3) |
| 3 | Train/validation split | [§3](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch4) |
| 4 | Context windows (`block_size`) | [§4](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch5) |
| 5 | Batching | [§5](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch6) |

### Baseline model

| # | Component | Notes |
| --- | --- | --- |
| 6 | Bigram language model | [§6](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch7) |
| 7 | Cross-entropy loss | [§7](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch8) |
| 8 | Autoregressive sampling | [§8](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch9) |
| 9 | Training loop, AdamW, loss estimation | [§9](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch10) |

### Self-attention

| # | Component | Notes |
| --- | --- | --- |
| 10 | Causal averaging, written as a loop | [§10](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch11) |
| 11 | The same average as a matrix product | [§11](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch12) |
| 12 | Batched matrix form | [§12](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch13) |
| 13 | Masked softmax | [§13](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch14) |
| 14 | Token and positional embeddings | [§14](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch15) |
| 15 | Single head: query, key, value | [§15](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch16) |
| 16 | Scaling by the square root of head size | [§17](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch18) |
| 17 | One head wired into the network | [§18](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch19) |
| 18 | Multi-head attention | [§19](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch20) |

### Transformer block

| # | Component | Notes |
| --- | --- | --- |
| 19 | Position-wise feed-forward network | [§20](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch21) |
| 20 | Residual connections and projections | [§21](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch22) |
| 21 | Layer normalisation | [§22](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch23) |
| 22 | Dropout and the scaled-up configuration | [§23](https://claude.ai/code/artifact/c0bb22bc-324e-458d-90d8-c542033da67e#ch24) |

Reference implementation: [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture).

---

## Repository layout

```
nanoGPT/
├── data/
│   └── tinyshakespeare/
│       ├── input.txt      # raw corpus
│       └── README.md      # provenance, checksum, corpus statistics
├── notebooks/             # exploration only; see notebooks/README.md
├── tests/                 # acceptance criteria, written before each component
├── LICENSE
└── README.md
```

Implementation modules live at the repository root and are added as the roadmap
progresses. Notebooks are for working things out; anything that needs to be
imported, tested or reviewed is moved into a module.

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
