# tiny Shakespeare

A ~1.1 MB plain-text concatenation of Shakespeare's plays, widely used as a
benchmark corpus for character-level language models. Shakespeare's works are in
the public domain.

## Source

```
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Retrieved 2026-09-15.

## Integrity

| Property | Value |
| --- | --- |
| Size | 1,115,394 bytes |
| SHA-256 | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |

Verify:

```bash
shasum -a 256 data/tinyshakespeare/input.txt
```

Re-download:

```bash
curl -sSL --create-dirs \
  -o data/tinyshakespeare/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Corpus statistics

| Property | Value |
| --- | --- |
| Characters | 1,115,394 |
| Lines | 40,000 |
| Distinct characters (vocabulary size) | 65 |
| Space characters | 169,892 |
| Digits present | `3` only, 27 occurrences |

The vocabulary breaks down as follows, and the file is pure ASCII — its byte
count and character count are identical.

| Group | Count | Characters |
| --- | --- | --- |
| Whitespace | 2 | newline, space |
| Punctuation and symbols | 10 | `!` `$` `&` `'` `,` `-` `.` `:` `;` `?` |
| Digits | 1 | `3` |
| Upper-case letters | 26 | `A`–`Z` |
| Lower-case letters | 26 | `a`–`z` |
| **Total** | **65** | |

## Format

Plain UTF-8 text. Speaker names appear on their own line followed by a colon;
dialogue follows on subsequent lines, with blank lines between speeches. No
preprocessing has been applied — the file is byte-identical to the source.
