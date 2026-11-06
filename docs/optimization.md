# Optimization log

All numbers: Apple M3 Max, Apple clang 21, `-O3 -march=native`, single core. "headers" is the
4.3 MB concatenation of the macOS SDK `usr/include/*.h` (43% comments, 30% preprocessor
lines); "real code" is 8 MB of `.c` files found on the machine; the four synthetic files isolate
one token kind each. Every step was verified bit-exact with `tests/difftest.c` before the
next one started.

| step | headers | real code | dense code | identifiers | punctuation | one comment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reference kernel | 142 | 140 | | | | |
| `lexnet_logic` | 21.8 | 21.9 | | | | |
| 1. token scanner | 0.79 | 1.36 | 2.38 | 4.16 | 5.24 | 0.033 |
| 2. + preprocessor fast path, 64-byte comment scan, raw-byte keyword keys | 0.71 | | 2.52 | 3.23 | 5.84 | |
| 3. 64-byte bitmask blocks, events via ctz | 0.55 | 0.95 | 1.27 | 2.42 | 1.43 | 0.032 |
| 4. perfect hash, override masks painted per block | 0.50 | 0.91 | 1.18 | 1.85 | 1.40 | |
| 5. branch-free word path, 63-byte blocks, paren-ahead mask, out-of-line slow paths | 0.62 | 0.77 | 1.40 | 1.59 | 1.52 | 0.031 |

(ns/byte; lower is better)

## 0. The if/else program (`lexnet_logic.c`)

Transcribing the network removes the float64 matrix-vector products (the interpreter does
~1000 multiply-adds per byte; the compiled kernel ~500) and replaces them with ~60 boolean
operations per byte. The two passes and the per-byte bit buffer are kept. 6.5x over the
kernel. The keyword tables are still searched linearly for every byte (~70 comparisons),
which is most of its remaining cost; it is left that way because this file is the readable
specification, not the fast path.

## 1. Token scanner

The first rewrite reads the program as what it is, a lexer, and processes tokens instead of
bytes:

- The forward state collapses to two bits carried across tokens: `P` (inside a `#` line) and
  `ls` (only whitespace since the last newline). Everything else is local to a token.
- The backward pass disappears. Its only non-local effects are (a) a word's class applies to
  all its bytes, which a token scanner knows anyway, (b) `fnc` needs "a `(` follows after
  spaces", a bounded look-ahead, (c) the first `/` of `//` or `/*` is a comment byte, which
  the scanner sees because it starts the comment at that `/`.
- Spans (block comments via a NEON `*/` search, line comments via `memchr`, literals via a
  NEON search for quote/backslash/newline, identifier runs via NEON) are found in 16-byte
  steps and painted with one `memset`.
- Keywords: the word's base-64 hashes `h1`, `h2` are computed while scanning and looked up in
  a small open-addressing table keyed by `h1`.
- Short spans are painted with an unconditional 16-byte store that overshoots; the next token
  overwrites the overshoot. Correct only because tokens are painted left to right.

180x over the kernel on headers. Comments run at 30 GB/s, i.e. memory bandwidth.

## 2. Small wins

- Preprocessor lines (30% of header bytes) get their own loop: NEON scans for the only four
  bytes that matter inside them (`\n " ' /`), everything in between is class 9 in one store.
- Block comments scan 64 bytes per iteration (`vextq` catches a `*/` straddling vectors).
- Keyword keys are the raw bytes instead of the base-64 hash: `h1` is injective on words of
  up to 8 bytes, so the zero-padded 8-byte load is an equivalent key, and the hash loop is gone.
  Words of 9..13 bytes are compared by length against the 8 long keywords; 14+ bytes only
  against the `_Static_assert` prefix (the network's quirk).
- Whitespace runs with NEON.

Only +10% overall. Ablation showed why: comment bytes were already free, and every token cost
~20 cycles of *latency*: NEON load, compare, `shrn`, `fmov` to a general register, `ctz`, and
only then is the next token's position known. Tokens are serial, so that chain is the clock.

## 3. Bitmask blocks (the simdjson idea)

Instead of one NEON chain per token, classify 64 bytes at once into bitmasks (one bit per
byte: alnum, numc, space, newline, special) with four loads, a handful of compares and a
`vpaddq`-based 64-bit movemask. Then:

- paint the whole block's base classes (whitespace / punctuation / identifier) with four
  vector stores: spaces, newlines and punctuation now cost nothing per token;
- visit only the "events" with a `ctz` loop: starts of alnum runs and the four span openers.
  A token's length is `ctz(~(mask >> p))`, a few 1-cycle integer ops;
- line-start detection becomes a carry-propagation trick: `((space + seed) ^ space) & space`
  fills every space run that starts right after a newline;
- a span opener ends the block; the span scanner runs; the next block starts after the span.
  A word or number that reaches the block end is finished by the byte scanners and also ends
  the block (so the next block repaints from a token boundary).

Dense code 2x faster, punctuation-heavy code 4x faster.

## 4. Perfect hash, painted overrides

- The keyword table gets a multiplier chosen so that all keys land in distinct slots: one
  multiply, one load, one compare, no probe loop.
- The unconditional 16-byte store from step 1 is no longer legal: the block was already
  painted, so an overshoot clobbers bytes nobody repaints. Instead each word's class is
  recorded as a bit range in one of four 64-bit override masks (keyword, type, function,
  number) and `paint_block` blends them in after the base classes, expanding each mask with
  `vqtbl1q` + `vtstq`.

## 5. Branch-free word path

Ablation on synthetic inputs (same word repeated: 0.69 ns/byte; random lengths: 1.14; 50%
keywords: 1.84) showed the remaining cost was branch misprediction, and the assembly showed
where: the compiler had turned `len > 8` in the key computation and the key comparison in the
lookup into real branches, and spilled the override masks because the vector constants of
classify/paint were live across the loop.

- Keys for short and long words are computed the same way: `key1` = bytes 0..min(len,8),
  `key2` = bytes 8..min(len,14), both zero-padded, masks derived from `len` with `csel`, no
  branch. The table holds all 74 words.
- The lookup returns `cls & -(match)` instead of `match ? cls : 0`.
- The `(` look-ahead is precomputed per block: `paren_ahead` bit p is set when the first
  non-space byte at or after p is `(` (bit-reverse, propagate through space runs, reverse
  back). A word needs one shift and one AND. Only spaces that run into the block end take a
  slow path.
- Blocks are 63 bytes (64 loaded, bit 63 cleared) so no mask is ever all ones and `ctz(~x)`
  is always defined - the "is it all ones" branch disappears.
- The event loop is its own `noinline` function so it is register-allocated without the NEON
  constants; the rare paths (`finish_long_token`, `paren_follows`, `open_span`) are
  `noinline` too.

Real code 0.91 -> 0.77 ns/byte, keyword-heavy input 1.84 -> 1.45. Perfectly predictable
inputs got slower (the branches used to be free there), which is why headers show 0.50 ->
0.62: long identifiers in headers are predictable, real code is not. Real code wins.

## Where the time goes now

On real code, `lexnet_fast` spends about 0.1 ns/byte in `classify_block` + `paint_block`
(2 NEON ops per byte), nothing measurable in comments and preprocessor lines, and the rest in
the event loop: roughly 20 cycles per word for `ctz`, key loads, perfect-hash lookup, the
`(` bit, four conditional ORs and the loop-carried `events` update. Further options,
not pursued (see `docs/decisions.md`): processing two blocks' events interleaved to hide
latency, SSE/AVX2 ports, speculative multi-threading.

## Sanity note on the reference numbers

The model's README reports 88 ns/char for its compiled kernel on an M4 Pro with a 188 KB
corpus. We measure 140 ns/char on an M3 Max with 4-8 MB corpora; the kernel keeps a
2-byte-per-char buffer between its passes, so larger inputs run from L2/L3 rather than L1.
All speed-ups in this document are against our own measurement of the reference on the same
corpus and machine.
