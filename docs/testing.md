# Testing

The claim is bit-exact equivalence with the network for every input, so the test is a
differential test against the network itself, not against a hand-written expectation.

## What runs

`tests/difftest.c` includes the original `model/lexnet.c` with its `main` renamed, loads
`model/lexnet_weights.bin`, checks the kernel's structure hash, and then for every input
compares the output of `lexnet_logic` and `lexnet_fast` byte by byte with the output of the
reference compiled kernel (`classify_compiled`). For the real files it also runs the
reference float64 interpreter (`classify`) and checks that kernel and interpreter agree, so
the comparison is anchored to the network, not only to its kernel.

Inputs, in order:

1. **Files** named on the command line. `make test` passes the model's sources, this
   project's sources and the 230 macOS SDK headers (4.3 MB). The final run also used 8 MB of
   real C code from this machine.
2. **Exhaustive strings** of length 1..6 over a 16-symbol alphabet that covers every feature
   class: `a Z _ 1 . space \n " ' \ / * # ( ;` and a byte >= 128. 17.9 million strings. This
   catches every short-range interaction of modes, escapes, openers and closers.
3. **Token soup**: 20 000 random sequences built from ~280 hand-picked tokens: every keyword,
   type and constant; near misses (`iff`, `_Static_assertion`, `constexp`, `uintptr_`);
   identifiers of lengths 1..26 including 7, 8, 9, 13, 14, 15; numbers (`0x1F`, `1.5e-3`,
   `123abc`, `.5`, `1..2`); strings and chars with escapes, unterminated, with backslash-
   newline; comments of every shape (`/**/`, `/*/`, nested-looking, `//` ending in backslash);
   preprocessor lines with continuations and embedded comments; function-call shapes with
   tabs, spaces and newlines before `(`; odd bytes (`\0`, `\x7f`, `\xff`, UTF-8); separated
   by random whitespace including `\r\n` and `\\\n`. Sizes from 300 bytes to 1 MB.
4. **Random bytes**: 20 000 buffers, half uniform over all 256 values, half over the
   16-symbol alphabet, up to 100 KB.

Each failure prints the input around the mismatch, escaped, with the reference and the
obtained class strings.

## Builds

The test runs twice: once as built for the machine (NEON on arm64) and once with
`-DLEXNET_NO_SIMD`, which exercises the scalar block classifier, painter and span scanners.
Both must pass; `make test-quick` runs a 10-second subset of both, `make test` the full set.

## Results of the final run

| build | cases | bytes compared | failures |
| --- | ---: | ---: | ---: |
| NEON | 17 935 934 | 275 781 100 | 0 |
| scalar | 17 935 931 | 269 115 517 | 0 |

The CLI binaries were also compared end to end: `lexnet --classes` and the coloured output of
`build/lexnet` and `build/lexnet-ref` are identical on the model's own source, the header
corpus and the real-code corpus (`cmp`).

## Bugs the tests caught during development

- An unconditional 16-byte store that overshoots a word is only correct when tokens are
  painted left to right; after painting became block-wise it clobbered the bytes after short
  words (caught by the exhaustive strings and the model's own source).
- The "spaces reach the end of the block" mask used bit 63 after blocks had been shortened to
  63 bytes, so a `(` just outside the block was missed (caught by token soup; reproduced by
  sliding `foo  (` over every offset 0..70).
- A stale "preprocessor" flag in a per-block struct that was no longer zero-initialised
  turned ordinary words into class 9 (caught by the model's own source).

## Limits

The exhaustive part proves equivalence for all inputs up to 6 bytes over the alphabet; beyond
that the evidence is statistical (hundreds of MB of structured random input plus real code).
Everything in the scanner is either a direct transcription of a gate or a bit-trick whose
correctness argument is written next to it; the tests are there to catch the arguments that
are wrong.
