# Lexnet as a C program

[Lexnet](https://huggingface.co/alpindale/lexnet) is a 984-parameter bidirectional recurrent
network that tags every byte of a C source file with one of 11 syntax-highlighting classes.
Every unit of the network is a threshold gate over 0/1 inputs (plus two integer accumulators
and one counter), so the network *is* an if/else program. This project:

1. reads that program out of the weights, unit by unit (`docs/network-to-logic.md`);
2. writes it down as plain C, two passes, no floating point (`src/lexnet_logic.c`);
3. rewrites it as a bit-parallel block scanner that produces the same output 180x faster
   than the model's own compiled kernel (`src/lexnet_fast.c`, `docs/optimization.md`);
4. proves both bit-exact against the original network on 275 MB of inputs
   (`tests/difftest.c`, `docs/testing.md`).

## Layout

| Path | What |
| --- | --- |
| `model/` | The original Hugging Face files (weights, reference `lexnet.c`, kernel header). Not committed: `scripts/fetch_model.sh` downloads them, `make` does so automatically. |
| `scripts/fetch_model.sh` | Downloads the model files (uses `hf` if installed, else `curl`). |
| `src/lexnet.h` | Public API: class ids, `lexnet_logic()`, `lexnet_fast()`. |
| `src/lexnet_logic.c` | The network as an if/else program. Readable spec, 22 ns/byte. |
| `src/lexnet_fast.c` | The optimized single-pass scanner. 0.6-0.8 ns/byte. |
| `src/main.c` | CLI, flag-compatible with the original. |
| `tests/difftest.c` | In-process differential test against the reference kernel and interpreter. |
| `bench/bench.sh` | Times reference, logic and fast on one file. |
| `docs/` | This documentation. |

## Build

Requires a C11 compiler and, for the first build, network access to huggingface.co (the
original model files are fetched into `model/`). NEON is used on arm64; everything else uses
the portable scalar paths.

```sh
make                # fetches model/, builds build/lexnet (ours) and build/lexnet-ref (the original)
make test-quick     # ~10 s differential test
make test           # full differential test (several minutes)
make bench          # benchmark on the macOS SDK headers (any file: make bench CORPUS=path)
```

## Usage

```sh
./build/lexnet file.c                   # ANSI-coloured output, same palette as the original
./build/lexnet --classes file.c         # one class digit per byte: 0123456789A
./build/lexnet --impl logic file.c      # run the readable if/else version instead
./build/lexnet --bench 50 file.c        # ns/byte
```

The class ids are those of the network's head:

| id | class | id | class |
| --- | --- | --- | --- |
| 0 | whitespace | 6 | string literal |
| 1 | keyword | 7 | char literal |
| 2 | type | 8 | comment |
| 3 | function name | 9 | preprocessor line |
| 4 | identifier | 10 | punctuation / anything else |
| 5 | number, `NULL`, `true`, ... | | |

As a library: include `src/lexnet.h`, compile `src/lexnet_fast.c` (and/or `src/lexnet_logic.c`),
call `lexnet_fast(src, n, cls)`. Both functions are pure, re-entrant and never fail.

## Results

Apple M3 Max, Apple clang 21, `-O3 -march=native`. Reference numbers are the model's own
`lexnet.c` built exactly as its README says.

| executor | SDK headers (4.3 MB) | real C code (8 MB) | speed-up vs kernel |
| --- | ---: | ---: | ---: |
| reference, float64 interpreter | 536 ns/byte | 543 ns/byte | 0.26x |
| reference, compiled kernel | 142 ns/byte | 140 ns/byte | 1x |
| `lexnet_logic` (if/else, two passes) | 21.8 ns/byte | 21.9 ns/byte | 6.5x |
| `lexnet_fast` (block scanner, NEON) | 0.62 ns/byte | 0.77 ns/byte | **180-230x** |

`lexnet_fast` runs at 1.3-1.6 GB/s on real code and at memory bandwidth (32 GB/s) inside
comments. The original README quotes tree-sitter-c at 51-100 ns/char on comparable hardware.

## Further reading

- `docs/architecture.md` - the network and how the program is organised around it
- `docs/network-to-logic.md` - the derivation of every gate, the invariants, the quirks
- `docs/optimization.md` - every optimization step with measurements
- `docs/testing.md` - how equivalence is established
- `docs/decisions.md` - key decisions and alternatives rejected
