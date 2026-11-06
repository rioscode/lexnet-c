# lexnet-c

[Lexnet](https://huggingface.co/alpindale/lexnet), a 984-parameter neural network that
syntax-highlights C, read out of its weights as the if/else program it really is, written
down in plain C, and then optimized to 180-230x the speed of the model's own compiled
kernel while staying bit-exact with it.

```sh
make                # fetches the model files, builds build/lexnet and the original build/lexnet-ref
./build/lexnet model/lexnet.c
make test-quick     # differential test against the network
make bench          # reference vs. logic vs. fast
```

| executor | ns/byte (real C code, Apple M3 Max) |
| --- | ---: |
| reference float64 interpreter | 543 |
| reference compiled kernel | 140 |
| `lexnet_logic` - the network as if/else | 21.9 |
| `lexnet_fast` - block scanner, NEON | 0.77 |

Documentation lives in [`docs/`](docs/README.md): architecture, the gate-by-gate derivation,
the optimization log, testing and decisions.
