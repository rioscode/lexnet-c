# Key decisions

**Equivalence to the network, not to C.** The target is the network's output, quirks
included (`docs/network-to-logic.md`, section 4). A "better" lexer would be a different
program, and the only way to know the transcription is right is to compare it with the
original on everything.

**Two implementations instead of one.** `lexnet_logic.c` is the specification: one line per
unit, in the network's order, with its invariants. `lexnet_fast.c` is the product. Keeping
both means every optimization has a readable thing to be checked against, and the
differential test compares both.

**Derive invariants by exhaustive search, not by argument.** The boolean collapse of each
gate depends on which unit combinations can occur. A breadth-first search over the 205
reachable forward states (ignoring the hash accumulators) settles that in milliseconds and
is reproducible; hand-waving about "a string can't be inside a comment" is not.

**Raw bytes as keyword keys.** The network hashes words into two base-64 integers. Both
hashes are injective on the byte ranges they cover, so the zero-padded raw bytes are an
equivalent key and the per-byte hash loop disappears. The subtle part, that bytes 14+ are
never hashed, is encoded by clamping the key length to 14.

**Block-wise bitmasks over a token loop.** A per-token NEON scan is latency-bound (~20
cycles before the next token's position is known). Classifying 63 bytes at once and walking
events with `ctz` turns that into 1-cycle integer ops and makes whitespace and punctuation
free. This is the single biggest win on dense code (2-4x) and the reason the scanner looks
the way it does.

**63-byte blocks.** Keeping bit 63 clear guarantees no mask is ever all ones, so
`ctz(~mask)` is always defined and no branch is needed to guard it. The cost is 1/64 of the
throughput and a slightly higher rate of block-crossing tokens.

**Spans end the block.** When a comment, literal or `#` line starts, the block is painted,
the span is scanned by a dedicated routine and the next block starts after it. Simpler than
masking spans out of the block, and the span scanners already run at memory bandwidth.

**Branch-free word path, at a cost on predictable input.** The last step trades
instructions for branches: slower on inputs where the branches were always predicted
(repeated words, headers with long predictable identifiers: +20%), faster on real code
(-15%) and keyword-heavy input (-20%). Real code is what matters.

**Perfect hash with a hardcoded multiplier plus fallback search.** The search takes ten
attempts for this key set; the known multiplier is used first so initialisation is
deterministic and instant, and the search still runs if someone edits the keyword list.

**Initialisation in a constructor.** Tables are built at load time so that concurrent first
calls cannot race; a lazy check remains for toolchains without constructor support.

**No `fill` overshoot into painted blocks.** The 16-byte unconditional store trick is kept
only in the span scanners, where everything to the right is repainted later. Inside blocks,
word classes go through override masks.

## Rejected or deferred

- **Multi-threading.** Lexing state at an arbitrary split point is unknown (could be inside
  a comment); speculative parallel lexing with resynchronisation is possible but makes the
  equivalence argument much harder. Single-core 1.3-1.6 GB/s is already far beyond what a
  highlighter needs.
- **x86 SIMD paths.** The scalar fallbacks are correct and tested; SSE2/AVX2 versions of
  `classify_block`, `paint_block` and the span scanners would be mechanical ports.
- **Interleaving two blocks** to hide the event loop's latency chain: plausible 1.5x on dense
  code, significant complexity.
- **A smarter `lexnet_logic`** (hash table instead of linear keyword search): would make it
  3-4x faster but it is the readable reference, and `lexnet_fast` exists.
