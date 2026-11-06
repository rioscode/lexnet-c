# From weights to if/else

This is the derivation behind `src/lexnet_logic.c`. Names follow `model/lexnet_kernel.h`
(`FState`, `BState`, `u_*`). Values were read with a small script from `lexnet_weights.bin`;
every threshold quoted here is exact (the file stores integers in float64).

## 1. Features (embedding table)

The embedding maps a byte to 14 numbers. 13 of them are 0/1 indicators, the 14th is a code:

| x | meaning | bytes |
| --- | --- | --- |
| x0 | alpha | `A-Z a-z _` |
| x1 | digit | `0-9` |
| x2 | alnum | `A-Z a-z 0-9 _` |
| x3 | number continuation | alnum and `.` |
| x4 | space | `\t \v \f \r` and space |
| x5 | newline | `\n` |
| x6 .. x12 | one byte each | `"` `'` `\` `/` `*` `#` `(` |
| x13 | code | `a-z` = 1..26, `A-Z` = 27..52, `0-9` = 53..62, `_` = 63, else 0 |

Bytes >= 128 use row 127, which is all zero: they are "other" bytes, like `@` or `\0`.

## 2. Gates

A unit `clip1(b + sum w_i v_i)` with 0/1 inputs and integer weights is a threshold gate. With
the invariants of section 3 each one collapses to the boolean expression used in
`lexnet_logic.c`. `p.` is the previous byte's state, `x` the current byte's features.

### Forward layer 0: continuation and helpers

| unit | weights | logic |
| --- | --- | --- |
| contC | `C - closed` | `p.C && !p.closed` |
| contL | `L - nl` | `p.L && !nl` |
| contS | `S - nl - Sclose` | `p.S && !nl && !p.Sclose` |
| contQ | `Q - nl - Qclose` | `p.Q && !nl && !p.Qclose` |
| contP | `P - nl + bsP` | `p.P && (!nl \|\| p.bsP)` |
| opened | `star + sf - 1` | `star && p.sf` (the `*` of `/*`) |
| lo | `slash + sf - 1` | `slash && p.sf` (second `/` of `//`) |
| closed | `slash + star_in - 1` | `slash && p.star_in` (the `/` of `*/`) |
| ls | `space + 2 nl + ls - 1` | `nl \|\| (space && p.ls)` |
| wcont | `alnum + W - 1` | `alnum && p.W` |
| wstart0 | `alpha - N` | `alpha && !p.N` |
| nany | `digit + N` | `digit \|\| p.N` |
| d | `14 - 15 W + d`, ceiling 14 | `p.W ? max(p.d - 1, 0) : 14` |
| lt8 | `8 - 15 W + d` | `!p.W \|\| p.d >= 8` (index of this byte in its word < 8) |
| ge8 | `-7 + 15 W - d` | `p.W && p.d <= 7` (index >= 8) |
| ge14 | `-13 + 15 W - d` | `p.W && p.d <= 1` (index >= 14) |

`d` counts down from 14 at the first byte of a word, so `d = max(14 - index, 0)`.

### Forward layer 1: mode flags, word and number, hash gates

With `inMode = contC || contL || contS || contQ` (at most one is set, see section 3):

| unit | logic |
| --- | --- |
| C | `contC \|\| opened` |
| L | `contL \|\| lo` |
| copen2 | `opened \|\| lo` (second byte of a comment opener; read by the backward pass) |
| S | `contS \|\| (dq && !inMode)` |
| Q | `contQ \|\| (sq && !inMode)` |
| P | `contP \|\| (hash && p.ls && !inMode)` |
| Sclose | `contS && dq && !p.E` |
| Qclose | `contQ && sq && !p.E` |
| E | `(contS \|\| contQ) && bs && !p.E` |
| star_in | `star && contC` |
| sf | `slash && !inMode && !lo` |
| bsP | `bs && contP` |
| W | `(wcont \|\| wstart0) && !inMode` |
| N | `nany && numc && !wcont && !wstart0 && !inMode` |

The six "accumulator gates" use weights of B = 2^49 and the ReLU ceiling of infinity to
select between keeping, resetting and holding the hash. Because `h1 < 64^8 = 2^48 < B`, a
term `h - B*k` with `k >= 1` is always negative and clips to 0:

| unit | logic |
| --- | --- |
| g1 | `(lt8 && alnum && p.W) ? p.h1 : 0` |
| c1 | `lt8 ? code : 0` |
| hold1 | `lt8 ? 0 : p.h1` |
| g2 | `(ge8 && !ge14 && alnum) ? p.h2 : 0` |
| c2 | `(ge8 && !ge14) ? code : 0` |
| hold2 | `(p.W && ge8 == ge14) ? p.h2 : 0` |

### Forward layer 2: accumulators

`h1 = 64 g1 + c1 + hold1`, `h2 = 64 g2 + c2 + hold2`. Net effect: `h1` is the base-64 number
formed by the codes of word bytes 0..7, `h2` by bytes 8..13, both restart at a word start
(`p.W == 0`), and bytes from index 14 on are ignored.

### Forward layers 3-5: keyword tables

Layer 3 has 162 units in pairs: `ge = clip1(h - (T-1))`, `le = clip1((T+1) - h)`. For integer
`h` exactly one of the pair is 1 unless `h == T`, when both are. Layer 4 sums the pairs of a
group and subtracts the group size, so a group unit is 1 iff one pair matched; the three
"short" groups additionally subtract `h2`, i.e. require `h2 == 0` (the word has at most 8
bytes). Layer 5 ORs the groups. Decoding the thresholds back to strings:

- **KW** (class 1): `auto break case const continue default do else enum extern for goto if
  inline register restrict return sizeof static struct switch typedef union volatile while
  _Alignas _Alignof _Atomic _Generic alignas alignof typeof` and, matched on `(h1, h2)`,
  `_Noreturn _Static_assert _Thread_local constexpr static_assert thread_local`
- **TY** (class 2): `char double float int long short signed unsigned void bool _Bool _Complex
  size_t ssize_t intptr_t wchar_t time_t off_t pid_t int8_t int16_t int32_t int64_t uint8_t
  uint16_t uint32_t uint64_t FILE va_list` and `ptrdiff_t uintptr_t`
- **CONST** (class 5): `NULL nullptr true false EOF`

### Forward output

15 bits per byte: `C L S Q P W N KW TY CONST copen2 slash space nl lparen`.

### Backward cell

`r.` is the state of the byte to the right (all zero after the last byte).

| unit | logic |
| --- | --- |
| lps | `lparen \|\| (space && r.lps)` - a `(` follows after spaces |
| cmt | `C \|\| L \|\| (slash && r.copen2)` - includes the first `/` of an opener |
| kwc | `(W && KW && !r.W) \|\| (W && r.W && r.kwc)` - keyword, from the last byte leftwards |
| tyc | same with TY |
| cst | same with CONST |
| fnc | `(W && r.lps && !KW && !TY && !CONST && !r.W) \|\| (W && r.W && r.fnc)` |

### Head

Logits: `0: 0.5`, `1: 7 kwc`, `2: 7 tyc`, `3: 7 fnc`, `4: 5 W`, `5: 6 (cst + N)`, `6: 10 S`,
`7: 10 Q`, `8: 9 cmt`, `9: 8 P`, `10: 1 - space - nl`. Argmax with the lowest id winning ties
is the priority chain `S, Q, cmt, P, kwc, tyc, fnc, cst|N, W, !space && !nl -> punct, else ws`.

## 3. Invariants

A breadth-first search over the reachable forward states (ignoring `h1`, `h2`, which do not
feed back into the boolean units) finds 205 states and confirms, on every reachable state and
every input class:

- `C`, `L`, `S`, `Q` are mutually exclusive, and so are `contC`, `contL`, `contS`, `contQ`;
- `W` and `N` are mutually exclusive;
- `star_in`, `closed` imply `C`; `Sclose` implies `S`; `Qclose` implies `Q`; `E` implies `S` or `Q`;
  `bsP` implies `P`; `sf`, `W`, `N` imply no comment/literal mode;
- `d` is an integer in 0..14, everything else is 0/1;
- `P` does coexist with every mode: a `#` line can contain strings, chars and comments.

These are what let the sums in section 2 be written as booleans (for example
`S = clip1(contS + dq - contC - contL - contQ)` is `contS || (dq && !inMode)` only because at
most one of the subtracted terms is ever 1).

## 4. Quirks that are preserved

The reimplementations reproduce the network, not the C standard. Behaviour worth knowing:

- A string or char literal ends at a newline (exclusive); the newline is whitespace.
  A backslash before that newline does not continue the literal.
- A preprocessor line ends at a newline unless the byte before the newline is a backslash,
  no matter where that backslash is: inside a string, a `//` comment or a block comment.
  Only the bytes of the `#` line itself are class 9; the terminating newline is 0, a
  continued newline is 9.
- Inside a `#` line, words, numbers, spaces and `#` are all class 9; strings, chars and
  comments keep their own class. A block comment that contains an unescaped newline ends
  the `#` line.
- `#` opens a preprocessor line only if every byte since the last newline (or the start of
  the file) is in the space class, and not inside a block comment.
- Only bytes 0..13 of a word are hashed. Short keywords require the word to end by byte 8;
  long keywords of 9..13 bytes require exact length; `_Static_assert` has 14 bytes, so any
  word that starts with `_Static_assert` is a keyword.
- A pp-number starts at a digit that does not continue a word and runs over `[A-Za-z0-9_.]`:
  `123abc`, `1.2.3`, `0x1F` are one number; `.5` is punctuation followed by a number;
  `1.e5` is one number but `1e-5` is a number, punctuation and a number.
- A word is `[A-Za-z_][A-Za-z0-9_]*` not preceded by a pp-number byte; after `"abc"` or `*/`
  a word may start immediately.
- Function name means: a word that is not a keyword/type/constant and is followed by `(`
  with only space-class bytes in between (tabs yes, newlines no).
- `/*/` does not close a comment; `/**/` does. An unterminated comment or literal runs to the
  end of the input.
- Bytes >= 128 and control bytes other than `\t \n \v \f \r` are punctuation (class 10).
