/* lexnet.h - Lexnet (https://huggingface.co/alpindale/lexnet) recreated as a plain C program.
 *
 * The original is a 984-parameter bidirectional recurrent network that tags every byte of a
 * C source file with one of 11 classes. Every unit of that network is a threshold gate over
 * 0/1 inputs (plus two integer accumulators and one counter), so the whole net is an exact
 * if/else program. This header exposes two bit-exact re-implementations of that program:
 *
 *   lexnet_logic  - the network transcribed unit by unit into boolean/integer C. Two passes
 *                   (forward, backward) over the bytes, exactly like the net. Readable spec.
 *   lexnet_fast   - the same program rewritten as a token scanner: one pass, no per-byte
 *                   state vector, table lookups, memset/memchr spans, NEON on arm64.
 *
 * Both produce byte-identical output to the reference compiled kernel (lexnet_kernel.h) for
 * every input, including the network's quirks (see docs/network-to-logic.md).
 */
#ifndef LEXNET_H
#define LEXNET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Output classes (same ids as the network's head). */
enum lexnet_class {
    LEX_WS      = 0,   /* whitespace (space, tab, newline ...)       */
    LEX_KEYWORD = 1,   /* if, while, struct, _Static_assert, ...      */
    LEX_TYPE    = 2,   /* int, char, size_t, uint32_t, FILE, ...      */
    LEX_FUNC    = 3,   /* identifier followed by optional spaces and '(' */
    LEX_IDENT   = 4,   /* any other identifier                        */
    LEX_NUMBER  = 5,   /* pp-number (123, 0x1F, 1.5e3) or NULL/true/false/EOF/nullptr */
    LEX_STRING  = 6,   /* "..." including the quotes                  */
    LEX_CHAR    = 7,   /* '...' including the quotes                  */
    LEX_COMMENT = 8,   /* // ... and block comments                   */
    LEX_PREPROC = 9,   /* a # line, except strings/chars/comments in it */
    LEX_PUNCT   = 10,  /* everything else (operators, braces, bytes >= 128 ...) */
    LEX_N_CLASSES = 11
};

/* Tag n bytes of src; writes one class id per byte into cls (cls must hold n bytes).
 * Both functions are pure, re-entrant and never fail. */
void lexnet_logic(const uint8_t *src, size_t n, uint8_t *cls);
void lexnet_fast(const uint8_t *src, size_t n, uint8_t *cls);

#ifdef __cplusplus
}
#endif
#endif /* LEXNET_H */
