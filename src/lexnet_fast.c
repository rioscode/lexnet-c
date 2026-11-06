/* lexnet_fast.c - the Lexnet if/else program rewritten as a single-pass, block-wise scanner.
 *
 * lexnet_logic.c evaluates ~60 boolean gates per byte, twice (forward + backward). But the
 * output class of a byte only depends on a handful of things that are cheap to find
 * directly once the program is read as a lexer:
 *
 *   - which span the byte is in: comments, string and char literals and preprocessor lines
 *     are found with NEON/memchr scans and painted with one memset each;
 *   - everything else is handled 63 bytes at a time: NEON computes one bitmask per byte
 *     class, whitespace / punctuation / identifier are painted for the whole block in four
 *     vector stores, and only the "events" (starts of words and numbers, span openers) are
 *     visited with ctz;
 *   - for words: whether the word is one of the 74 keywords/types/constants the network
 *     recognises (its two base-64 hashes h1/h2 are injective, so comparing the raw bytes is
 *     equivalent: one perfect-hash lookup), and whether '(' follows after optional spaces
 *     (the network's backward "lps" unit), precomputed for the whole block as a bitmask
 *     instead of a backward pass;
 *   - the preprocessor flag P (set by '#' at line start, cleared at a newline that is not
 *     preceded by a backslash) and the line-start flag, carried across blocks.
 *
 * docs/optimization.md explains every step and why it is equivalent to the network;
 * tests/difftest.c proves it bit-exact against the reference kernel.
 */
#include "lexnet.h"

#include <stdbool.h>
#include <string.h>

#if defined(__aarch64__) && !defined(LEXNET_NO_SIMD)
#include <arm_neon.h>
#define LEXNET_NEON 1
#else
#define LEXNET_NEON 0
#endif

/* ---- byte tables ---------------------------------------------------------------------- */
enum { CF_ALNUM = 1, CF_NUMC = 2, CF_SPACE = 4, CF_PP_STOP = 8 };

static uint8_t cflags[256];    /* CF_* membership bits  */

/* ---- keyword table ---------------------------------------------------------------------- */
/* The network matches a word by two base-64 hashes: h1 over bytes 0..7 and h2 over bytes
 * 8..13 (bytes 14 and beyond are never hashed). Both hashes are injective, so the raw bytes,
 * zero-padded and loaded little-endian, are an equivalent key:
 *   key1 = bytes 0..min(len,8)   as 8 bytes,  key2 = bytes 8..min(len,14) as 6 bytes.
 * A word matches an entry iff key1 and key2 are equal. That covers the short keywords
 * (key2 == 0 means the word really ends by byte 8) and the long ones, including the
 * network's prefix quirk: "_Static_assert" is 14 bytes, so every word starting with it matches. */
typedef struct { uint64_t key1, key2; uint8_t cls; } KwEntry;

/* Perfect hash on key1: a multiplier is chosen at init so that the 74 keys land in distinct
 * slots. A lookup is then one multiply, one load and two compares, with no probing loop. */
#define KW_SLOTS 512
static KwEntry kwtab[KW_SLOTS];
static uint64_t kw_mul;

static const char *const KEYWORDS[] = {
    "auto", "break", "case", "const", "continue", "default", "do", "else", "enum", "extern",
    "for", "goto", "if", "inline", "register", "restrict", "return", "sizeof", "static",
    "struct", "switch", "typedef", "union", "volatile", "while", "_Alignas", "_Alignof",
    "_Atomic", "_Generic", "alignas", "alignof", "typeof",
    "_Noreturn", "_Static_assert", "_Thread_local", "constexpr", "static_assert", "thread_local",
};
static const char *const TYPES[] = {
    "char", "double", "float", "int", "long", "short", "signed", "unsigned", "void", "bool",
    "_Bool", "_Complex", "size_t", "ssize_t", "intptr_t", "wchar_t", "time_t", "off_t", "pid_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "FILE", "va_list", "ptrdiff_t", "uintptr_t",
};
static const char *const CONSTANTS[] = { "NULL", "nullptr", "true", "false", "EOF" };

static inline size_t kw_slot(uint64_t key1) {
    return (size_t)((key1 * kw_mul) >> 55);   /* top 9 bits */
}

static inline uint64_t load_le64(const void *p) {
    uint64_t v;
    memcpy(&v, p, 8);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    v = __builtin_bswap64(v);
#endif
    return v;
}

/* keys of the word at w with len bytes, computed byte by byte (init and the tail of the input) */
static __attribute__((noinline)) void kw_keys_slow(const uint8_t *w, size_t len, uint64_t *key1, uint64_t *key2) {
    uint8_t b[16] = { 0 };
    memcpy(b, w, len < 14 ? len : 14);
    *key1 = load_le64(b);
    *key2 = load_le64(b + 8);
}

/* keys of the word at s+i with len bytes, when 16 bytes can be loaded; branch-free */
static inline void kw_keys(const uint8_t *s, size_t i, unsigned len, uint64_t *key1, uint64_t *key2) {
    const uint64_t v1 = load_le64(s + i), v2 = load_le64(s + i + 8);
    const unsigned keep1 = len < 8 ? len : 8;                        /* 1..8 bytes of key1 */
    const unsigned keep2 = len <= 8 ? 0 : (len >= 14 ? 6 : len - 8);  /* 0..6 bytes of key2 */
    *key1 = v1 & (~0ULL >> (64 - 8 * keep1));
    *key2 = v2 & ((~0ULL >> (63 - 8 * keep2)) >> 1);                 /* keep2 == 0 -> mask 0 */
}

/* returns false on a slot collision */
static bool kw_insert(const char *w, uint8_t cls) {
    uint64_t key1, key2;
    kw_keys_slow((const uint8_t *)w, strlen(w), &key1, &key2);
    size_t s = kw_slot(key1);
    if (kwtab[s].key1) return false;
    kwtab[s].key1 = key1; kwtab[s].key2 = key2; kwtab[s].cls = cls;
    return true;
}

static void kw_build(void) {
    /* attempt 0 is the known collision-free multiplier for this key set; the search only
     * runs if the table contents change */
    for (unsigned attempt = 0;; attempt++) {
        memset(kwtab, 0, sizeof kwtab);
        kw_mul = attempt == 0 ? 0x3D22ABF114F2081FULL
                              : 0x9E3779B97F4A7C15ULL + 2 * (uint64_t)attempt * 0x2545F4914F6CDD1DULL;
        bool ok = true;
        for (size_t i = 0; ok && i < sizeof KEYWORDS / sizeof *KEYWORDS; i++) ok = kw_insert(KEYWORDS[i], LEX_KEYWORD);
        for (size_t i = 0; ok && i < sizeof TYPES / sizeof *TYPES; i++) ok = kw_insert(TYPES[i], LEX_TYPE);
        for (size_t i = 0; ok && i < sizeof CONSTANTS / sizeof *CONSTANTS; i++) ok = kw_insert(CONSTANTS[i], LEX_NUMBER);
        if (ok) return;
    }
}

/* class of the word with these keys, or 0 for a plain identifier; branch-free */
static inline uint8_t kw_lookup(uint64_t key1, uint64_t key2) {
    const KwEntry *e = &kwtab[kw_slot(key1)];
    const unsigned match = (e->key1 == key1) & (e->key2 == key2);
    return (uint8_t)(e->cls & (0u - match));
}

static bool tables_ready;

/* Built once at load time (so concurrent first calls never race); the lazy check in
 * lexnet_fast() is only a fallback for toolchains without constructor support. */
static __attribute__((constructor)) void init_tables(void) {
    for (unsigned c = 0; c < 256; c++) {
        bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        bool digit = c >= '0' && c <= '9';
        uint8_t f = 0;
        if (alpha || digit) f |= CF_ALNUM | CF_NUMC;
        if (c == '.') f |= CF_NUMC;
        if (c == ' ' || c == '\t' || c == '\v' || c == '\f' || c == '\r') f |= CF_SPACE;
        if (c == '\n' || c == '"' || c == '\'' || c == '/') f |= CF_PP_STOP;
        cflags[c] = f;
    }
    kw_build();
    tables_ready = true;
}

/* ---- span writer -------------------------------------------------------------------------- */
/* Write len copies of k at cls+i. Spans are written left to right, so a short span may
 * splat a full 16-byte block: whatever lies beyond the span is overwritten by the next one. */
static inline void fill(uint8_t *cls, size_t i, size_t len, uint8_t k, size_t n) {
    if (i + 16 <= n) {
        memset(cls + i, k, 16);
        if (len > 16) memset(cls + i + 16, k, len - 16);
    } else {
        memset(cls + i, k, len);
    }
}

/* ---- scanners ---------------------------------------------------------------------------- */
#if LEXNET_NEON
/* 4-bit-per-lane movemask: lane k of the comparison result -> bits 4k..4k+3 */
static inline uint64_t neon_mask(uint8x16_t m) {
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(m), 4)), 0);
}
static inline uint8x16_t neon_is_alnum(uint8x16_t v) {
    uint8x16_t lower = vorrq_u8(v, vdupq_n_u8(0x20));
    uint8x16_t alpha = vcleq_u8(vsubq_u8(lower, vdupq_n_u8('a')), vdupq_n_u8(25));
    uint8x16_t digit = vcleq_u8(vsubq_u8(v, vdupq_n_u8('0')), vdupq_n_u8(9));
    uint8x16_t us = vceqq_u8(v, vdupq_n_u8('_'));
    return vorrq_u8(vorrq_u8(alpha, digit), us);
}
/* ' ' \t \v \f \r  ==  32 or (9..13 without 10) */
static inline uint8x16_t neon_is_space(uint8x16_t v) {
    uint8x16_t ctl = vandq_u8(vcleq_u8(vsubq_u8(v, vdupq_n_u8(9)), vdupq_n_u8(4)),
                              vmvnq_u8(vceqq_u8(v, vdupq_n_u8('\n'))));
    return vorrq_u8(vceqq_u8(v, vdupq_n_u8(' ')), ctl);
}
/* \n " ' /  : the bytes that end a run of plain preprocessor bytes */
static inline uint8x16_t neon_is_pp_stop(uint8x16_t v) {
    return vorrq_u8(vorrq_u8(vceqq_u8(v, vdupq_n_u8('\n')), vceqq_u8(v, vdupq_n_u8('"'))),
                    vorrq_u8(vceqq_u8(v, vdupq_n_u8('\'')), vceqq_u8(v, vdupq_n_u8('/'))));
}
#endif

/* end of the [A-Za-z0-9_] run starting at j */
static inline size_t scan_word(const uint8_t *s, size_t j, size_t n) {
#if LEXNET_NEON
    while (j + 16 <= n) {
        uint64_t m = neon_mask(neon_is_alnum(vld1q_u8(s + j)));
        if (m != ~0ULL) return j + (size_t)(__builtin_ctzll(~m) >> 2);
        j += 16;
    }
#endif
    while (j < n && (cflags[s[j]] & CF_ALNUM)) j++;
    return j;
}

/* end of the [A-Za-z0-9_.] run starting at j */
static inline size_t scan_number(const uint8_t *s, size_t j, size_t n) {
    while (j < n && (cflags[s[j]] & CF_NUMC)) j++;
    return j;
}

/* end of the run of space-class bytes (' ' \t \v \f \r) starting at j */
static inline size_t scan_spaces(const uint8_t *s, size_t j, size_t n) {
#if LEXNET_NEON
    while (j + 16 <= n) {
        uint64_t m = neon_mask(neon_is_space(vld1q_u8(s + j)));
        if (m != ~0ULL) return j + (size_t)(__builtin_ctzll(~m) >> 2);
        j += 16;
    }
#endif
    while (j < n && (cflags[s[j]] & CF_SPACE)) j++;
    return j;
}

/* end of the run of bytes that are plain preprocessor bytes (not \n " ' /) starting at j */
static inline size_t scan_pp_plain(const uint8_t *s, size_t j, size_t n) {
#if LEXNET_NEON
    while (j + 16 <= n) {
        uint64_t m = neon_mask(neon_is_pp_stop(vld1q_u8(s + j)));
        if (m) return j + (size_t)(__builtin_ctzll(m) >> 2);
        j += 16;
    }
#endif
    while (j < n && !(cflags[s[j]] & CF_PP_STOP)) j++;
    return j;
}

/* String or char literal opened by the quote q at i. Returns the index just past the literal:
 * past the closing quote, or at the newline / end of input that cuts it short. A backslash
 * escapes the next byte, except that a newline always ends the literal. */
static inline size_t scan_quoted(const uint8_t *s, size_t i, size_t n, uint8_t q) {
    size_t j = i + 1;
    for (;;) {
#if LEXNET_NEON
        while (j + 16 <= n) {
            uint8x16_t v = vld1q_u8(s + j);
            uint8x16_t hit = vorrq_u8(vorrq_u8(vceqq_u8(v, vdupq_n_u8(q)), vceqq_u8(v, vdupq_n_u8('\\'))),
                                      vceqq_u8(v, vdupq_n_u8('\n')));
            uint64_t m = neon_mask(hit);
            if (m) { j += (size_t)(__builtin_ctzll(m) >> 2); break; }
            j += 16;
        }
#endif
        while (j < n) {
            uint8_t c = s[j];
            if (c == q || c == '\\' || c == '\n') break;
            j++;
        }
        if (j >= n) return n;
        if (s[j] == '\n') return j;                 /* unterminated: newline is not part of it */
        if (s[j] == q) return j + 1;                /* closing quote belongs to the literal     */
        /* backslash: the next byte is escaped unless it is a newline (which still ends it) */
        j++;
        if (j < n && s[j] == '\n') return j;
        j++;
    }
}

/* Block comment opened at i ("/*"). Returns the index just past the closing "* /" or n.
 * The '*' of the opener cannot be the '*' of the closer, so the search starts at i+2. */
static inline size_t scan_block_comment(const uint8_t *s, size_t i, size_t n) {
    size_t j = i + 2;
#if LEXNET_NEON
    const uint8x16_t star = vdupq_n_u8('*'), slash = vdupq_n_u8('/');
    /* 64 bytes per iteration: a "* /" pair straddling vectors is caught by vextq */
    while (j + 80 <= n) {
        uint8x16_t a0 = vld1q_u8(s + j), a1 = vld1q_u8(s + j + 16), a2 = vld1q_u8(s + j + 32),
                   a3 = vld1q_u8(s + j + 48), a4 = vld1q_u8(s + j + 64);
        uint8x16_t h0 = vandq_u8(vceqq_u8(a0, star), vceqq_u8(vextq_u8(a0, a1, 1), slash));
        uint8x16_t h1 = vandq_u8(vceqq_u8(a1, star), vceqq_u8(vextq_u8(a1, a2, 1), slash));
        uint8x16_t h2 = vandq_u8(vceqq_u8(a2, star), vceqq_u8(vextq_u8(a2, a3, 1), slash));
        uint8x16_t h3 = vandq_u8(vceqq_u8(a3, star), vceqq_u8(vextq_u8(a3, a4, 1), slash));
        uint8x16_t any = vorrq_u8(vorrq_u8(h0, h1), vorrq_u8(h2, h3));
        if (vmaxvq_u8(any)) {
            uint64_t m;
            if ((m = neon_mask(h0))) return j + (size_t)(__builtin_ctzll(m) >> 2) + 2;
            if ((m = neon_mask(h1))) return j + 16 + (size_t)(__builtin_ctzll(m) >> 2) + 2;
            if ((m = neon_mask(h2))) return j + 32 + (size_t)(__builtin_ctzll(m) >> 2) + 2;
            m = neon_mask(h3);       return j + 48 + (size_t)(__builtin_ctzll(m) >> 2) + 2;
        }
        j += 64;
    }
    while (j + 17 <= n) {
        uint8x16_t a = vld1q_u8(s + j), b = vld1q_u8(s + j + 1);
        uint64_t m = neon_mask(vandq_u8(vceqq_u8(a, star), vceqq_u8(b, slash)));
        if (m) return j + (size_t)(__builtin_ctzll(m) >> 2) + 2;
        j += 16;
    }
#endif
    while (j + 1 < n) {
        const uint8_t *p = memchr(s + j, '*', n - 1 - j);
        if (!p) return n;
        j = (size_t)(p - s);
        if (s[j + 1] == '/') return j + 2;
        j++;
    }
    return n;
}

/* The preprocessor flag survives a newline only when the byte before it is a backslash.
 * Returns P after the newlines in [a, b). */
static inline bool preproc_through(const uint8_t *s, size_t a, size_t b, bool P) {
    while (P && a < b) {
        const uint8_t *p = memchr(s + a, '\n', b - a);
        if (!p) break;
        size_t t = (size_t)(p - s);
        P = t > 0 && s[t - 1] == '\\';
        a = t + 1;
    }
    return P;
}

/* ---- preprocessor lines --------------------------------------------------------------------- */
/* Called with P true at i (src[i] is the '#', or the first byte of a continued line).
 * Everything is LEX_PREPROC except string/char literals and comments, which keep their own
 * class. Returns the index at which P became false; *ls_out is the line-start flag there. */
static size_t scan_preproc(const uint8_t *src, size_t n, uint8_t *cls, size_t i, bool *ls_out) {
    for (;;) {
        size_t j = scan_pp_plain(src, i, n);
        if (j > i) { fill(cls, i, j - i, LEX_PREPROC, n); i = j; }
        if (i >= n) { *ls_out = false; return n; }
        const uint8_t c = src[i];
        if (c == '\n') {
            bool P = src[i - 1] == '\\';          /* i > 0: a '#' precedes any newline in P */
            cls[i] = P ? LEX_PREPROC : LEX_WS;
            i++;
            if (!P) { *ls_out = true; return i; }
        } else if (c == '"' || c == '\'') {
            j = scan_quoted(src, i, n, c);
            fill(cls, i, j - i, c == '"' ? LEX_STRING : LEX_CHAR, n);
            i = j;
        } else {   /* '/' */
            const uint8_t next = i + 1 < n ? src[i + 1] : 0;
            if (next == '*') {
                j = scan_block_comment(src, i, n);
                fill(cls, i, j - i, LEX_COMMENT, n);
                bool P = preproc_through(src, i + 2, j, true);
                i = j;
                if (!P) { *ls_out = false; return i; }
            } else if (next == '/') {
                const uint8_t *p = memchr(src + i, '\n', n - i);
                j = p ? (size_t)(p - src) : n;
                fill(cls, i, j - i, LEX_COMMENT, n);
                i = j;
            } else {
                cls[i] = LEX_PREPROC;
                i++;
            }
        }
    }
}

/* ---- blocks of ordinary code ------------------------------------------------------------------ */
/* Outside comments, literals and preprocessor lines the scanner works on blocks of 63 bytes
 * (64 are loaded; bit 63 of every mask is kept clear so that a run never fills a whole mask
 * and ctz is always defined). Bitmasks, one bit per byte, are computed with NEON. Only the
 * events that need per-token work are visited with ctz: starts of alnum runs (keyword,
 * function name, number) and the bytes that may open a span ( " ' / # ). Word classes are
 * collected as bitmasks and the whole block is painted afterwards in four vector stores. */
#define BLOCK 63

typedef struct {
    uint64_t alnum;     /* [A-Za-z0-9_]            */
    uint64_t digit;     /* [0-9]                   */
    uint64_t numc;      /* [A-Za-z0-9_.]           */
    uint64_t space;     /* ' ' \t \v \f \r         */
    uint64_t nl;        /* \n                      */
    uint64_t lparen;    /* (                       */
    uint64_t special;   /* " ' / #                 */
#if LEXNET_NEON
    uint8x16_t alnum_v[4], ws_v[4];   /* the same as vectors, for painting */
#endif
} Masks;

/* bits [p, p+len) for 1 <= len <= 64, p + len <= 64 */
static inline uint64_t bit_range(unsigned p, unsigned len) {
    return (~0ULL >> (64 - len)) << p;
}

/* length of the run of set bits at the bottom of x; x must not be all ones */
static inline unsigned run_len(uint64_t x) {
    return (unsigned)__builtin_ctzll(~x);
}

#if LEXNET_NEON
static inline uint64_t neon_movemask64(uint8x16_t p0, uint8x16_t p1, uint8x16_t p2, uint8x16_t p3) {
    const uint8x16_t bit = { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
    uint8x16_t s0 = vpaddq_u8(vandq_u8(p0, bit), vandq_u8(p1, bit));
    uint8x16_t s1 = vpaddq_u8(vandq_u8(p2, bit), vandq_u8(p3, bit));
    s0 = vpaddq_u8(s0, s1);
    s0 = vpaddq_u8(s0, s0);
    return vgetq_lane_u64(vreinterpretq_u64_u8(s0), 0);
}

/* masks for the 64 bytes at b */
static inline void classify_block(const uint8_t *b, Masks *m) {
    uint8x16_t digit[4], numc[4], space[4], nl[4], lparen[4], special[4];
    for (int k = 0; k < 4; k++) {
        uint8x16_t v = vld1q_u8(b + 16 * k);
        m->alnum_v[k] = neon_is_alnum(v);
        digit[k] = vcleq_u8(vsubq_u8(v, vdupq_n_u8('0')), vdupq_n_u8(9));
        numc[k] = vorrq_u8(m->alnum_v[k], vceqq_u8(v, vdupq_n_u8('.')));
        space[k] = neon_is_space(v);
        nl[k] = vceqq_u8(v, vdupq_n_u8('\n'));
        lparen[k] = vceqq_u8(v, vdupq_n_u8('('));
        m->ws_v[k] = vorrq_u8(space[k], nl[k]);
        special[k] = vorrq_u8(vorrq_u8(vceqq_u8(v, vdupq_n_u8('"')), vceqq_u8(v, vdupq_n_u8('\''))),
                              vorrq_u8(vceqq_u8(v, vdupq_n_u8('/')), vceqq_u8(v, vdupq_n_u8('#'))));
    }
    m->alnum = neon_movemask64(m->alnum_v[0], m->alnum_v[1], m->alnum_v[2], m->alnum_v[3]);
    m->digit = neon_movemask64(digit[0], digit[1], digit[2], digit[3]);
    m->numc = neon_movemask64(numc[0], numc[1], numc[2], numc[3]);
    m->space = neon_movemask64(space[0], space[1], space[2], space[3]);
    m->nl = neon_movemask64(nl[0], nl[1], nl[2], nl[3]);
    m->lparen = neon_movemask64(lparen[0], lparen[1], lparen[2], lparen[3]);
    m->special = neon_movemask64(special[0], special[1], special[2], special[3]);
}

/* lanes 0..15 of chunk q (bytes 16q..16q+15) of a 64-bit mask, as 0x00/0xFF */
static inline uint8x16_t expand16(uint8x16_t mask_bytes, int q) {
    const uint8x16_t bit = { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
    uint8x16_t idx = vdupq_n_u8((uint8_t)(2 * q));
    idx = vcombine_u8(vget_low_u8(idx), vadd_u8(vget_low_u8(idx), vdup_n_u8(1)));
    return vtstq_u8(vqtbl1q_u8(mask_bytes, idx), bit);
}

/* paint the 64 bytes at out: identifier / whitespace / punctuation from the masks, then the
 * four override classes (keyword, type, function, number) collected by the event loop */
static inline void paint_block(uint8_t *out, const Masks *m, const uint64_t *ov) {
    const uint8x16_t WS = vdupq_n_u8(LEX_WS), ID = vdupq_n_u8(LEX_IDENT), PU = vdupq_n_u8(LEX_PUNCT);
    uint8x16_t c[4];
    for (int q = 0; q < 4; q++) c[q] = vbslq_u8(m->alnum_v[q], ID, vbslq_u8(m->ws_v[q], WS, PU));
    static const uint8_t classes[4] = { LEX_KEYWORD, LEX_TYPE, LEX_FUNC, LEX_NUMBER };
    for (int t = 0; t < 4; t++) {
        if (!ov[t]) continue;
        uint8x16_t bytes = vreinterpretq_u8_u64(vdupq_n_u64(ov[t]));
        uint8x16_t k = vdupq_n_u8(classes[t]);
        for (int q = 0; q < 4; q++) c[q] = vbslq_u8(expand16(bytes, q), k, c[q]);
    }
    for (int q = 0; q < 4; q++) vst1q_u8(out + 16 * q, c[q]);
}
#else
static inline void classify_block(const uint8_t *b, Masks *m) {
    memset(m, 0, sizeof *m);
    for (int k = 0; k < 64; k++) {
        uint8_t f = cflags[b[k]], c = b[k];
        uint64_t bit = 1ULL << k;
        if (f & CF_ALNUM) m->alnum |= bit;
        if (c >= '0' && c <= '9') m->digit |= bit;
        if (f & CF_NUMC) m->numc |= bit;
        if (f & CF_SPACE) m->space |= bit;
        if (c == '\n') m->nl |= bit;
        if (c == '(') m->lparen |= bit;
        if (c == '"' || c == '\'' || c == '/' || c == '#') m->special |= bit;
    }
}
static inline void paint_block(uint8_t *out, const Masks *m, const uint64_t *ov) {
    for (int k = 0; k < 64; k++) {
        uint64_t bit = 1ULL << k;
        uint8_t c = (m->alnum & bit) ? LEX_IDENT : ((m->space | m->nl) & bit) ? LEX_WS : LEX_PUNCT;
        if (ov[0] & bit) c = LEX_KEYWORD;
        else if (ov[1] & bit) c = LEX_TYPE;
        else if (ov[2] & bit) c = LEX_FUNC;
        else if (ov[3] & bit) c = LEX_NUMBER;
        out[k] = c;
    }
}
#endif

static inline uint64_t bitrev64(uint64_t x) {
#if defined(__aarch64__)
    uint64_t r;
    __asm__("rbit %0, %1" : "=r"(r) : "r"(x));
    return r;
#else
    return __builtin_bitreverse64(x);
#endif
}

/* Propagate `seed` bits through runs of `run` bits towards higher positions: every run bit
 * at or above a seed bit (within the same run) is returned. */
static inline uint64_t fill_up(uint64_t run, uint64_t seed) {
    return ((run + (seed & run)) ^ run) & run;
}

/* ---- slow paths of the block scanner ------------------------------------------------------------- */
/* A span that ends a block: filled after the block is painted, the next block starts after it. */
typedef struct { size_t i, j; uint8_t k; int preproc; } Span;

/* a word or number run that reaches the end of the block; finish it with the byte scanners */
static __attribute__((noinline)) void finish_long_token(const uint8_t *src, size_t n, size_t i, size_t block_end,
                                                        int is_number, Span *sp) {
    if (is_number) {
        sp->i = i; sp->j = scan_number(src, block_end, n); sp->k = LEX_NUMBER;
        return;
    }
    size_t j = scan_word(src, block_end, n), mm = scan_spaces(src, j, n), len = j - i;
    uint64_t key1, key2;
    if (i + 16 <= n) kw_keys(src, i, (unsigned)(len < 64 ? len : 64), &key1, &key2);
    else kw_keys_slow(src + i, len, &key1, &key2);
    uint8_t k = kw_lookup(key1, key2);
    if (!k) k = (mm < n && src[mm] == '(') ? LEX_FUNC : LEX_IDENT;
    sp->i = i; sp->j = j; sp->k = k;
}

/* a word whose trailing spaces reach the block end: is the next non-space byte a '(' ? */
static __attribute__((noinline)) int paren_follows(const uint8_t *src, size_t n, size_t block_end) {
    size_t mm = scan_spaces(src, block_end, n);
    return mm < n && src[mm] == '(';
}

/* the span opened by the special byte c at i: comment, literal, or a lone '/' (returns 0) */
static __attribute__((noinline)) int open_span(const uint8_t *src, size_t n, size_t i, uint8_t c, Span *sp) {
    sp->i = i;
    if (c == '/') {
        const uint8_t nx = i + 1 < n ? src[i + 1] : 0;
        if (nx == '*') { sp->j = scan_block_comment(src, i, n); sp->k = LEX_COMMENT; return 1; }
        if (nx == '/') {
            const uint8_t *q = memchr(src + i, '\n', n - i);
            sp->j = q ? (size_t)(q - src) : n; sp->k = LEX_COMMENT; return 1;
        }
        return 0;
    }
    sp->j = scan_quoted(src, i, n, c);
    sp->k = c == '"' ? LEX_STRING : LEX_CHAR;
    return 1;
}

/* ---- the event loop ----------------------------------------------------------------------- */
typedef struct {
    uint64_t lsq;           /* bit p: the byte before p is at line start                     */
    uint64_t paren_ahead;   /* bit p: first non-space byte at or after p is '('              */
    uint64_t space_tail;    /* bit p: p is in the space run that reaches the end of the block */
    uint64_t ov[4];         /* out: override bits for keyword, type, function, number        */
    Span span;              /* out: the span that ended the block, if any                    */
    int have_span;          /* out                                                           */
    size_t next;            /* out: where the next block starts                              */
    bool next_ls;           /* out: line-start flag of the byte before `next`                */
} BlockState;

/* Visit the events of one block. Kept out of line so that the hot loop is register-allocated
 * on its own, without the NEON constants of classify/paint. */
static __attribute__((noinline)) void scan_events(const uint8_t *src, size_t n, size_t pos, size_t rem,
                                                  const Masks *m, BlockState *st) {
    uint64_t ov_kw = 0, ov_ty = 0, ov_fn = 0, ov_num = 0;
    const uint64_t alnum = m->alnum, digit = m->digit, numc = m->numc;
    const uint64_t lsq = st->lsq, paren_ahead = st->paren_ahead, space_tail = st->space_tail;
    st->have_span = 0;
    st->span.preproc = 0;
    st->next = pos + rem;

    /* events: starts of alnum runs, and the span openers */
    uint64_t events = (alnum & ~(alnum << 1)) | m->special;
    while (events) {
        const unsigned p = (unsigned)__builtin_ctzll(events);
        const size_t i = pos + p;
        unsigned end;                       /* first byte after this token, block-relative */
        if ((alnum >> p) & 1) {
            const int is_number = (digit >> p) & 1;
            const unsigned len = run_len((is_number ? numc : alnum) >> p);
            end = p + len;
            if (end >= rem && rem == BLOCK) {
                /* the run reaches the block end: finish it with the byte scanners */
                finish_long_token(src, n, i, pos + BLOCK, is_number, &st->span);
                st->have_span = 1; st->next = st->span.j; st->next_ls = false;
                break;
            }
            const uint64_t bits = bit_range(p, len);
            if (is_number) {
                ov_num |= bits;
            } else {
                /* word: keyword / type / constant / function name / identifier */
                uint64_t key1, key2;
                if (i + 16 <= n) kw_keys(src, i, len, &key1, &key2);
                else kw_keys_slow(src + i, len, &key1, &key2);
                uint8_t k = kw_lookup(key1, key2);
                int fn = (paren_ahead >> end) & 1;
                if ((space_tail >> end) & 1) fn = paren_follows(src, n, pos + rem);
                k = k ? k : (fn ? LEX_FUNC : LEX_IDENT);
                ov_kw |= k == LEX_KEYWORD ? bits : 0;
                ov_ty |= k == LEX_TYPE ? bits : 0;
                ov_fn |= k == LEX_FUNC ? bits : 0;
                ov_num |= k == LEX_NUMBER ? bits : 0;
            }
        } else {
            const uint8_t c = src[i];
            if (c == '#') {
                if ((lsq >> p) & 1) {
                    st->span.i = i; st->span.preproc = 1;
                    st->have_span = 1;
                    break;
                }
                end = p + 1;                       /* '#' not at line start: punctuation */
            } else if (open_span(src, n, i, c, &st->span)) {
                st->have_span = 1; st->next = st->span.j; st->next_ls = false;
                break;
            } else {
                end = p + 1;                       /* a lone '/' is punctuation */
            }
        }
        events &= ~0ULL << end;             /* end <= 63 here */
    }
    st->ov[0] = ov_kw; st->ov[1] = ov_ty; st->ov[2] = ov_fn; st->ov[3] = ov_num;
}

/* ---- the scanner ------------------------------------------------------------------------- */
void lexnet_fast(const uint8_t *src, size_t n, uint8_t *cls) {
    if (!tables_ready) init_tables();

    size_t pos = 0;
    bool ls = true;      /* line-start flag of the byte before pos (network unit ls, h0: 1) */
    while (pos < n) {
        /* --- one block of ordinary code: bytes [pos, pos + rem) --- */
        size_t rem = n - pos;
        Masks m;
        uint8_t tmp[64];
        if (rem >= 64) {
            classify_block(src + pos, &m);
            rem = BLOCK;
        } else {
            memset(tmp, 0, sizeof tmp);
            memcpy(tmp, src + pos, rem);
            classify_block(tmp, &m);
            if (rem > BLOCK) rem = BLOCK;
        }
        const uint64_t valid = ~0ULL >> (64 - rem);
        m.alnum &= valid; m.digit &= valid; m.numc &= valid; m.space &= valid;
        m.nl &= valid; m.lparen &= valid; m.special &= valid;

        BlockState st;
        /* line-start bits: a byte is "ls" if it is a newline, or a space whose left neighbour is
         * ls; `lsq` bit p says the byte before p is ls (what a '#' at p needs to open a # line) */
        const uint64_t after = (m.nl << 1) | (ls ? 1 : 0);
        const uint64_t lsbits = m.nl | fill_up(m.space, after);
        st.lsq = (lsbits << 1) | (ls ? 1 : 0);

        /* function-name look-ahead for the whole block at once: bit p of `paren_ahead` says
         * that the first non-space byte at or after p is a '('. Spaces that run into the block
         * end (`space_tail`) cannot know, those take the slow path. */
        const uint64_t r_space = bitrev64(m.space), r_lparen = bitrev64(m.lparen);
        st.paren_ahead = m.lparen | bitrev64(fill_up(r_space, r_lparen << 1));
        const uint64_t rs = r_space >> (64 - rem);                    /* bit 0 = last valid byte */
        st.space_tail = bitrev64(((rs & 1) ? (rs & ((rs + 1) ^ rs)) : 0) << (64 - rem));
        st.next_ls = (lsbits >> (rem - 1)) & 1;

        scan_events(src, n, pos, rem, &m, &st);

        /* paint the block, then the span that ended it (the next block repaints from `next`) */
        if (rem == BLOCK && pos + 64 <= n) {
            paint_block(cls + pos, &m, st.ov);
        } else {
            paint_block(tmp, &m, st.ov);
            memcpy(cls + pos, tmp, rem);
        }
        if (st.have_span) {
            if (st.span.preproc) st.next = scan_preproc(src, n, cls, st.span.i, &st.next_ls);
            else memset(cls + st.span.i, st.span.k, st.span.j - st.span.i);
        }
        pos = st.next; ls = st.next_ls;
    }
}
