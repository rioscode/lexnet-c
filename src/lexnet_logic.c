/* lexnet_logic.c - Lexnet transcribed unit by unit into an if/else program.
 *
 * Every line below corresponds to one unit of the network (see model/lexnet_kernel.h and
 * docs/network-to-logic.md for the mapping). A unit "clip1(b + sum w_i * v_i)" over 0/1
 * inputs is a threshold gate; with the invariants proven in docs/network-to-logic.md
 * (block comment / line comment / string / char literal are mutually exclusive, word and
 * number are mutually exclusive) each gate collapses to the boolean expression written here.
 *
 * Layout of the network, and of this file:
 *   1. 14 per-byte features (the embedding table)            -> feat_of()
 *   2. forward recurrent cell, 18 state units, 6 layers       -> fwd_step()
 *      - layer 0: "continue" gates and the word-length counter
 *      - layer 1: mode flags, word/number flags, hash gates
 *      - layer 2: the two base-64 hash accumulators h1 (chars 0..7), h2 (chars 8..13)
 *      - layer 3-5: keyword / type / constant match on (h1, h2)
 *   3. backward recurrent cell, 7 state units, 2 layers       -> bwd_step()
 *   4. linear head, argmax                                    -> head()
 */
#include "lexnet.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* ---- 1. features ---------------------------------------------------------------- */
/* The embedding table maps a byte (>= 128 -> all zero) to 13 indicator features plus a
 * 6-bit character code used by the hash accumulators. */
enum {
    F_ALPHA  = 1 << 0,   /* x[0]  [A-Za-z_]                       */
    F_DIGIT  = 1 << 1,   /* x[1]  [0-9]                           */
    F_ALNUM  = 1 << 2,   /* x[2]  [A-Za-z0-9_]                    */
    F_NUMC   = 1 << 3,   /* x[3]  [A-Za-z0-9_.]  number continuation */
    F_SPACE  = 1 << 4,   /* x[4]  ' ' \t \v \f \r                 */
    F_NL     = 1 << 5,   /* x[5]  \n                              */
    F_DQ     = 1 << 6,   /* x[6]  "                               */
    F_SQ     = 1 << 7,   /* x[7]  '                               */
    F_BS     = 1 << 8,   /* x[8]  backslash                       */
    F_SLASH  = 1 << 9,   /* x[9]  /                               */
    F_STAR   = 1 << 10,  /* x[10] *                               */
    F_HASH   = 1 << 11,  /* x[11] #                               */
    F_LPAREN = 1 << 12,  /* x[12] (                               */
};

typedef struct { uint16_t f; uint8_t code; } Feat;

/* x[13]: a=1..z=26, A=27..Z=52, 0=53..9=62, _=63, everything else 0. */
static uint8_t code_of(unsigned c) {
    if (c >= 'a' && c <= 'z') return (uint8_t)(c - 'a' + 1);
    if (c >= 'A' && c <= 'Z') return (uint8_t)(c - 'A' + 27);
    if (c >= '0' && c <= '9') return (uint8_t)(c - '0' + 53);
    if (c == '_') return 63;
    return 0;
}

static Feat feat_of(unsigned c) {
    Feat r = { 0, 0 };
    if (c >= 128) return r;                                  /* net maps these to a null row */
    bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    bool digit = c >= '0' && c <= '9';
    if (alpha) r.f |= F_ALPHA;
    if (digit) r.f |= F_DIGIT;
    if (alpha || digit) r.f |= F_ALNUM | F_NUMC;
    if (c == '.') r.f |= F_NUMC;
    if (c == ' ' || c == '\t' || c == '\v' || c == '\f' || c == '\r') r.f |= F_SPACE;
    if (c == '\n') r.f |= F_NL;
    if (c == '"') r.f |= F_DQ;
    if (c == '\'') r.f |= F_SQ;
    if (c == '\\') r.f |= F_BS;
    if (c == '/') r.f |= F_SLASH;
    if (c == '*') r.f |= F_STAR;
    if (c == '#') r.f |= F_HASH;
    if (c == '(') r.f |= F_LPAREN;
    r.code = code_of(c);
    return r;
}

/* ---- 2. forward cell -------------------------------------------------------------- */
/* FState in the kernel, same field order. h1/h2 are the "accumulator" units (ceiling inf),
 * d is the "counter" unit (ceiling 14), everything else is a 0/1 logic unit. */
typedef struct {
    bool C;        /* inside a block comment                              */
    bool L;        /* inside a line comment                               */
    bool S;        /* inside a string literal                             */
    bool Q;        /* inside a char literal                               */
    bool P;        /* inside a preprocessor line                          */
    bool E;        /* this byte is a backslash that escapes the next one  */
    bool sf;       /* this byte is a '/' that may open a comment          */
    bool star_in;  /* this byte is a '*' inside a block comment           */
    bool closed;   /* this byte is the '/' of a closing "* /"             */
    bool Sclose;   /* this byte is the closing '"'                        */
    bool Qclose;   /* this byte is the closing '\''                       */
    bool ls;       /* only whitespace seen since the last newline         */
    bool bsP;      /* backslash inside a preprocessor line                */
    bool W;        /* this byte belongs to a word (identifier)            */
    bool N;        /* this byte belongs to a pp-number                    */
    int  d;        /* 14 - (index of this byte in its word), floored at 0 */
    uint64_t h1;   /* base-64 hash of word bytes 0..7                     */
    uint64_t h2;   /* base-64 hash of word bytes 8..13                    */
} FState;

/* The 15 bits the forward pass hands to the backward pass (f_out order). */
enum {
    B_C = 1 << 0, B_L = 1 << 1, B_S = 1 << 2, B_Q = 1 << 3, B_P = 1 << 4, B_W = 1 << 5,
    B_N = 1 << 6, B_KW = 1 << 7, B_TY = 1 << 8, B_CONST = 1 << 9, B_COPEN2 = 1 << 10,
    B_SLASH = 1 << 11, B_SPACE = 1 << 12, B_NL = 1 << 13, B_LPAREN = 1 << 14,
};

/* Keyword tables: layers 3..5 compare h1 (and h2) against the hashes of these words.
 * "short" words (<= 8 bytes) additionally require h2 == 0, i.e. the word really ends there.
 * "long" words (9..14 bytes) require h1 == hash(bytes 0..7) and h2 == hash(bytes 8..13);
 * bytes 14 and beyond are never hashed, so "_Static_assert" also matches "_Static_assertXYZ". */
static const char *const KW_SHORT[] = {
    "auto", "break", "case", "const", "continue", "default", "do", "else", "enum", "extern",
    "for", "goto", "if", "inline", "register", "restrict", "return", "sizeof", "static",
    "struct", "switch", "typedef", "union", "volatile", "while", "_Alignas", "_Alignof",
    "_Atomic", "_Generic", "alignas", "alignof", "typeof",
};
static const char *const KW_LONG[] = {
    "_Noreturn", "_Static_assert", "_Thread_local", "constexpr", "static_assert", "thread_local",
};
static const char *const TY_SHORT[] = {
    "char", "double", "float", "int", "long", "short", "signed", "unsigned", "void", "bool",
    "_Bool", "_Complex", "size_t", "ssize_t", "intptr_t", "wchar_t", "time_t", "off_t", "pid_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "FILE", "va_list",
};
static const char *const TY_LONG[] = { "ptrdiff_t", "uintptr_t" };
static const char *const CONST_SHORT[] = { "NULL", "nullptr", "true", "false", "EOF" };

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

typedef struct { uint64_t h1, h2; } Hash2;

static Hash2 hash_word(const char *w) {
    Hash2 h = { 0, 0 };
    for (int i = 0; w[i]; i++) {
        if (i < 8) h.h1 = h.h1 * 64 + code_of((unsigned char)w[i]);
        else if (i < 14) h.h2 = h.h2 * 64 + code_of((unsigned char)w[i]);
    }
    return h;
}

static Hash2 kw_short[COUNT(KW_SHORT)], kw_long[COUNT(KW_LONG)], ty_short[COUNT(TY_SHORT)],
             ty_long[COUNT(TY_LONG)], const_short[COUNT(CONST_SHORT)];
static Feat  feat[256];
static bool  tables_ready;

/* built once at load time; the lazy check in lexnet_logic() is only a fallback */
static __attribute__((constructor)) void init_tables(void) {
    for (size_t i = 0; i < COUNT(KW_SHORT); i++) kw_short[i] = hash_word(KW_SHORT[i]);
    for (size_t i = 0; i < COUNT(KW_LONG); i++) kw_long[i] = hash_word(KW_LONG[i]);
    for (size_t i = 0; i < COUNT(TY_SHORT); i++) ty_short[i] = hash_word(TY_SHORT[i]);
    for (size_t i = 0; i < COUNT(TY_LONG); i++) ty_long[i] = hash_word(TY_LONG[i]);
    for (size_t i = 0; i < COUNT(CONST_SHORT); i++) const_short[i] = hash_word(CONST_SHORT[i]);
    for (unsigned c = 0; c < 256; c++) feat[c] = feat_of(c);
    tables_ready = true;
}

static bool match_short(const Hash2 *t, size_t n, uint64_t h1, uint64_t h2) {
    if (h2 != 0) return false;                      /* layer 4: "- h2" term  */
    for (size_t i = 0; i < n; i++) if (t[i].h1 == h1) return true;   /* ge && le pair */
    return false;
}
static bool match_long(const Hash2 *t, size_t n, uint64_t h1, uint64_t h2) {
    for (size_t i = 0; i < n; i++) if (t[i].h1 == h1 && t[i].h2 == h2) return true;
    return false;
}

/* One forward step: byte features x, previous state p -> new state s, returns the 15 bits. */
static uint16_t fwd_step(Feat x, const FState *p, FState *s) {
    const bool alpha = x.f & F_ALPHA, digit = x.f & F_DIGIT, alnum = x.f & F_ALNUM;
    const bool numc = x.f & F_NUMC, space = x.f & F_SPACE, nl = x.f & F_NL;
    const bool dq = x.f & F_DQ, sq = x.f & F_SQ, bs = x.f & F_BS, slash = x.f & F_SLASH;
    const bool star = x.f & F_STAR, hash = x.f & F_HASH, lparen = x.f & F_LPAREN;

    /* layer 0: does the previous mode continue into this byte? */
    const bool contC = p->C && !p->closed;              /* block comment not closed yet   */
    const bool contL = p->L && !nl;                     /* line comment ends at newline   */
    const bool contS = p->S && !nl && !p->Sclose;       /* string ends at newline or "    */
    const bool contQ = p->Q && !nl && !p->Qclose;       /* char literal likewise          */
    const bool contP = p->P && (!nl || p->bsP);         /* # line ends at newline unless \ */
    const bool opened = star && p->sf;                  /* the '*' of "/*"                 */
    const bool lo = slash && p->sf;                     /* the second '/' of "//"          */
    const bool closed = slash && p->star_in;            /* the '/' of "* /"                */
    const bool ls = nl || (space && p->ls);             /* still at line start             */
    const bool wcont = alnum && p->W;                   /* word continues                  */
    const bool wstart0 = alpha && !p->N;                /* word may start here             */
    const bool nany = digit || p->N;                    /* number starts or continues      */
    const int  d = p->W ? (p->d > 0 ? p->d - 1 : 0) : 14;   /* counter: 14 - index in word */
    const bool lt8 = !p->W || p->d >= 8;                /* index of this byte in word < 8  */
    const bool ge8 = p->W && p->d <= 7;                 /* index >= 8                      */
    const bool ge14 = p->W && p->d <= 1;                /* index >= 14                     */
    const bool inMode = contC || contL || contS || contQ;   /* at most one is set          */

    /* layer 1: new mode flags */
    const bool C = contC || opened;
    const bool L = contL || lo;
    const bool copen2 = opened || lo;                   /* second byte of a comment opener */
    const bool S = contS || (dq && !inMode);
    const bool Q = contQ || (sq && !inMode);
    const bool P = contP || (hash && p->ls && !inMode);
    const bool Sclose = contS && dq && !p->E;
    const bool Qclose = contQ && sq && !p->E;
    const bool E = (contS || contQ) && bs && !p->E;
    const bool star_in = star && contC;
    const bool sf = slash && !inMode && !lo;
    const bool bsP = bs && contP;
    const bool W = (wcont || wstart0) && !inMode;
    const bool N = nany && numc && !wcont && !wstart0 && !inMode;
    /* hash gates (the 2^49-scaled units): keep / reset / hold the accumulators */
    const uint64_t g1 = (lt8 && alnum && p->W) ? p->h1 : 0;
    const uint64_t c1 = lt8 ? x.code : 0;
    const uint64_t hold1 = lt8 ? 0 : p->h1;
    const uint64_t g2 = (ge8 && !ge14 && alnum) ? p->h2 : 0;
    const uint64_t c2 = (ge8 && !ge14) ? x.code : 0;
    const uint64_t hold2 = (p->W && ge8 == ge14) ? p->h2 : 0;

    /* layer 2: accumulators */
    const uint64_t h1 = 64 * g1 + c1 + hold1;
    const uint64_t h2 = 64 * g2 + c2 + hold2;

    /* layers 3..5: keyword tables */
    const bool KW = match_short(kw_short, COUNT(kw_short), h1, h2) ||
                    match_long(kw_long, COUNT(kw_long), h1, h2);
    const bool TY = match_short(ty_short, COUNT(ty_short), h1, h2) ||
                    match_long(ty_long, COUNT(ty_long), h1, h2);
    const bool CONST = match_short(const_short, COUNT(const_short), h1, h2);

    s->C = C; s->L = L; s->S = S; s->Q = Q; s->P = P; s->E = E; s->sf = sf;
    s->star_in = star_in; s->closed = closed; s->Sclose = Sclose; s->Qclose = Qclose;
    s->ls = ls; s->bsP = bsP; s->W = W; s->N = N; s->d = d; s->h1 = h1; s->h2 = h2;

    return (uint16_t)((C ? B_C : 0) | (L ? B_L : 0) | (S ? B_S : 0) | (Q ? B_Q : 0) |
                      (P ? B_P : 0) | (W ? B_W : 0) | (N ? B_N : 0) | (KW ? B_KW : 0) |
                      (TY ? B_TY : 0) | (CONST ? B_CONST : 0) | (copen2 ? B_COPEN2 : 0) |
                      (slash ? B_SLASH : 0) | (space ? B_SPACE : 0) | (nl ? B_NL : 0) |
                      (lparen ? B_LPAREN : 0));
}

/* ---- 3. backward cell ------------------------------------------------------------- */
typedef struct {
    bool W;       /* the byte to the right is a word byte                                */
    bool copen2;  /* the byte to the right is the second byte of a comment opener        */
    bool lps;     /* a '(' follows, with only spaces in between                          */
    bool kwc;     /* the word this byte belongs to is a keyword                          */
    bool tyc;     /* ... a type                                                          */
    bool cst;     /* ... a constant (NULL, true, ...)                                    */
    bool fnc;     /* ... a function name (followed by optional spaces and '(')           */
} BState;

/* One backward step: forward bits of this byte, state r of the byte to the right -> new state
 * s, returns the class id of this byte. */
static int bwd_step(uint16_t b, const BState *r, BState *s) {
    const bool C = b & B_C, L = b & B_L, S = b & B_S, Q = b & B_Q, P = b & B_P, W = b & B_W;
    const bool N = b & B_N, KW = b & B_KW, TY = b & B_TY, CONST = b & B_CONST;
    const bool slash = b & B_SLASH, space = b & B_SPACE, nl = b & B_NL, lparen = b & B_LPAREN;

    /* layer 0 */
    const bool lps = lparen || (space && r->lps);
    const bool cmt = C || L || (slash && r->copen2);          /* first '/' of an opener too */
    const bool kwc_end = W && KW && !r->W;                    /* last byte of a keyword     */
    const bool kwc_carry = W && r->W && r->kwc;               /* propagate to the left      */
    const bool tyc_end = W && TY && !r->W;
    const bool tyc_carry = W && r->W && r->tyc;
    const bool cst_end = W && CONST && !r->W;
    const bool cst_carry = W && r->W && r->cst;
    const bool fnc_end = W && r->lps && !KW && !TY && !CONST && !r->W;
    const bool fnc_carry = W && r->W && r->fnc;
    /* layer 1 */
    const bool kwc = kwc_end || kwc_carry;
    const bool tyc = tyc_end || tyc_carry;
    const bool cst = cst_end || cst_carry;
    const bool fnc = fnc_end || fnc_carry;

    s->W = W; s->copen2 = b & B_COPEN2; s->lps = lps;
    s->kwc = kwc; s->tyc = tyc; s->cst = cst; s->fnc = fnc;

    /* ---- 4. head: argmax over 11 logits, lowest id wins ties ----
     *   0: 0.5   1: 7*kwc   2: 7*tyc   3: 7*fnc   4: 5*W   5: 6*(cst+N)
     *   6: 10*S  7: 10*Q    8: 9*cmt   9: 8*P     10: 1 - space - nl               */
    if (S) return LEX_STRING;
    if (Q) return LEX_CHAR;
    if (cmt) return LEX_COMMENT;
    if (P) return LEX_PREPROC;
    if (kwc) return LEX_KEYWORD;
    if (tyc) return LEX_TYPE;
    if (fnc) return LEX_FUNC;
    if (cst || N) return LEX_NUMBER;
    if (W) return LEX_IDENT;
    if (!space && !nl) return LEX_PUNCT;
    return LEX_WS;
}

/* ---- driver ------------------------------------------------------------------------ */
void lexnet_logic(const uint8_t *src, size_t n, uint8_t *cls) {
    if (!tables_ready) init_tables();

    /* The backward pass needs the forward bits of every byte, so they are kept in a
     * temporary buffer, exactly like the reference kernel does. */
    uint16_t stack_bits[4096];
    uint16_t *fb = n <= 4096 ? stack_bits : (uint16_t *)malloc(n * sizeof(uint16_t));

    FState a, b;
    memset(&a, 0, sizeof a);
    a.ls = 1;            /* h0: ls = 1 (start of file counts as line start) */
    a.d = 14;            /* h0: d = 14                                        */
    FState *p = &a, *s = &b;
    for (size_t i = 0; i < n; i++) {
        fb[i] = fwd_step(feat[src[i]], p, s);
        FState *t = p; p = s; s = t;
    }
    BState ra, rb;
    memset(&ra, 0, sizeof ra);   /* backward state right of the last byte is all zero */
    BState *r = &ra, *q = &rb;
    for (size_t i = n; i-- > 0;) {
        cls[i] = (uint8_t)bwd_step(fb[i], r, q);
        BState *t = r; r = q; q = t;
    }
    if (fb != stack_bits) free(fb);
}
