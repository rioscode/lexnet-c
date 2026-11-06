/* difftest.c - differential test: reference network vs. lexnet_logic vs. lexnet_fast.
 *
 * The reference is the original lexnet.c + lexnet_kernel.h from the Hugging Face repo,
 * included here with its main() renamed, so the comparison runs in-process against the very
 * same compiled kernel the model ships (and, for a sample, against its float64 interpreter).
 *
 * Inputs:
 *   1. every string of length 1..EXH_LEN over a 16-symbol alphabet that covers all feature classes
 *   2. random "token soup" built from keywords, near-keywords, numbers, strings, comments, ...
 *   3. uniformly random bytes
 *   4. the files named on the command line
 *
 * usage: difftest [--quick] [--seed N] [file ...]
 */
#define main lexnet_ref_main
#include "../model/lexnet.c"
#undef main

#include "../src/lexnet.h"

typedef void (*impl_fn)(const uint8_t *, size_t, uint8_t *);
typedef struct { const char *name; impl_fn fn; } Impl;

static Net net;
static double kc[K_N_COEF], kb[K_N_BIAS];
static long long total_bytes, total_cases, failures;

static void escape_print(const uint8_t *s, size_t n, size_t mark) {
    for (size_t i = 0; i < n; i++) {
        if (i == mark) fputs("[[", stderr);
        unsigned c = s[i];
        if (c == '\n') fputs("\\n", stderr);
        else if (c == '\t') fputs("\\t", stderr);
        else if (c == '\\') fputs("\\\\", stderr);
        else if (c >= 32 && c < 127) fputc((int)c, stderr);
        else fprintf(stderr, "\\x%02x", c);
        if (i == mark) fputs("]]", stderr);
    }
    fputc('\n', stderr);
}

static int check_one(const Impl *impls, int n_impl, const uint8_t *src, size_t n,
                     uint8_t *ref, uint8_t *got, uint16_t *scratch, const char *what) {
    classify_compiled(&net, kc, kb, src, n, ref, scratch);
    total_bytes += (long long)n; total_cases++;
    for (int k = 0; k < n_impl; k++) {
        memset(got, 0xEE, n);
        impls[k].fn(src, n, got);
        for (size_t i = 0; i < n; i++) {
            if (got[i] != ref[i]) {
                failures++;
                fprintf(stderr, "MISMATCH %s (%s) at byte %zu of %zu: ref=%d got=%d\n",
                        impls[k].name, what, i, n, ref[i], got[i]);
                size_t a = i > 60 ? i - 60 : 0, b = i + 40 < n ? i + 40 : n;
                fputs("  src: ", stderr); escape_print(src + a, b - a, i - a);
                fputs("  ref: ", stderr);
                for (size_t j = a; j < b; j++) fputc("0123456789A"[ref[j]], stderr);
                fputc('\n', stderr);
                fputs("  got: ", stderr);
                for (size_t j = a; j < b; j++) fputc("0123456789A"[got[j]], stderr);
                fputc('\n', stderr);
                return 0;
            }
        }
    }
    return 1;
}

/* ---- exhaustive short strings ---- */
static const uint8_t ALPHABET[] = { 'a', 'Z', '_', '1', '.', ' ', '\n', '"', '\'', '\\', '/', '*', '#', '(', ';', 0xC3 };
#define NALPHA (sizeof ALPHABET)

static void exhaustive(const Impl *impls, int n_impl, int maxlen, uint8_t *ref, uint8_t *got, uint16_t *scratch) {
    uint8_t s[16];
    for (int len = 1; len <= maxlen; len++) {
        unsigned long long count = 1;
        for (int i = 0; i < len; i++) count *= NALPHA;
        for (unsigned long long idx = 0; idx < count; idx++) {
            unsigned long long t = idx;
            for (int i = 0; i < len; i++) { s[i] = ALPHABET[t % NALPHA]; t /= NALPHA; }
            if (!check_one(impls, n_impl, s, (size_t)len, ref, got, scratch, "exhaustive")) {
                if (failures > 20) exit(1);
            }
        }
        fprintf(stderr, "  exhaustive len %d: %llu strings ok\n", len, count);
    }
}

/* ---- token soup ---- */
static const char *TOKENS[] = {
    /* keywords, types, constants and near misses */
    "auto", "break", "case", "const", "continue", "default", "do", "else", "enum", "extern", "for",
    "goto", "if", "inline", "register", "restrict", "return", "sizeof", "static", "struct", "switch",
    "typedef", "union", "volatile", "while", "_Alignas", "_Alignof", "_Atomic", "_Generic", "alignas",
    "alignof", "typeof", "_Noreturn", "_Static_assert", "_Thread_local", "constexpr", "static_assert",
    "thread_local", "char", "double", "float", "int", "long", "short", "signed", "unsigned", "void",
    "bool", "_Bool", "_Complex", "size_t", "ssize_t", "intptr_t", "wchar_t", "time_t", "off_t", "pid_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "FILE",
    "va_list", "ptrdiff_t", "uintptr_t", "NULL", "nullptr", "true", "false", "EOF",
    "iff", "i", "whilee", "whil", "_Static_assertion", "_Static_assert1", "_Static_asser", "_Noreturnn",
    "_Noretur", "constexp", "constexprr", "thread_locall", "ptrdiff_tt", "uintptr_", "uintptr_tx",
    "registe", "registers", "unsignedd", "NULLs", "EOFF", "Null", "TRUE", "If", "INT", "Int",
    /* identifiers of interesting lengths */
    "a", "ab", "abcdefg", "abcdefgh", "abcdefghi", "abcdefghijklm", "abcdefghijklmn", "abcdefghijklmno",
    "abcdefghijklmnopqrstuvwxyz", "_", "__", "x1", "_1", "A_B_C", "foo_bar", "n", "i", "j", "buf",
    "printf", "main", "argv", "memcpy", "size", "len", "ptr", "value", "state",
    /* numbers */
    "0", "1", "42", "0x1F", "0X1f", "1.5", "1.5e-3", "1e10", "123abc", ".5", "1.", "0b101", "0777",
    "1u", "1UL", "1.0f", "3.14159", "1'000", "0x.p1", "1..2", "1.2.3", "9z", "00", "123_456",
    /* strings and chars */
    "\"\"", "\"abc\"", "\"a\\\"b\"", "\"\\\\\"", "\"\\\\\\\"\"", "\"abc", "\"a\\\nb\"", "\"\\\"", "\"/* no */\"",
    "\"// no\"", "\"#no\"", "\"\\n\"", "\"'\"", "\"\\'\"", "'a'", "'\\''", "'\\\\'", "'\"'", "'ab", "''", "'",
    "\"", "'\\n'", "'/*'", "\"\\\\\\\\\"", "'\\",
    /* comments */
    "/**/", "/***/", "/*/", "/* a */", "/* a\n b */", "/*\n#define X\n*/", "/* \\\n */", "/*/ x */",
    "/* /* nested */", "/* a ** b **/", "// line", "// line \\", "///", "//*", "// /* x", "/ /", "/", "*/",
    "*", "/*", "// \"str\"", "/* \"str\" */", "// 'c'", "//\\",
    /* preprocessor */
    "#define X 1", "#include \"x.h\"", "#include <x.h>", "# if", "#", "##", "#define X \\", "#define Y(a) (a)",
    "#endif", "#error \"x\"", "#define S \"abc\\", "#define C /* c", "#define D // c \\", "#pragma once",
    /* punctuation and function-call shapes */
    "(", ")", "( ", " (", "(x)", "{", "}", "[", "]", ";", ",", "->", ".", "...", "+", "-", "*", "&", "|",
    "^", "~", "!", "=", "<", ">", "?", ":", "%", "/", "\\", "@", "$", "`", "f(", "f (", "f\t(", "f\n(",
    "f /**/ (", "if(", "if (", "int(", "NULL(", "sizeof(", "foo  (", "foo(bar(baz))", "x = f(1, g(2));",
    /* whitespace and odd bytes */
    " ", "  ", "\t", "\n", "\r\n", "\r", "\v", "\f", "\n\n", "\\\n", "\\\r\n", "\x80", "\xff", "\xc3\xa4",
    "\x01", "\x7f", "\0",
};
#define NTOKENS (sizeof TOKENS / sizeof TOKENS[0])
static const char *SEPS[] = { "", " ", " ", "  ", "\n", "\t", "\n\n", "\r\n", " \n ", "\\\n" };
#define NSEPS (sizeof SEPS / sizeof SEPS[0])

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rnd(void) {   /* xorshift64* */
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1DULL;
}

static size_t gen_soup(uint8_t *buf, size_t cap) {
    size_t n = 0, target = 1 + rnd() % cap;
    while (n < target) {
        const char *t = TOKENS[rnd() % NTOKENS];
        size_t tl = t[0] ? strlen(t) : 1;      /* "\0" token is one NUL byte */
        if (n + tl + 4 > cap) break;
        memcpy(buf + n, t, tl); n += tl;
        const char *s = SEPS[rnd() % NSEPS];
        size_t sl = strlen(s);
        memcpy(buf + n, s, sl); n += sl;
    }
    return n;
}

static size_t gen_random(uint8_t *buf, size_t cap, int biased) {
    size_t n = 1 + rnd() % cap;
    for (size_t i = 0; i < n; i++) buf[i] = biased ? ALPHABET[rnd() % NALPHA] : (uint8_t)rnd();
    return n;
}

static void check_file(const Impl *impls, int n_impl, const char *path, uint8_t *ref, uint8_t *got, uint16_t *scratch, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return; }
    uint8_t *src = malloc(cap);
    size_t n = fread(src, 1, cap, f);
    fclose(f);
    if (!check_one(impls, n_impl, src, n, ref, got, scratch, path)) fprintf(stderr, "  in file %s\n", path);
    /* also compare the compiled kernel against the float64 interpreter on real files */
    double *feats = malloc((n + 1) * (size_t)net.n_fout * sizeof(double));
    uint8_t *interp = malloc(n + 1);
    classify(&net, src, n, interp, feats);
    classify_compiled(&net, kc, kb, src, n, ref, scratch);
    for (size_t i = 0; i < n; i++) if (interp[i] != ref[i]) {
        fprintf(stderr, "REFERENCE INCONSISTENCY: kernel vs interpreter differ in %s at %zu\n", path, i);
        failures++; break;
    }
    free(feats); free(interp); free(src);
}

int main(int argc, char **argv) {
    int quick = 0, exh_len = 6;
    long soup_rounds = 20000, rand_rounds = 20000;
    const char *files[4096]; int n_files = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--seed")) rng_state = strtoull(argv[++i], NULL, 0) | 1;
        else if (n_files < 4096) files[n_files++] = argv[i];
    }
    if (quick) { exh_len = 4; soup_rounds = 2000; rand_rounds = 2000; }

    net = load("model/lexnet_weights.bin");
    if (struct_hash(&net) != K_STRUCT_HASH) { fprintf(stderr, "kernel/weights hash mismatch\n"); return 1; }
    gather(&net, kc, kb);

    const Impl impls[] = { { "logic", lexnet_logic }, { "fast", lexnet_fast } };
    const int n_impl = 2;

    const size_t CAP = 8u << 20;
    uint8_t *buf = malloc(CAP), *ref = malloc(CAP), *got = malloc(CAP);
    uint16_t *scratch = malloc(CAP * sizeof(uint16_t));

    fprintf(stderr, "[1/4] files\n");
    for (int i = 0; i < n_files; i++) check_file(impls, n_impl, files[i], ref, got, scratch, CAP);

    fprintf(stderr, "[2/4] exhaustive strings up to length %d over %zu symbols\n", exh_len, NALPHA);
    exhaustive(impls, n_impl, exh_len, ref, got, scratch);

    fprintf(stderr, "[3/4] token soup, %ld rounds\n", soup_rounds);
    for (long r = 0; r < soup_rounds && failures < 20; r++) {
        size_t cap = (r % 100 == 0) ? 1u << 20 : (r % 10 == 0) ? 20000 : 300;
        size_t n = gen_soup(buf, cap);
        check_one(impls, n_impl, buf, n, ref, got, scratch, "soup");
    }

    fprintf(stderr, "[4/4] random bytes, %ld rounds\n", rand_rounds);
    for (long r = 0; r < rand_rounds && failures < 20; r++) {
        size_t n = gen_random(buf, r % 50 == 0 ? 100000 : 400, (int)(r & 1));
        check_one(impls, n_impl, buf, n, ref, got, scratch, "random");
    }

    fprintf(stderr, "%lld cases, %lld bytes compared, %lld failures\n", total_cases, total_bytes, failures);
    return failures ? 1 : 0;
}
