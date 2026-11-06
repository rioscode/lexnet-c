/* main.c - command line front end, flag-compatible with the original lexnet.c.
 *
 *   lexnet [--impl logic|fast] [--classes | --bench N] file.c
 *
 * default   : print the file with ANSI colours (same palette as the original)
 * --classes : print one class digit per byte ("0123456789A")
 * --bench N : classify the file N times and report ns/byte
 * -w PATH   : accepted and ignored (there are no weights any more)
 */
#include "lexnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *ANSI[LEX_N_CLASSES] = {
    "", "\x1b[1;35m", "\x1b[36m", "\x1b[1;34m", "", "\x1b[33m",
    "\x1b[32m", "\x1b[32m", "\x1b[2;3;37m", "\x1b[38;5;208m", "\x1b[90m",
};

static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    if (len < 0) { perror(path); exit(1); }
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { perror(path); exit(1); }
    fclose(f);
    *n = (size_t)len;
    return buf;
}

int main(int argc, char **argv) {
    int mode = 0, reps = 1;
    const char *path = NULL;
    void (*impl)(const uint8_t *, size_t, uint8_t *) = lexnet_fast;
    const char *impl_name = "fast";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--classes")) mode = 1;
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) mode = 2, reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--impl") && i + 1 < argc) {
            impl_name = argv[++i];
            if (!strcmp(impl_name, "logic")) impl = lexnet_logic;
            else if (!strcmp(impl_name, "fast")) impl = lexnet_fast;
            else { fprintf(stderr, "unknown --impl %s (logic|fast)\n", impl_name); return 2; }
        }
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) i++;      /* ignored */
        else if (argv[i][0] == '-' && argv[i][1]) { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
        else path = argv[i];
    }
    if (!path) {
        fprintf(stderr, "usage: lexnet [--impl logic|fast] [--classes | --bench N] file.c\n");
        return 2;
    }
    size_t n;
    uint8_t *src = read_file(path, &n);
    uint8_t *cls = malloc(n + 1);

    if (mode == 2) {
        if (reps < 1) reps = 1;
        impl(src, n, cls);                      /* warm up tables and caches */
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int i = 0; i < reps; i++) impl(src, n, cls);
        clock_gettime(CLOCK_MONOTONIC, &b);
        double sec = (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) * 1e-9;
        printf("%s: %zu bytes x %d: %.2f ms  (%.3f ns/byte, %.1f M bytes/s)\n",
               impl_name, n, reps, sec * 1e3, sec / reps / (double)n * 1e9, (double)n * reps / sec / 1e6);
        return 0;
    }
    impl(src, n, cls);
    if (mode == 1) {
        for (size_t t = 0; t < n; t++) putchar("0123456789A"[cls[t]]);
        return 0;
    }
    int cur = -1;
    for (size_t t = 0; t < n; t++) {
        if (cls[t] != cur) { fputs("\x1b[0m", stdout); fputs(ANSI[cls[t]], stdout); cur = cls[t]; }
        putchar(src[t]);
    }
    fputs("\x1b[0m", stdout);
    return 0;
}
