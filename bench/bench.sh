#!/bin/sh
# bench.sh CORPUS [REPS] - time the reference kernel and both re-implementations on one file.
set -e
CORPUS=${1:?usage: bench.sh corpus.c [reps]}
REPS=${2:-50}
DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD="$DIR/build"

size=$(wc -c < "$CORPUS" | tr -d ' ')
echo "corpus: $CORPUS ($size bytes), $REPS repetitions each, $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
echo

# the reference prints "N chars x R: ms (ns/char, M chars/s)"; interpreter mode is ~5x slower
# than its compiled kernel, so give it fewer repetitions
irep=$(( REPS / 10 )); [ "$irep" -lt 1 ] && irep=1
printf '%-22s' "reference --interp"; "$BUILD/lexnet-ref" -w "$DIR/model/lexnet_weights.bin" --interp --bench "$irep" "$CORPUS"
printf '%-22s' "reference kernel";   "$BUILD/lexnet-ref" -w "$DIR/model/lexnet_weights.bin" --bench "$REPS" "$CORPUS"
printf '%-22s' "logic";              "$BUILD/lexnet" --impl logic --bench "$REPS" "$CORPUS"
printf '%-22s' "fast";               "$BUILD/lexnet" --impl fast --bench "$REPS" "$CORPUS"
