# Lexnet as a C program. See docs/README.md.
CC      ?= cc
CFLAGS  ?= -O3 -march=native -Wall -Wextra -Wno-comment
LDFLAGS ?=

BUILD   := build
CORPUS  ?= $(BUILD)/corpus.c
REPS    ?= 50

SRC := src/lexnet_logic.c src/lexnet_fast.c

.PHONY: all test test-quick bench clean reference corpus model

all: $(BUILD)/lexnet $(BUILD)/lexnet-ref

$(BUILD):
	mkdir -p $(BUILD)

# the re-implementation (default --impl fast)
$(BUILD)/lexnet: src/main.c $(SRC) src/lexnet.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/main.c $(SRC) $(LDFLAGS)

# the original network files are not committed; fetch them from Hugging Face on demand
MODEL := model/lexnet.c model/lexnet_kernel.h model/lexnet_weights.bin
$(MODEL):
	scripts/fetch_model.sh

model: $(MODEL)

# the original network, built exactly as its README says
$(BUILD)/lexnet-ref: $(MODEL) | $(BUILD)
	$(CC) -O3 -march=native -w -o $@ model/lexnet.c -lm

reference: $(BUILD)/lexnet-ref

# differential test against the reference kernel (in-process); NEON and scalar builds
$(BUILD)/difftest: tests/difftest.c $(SRC) src/lexnet.h $(MODEL) | $(BUILD)
	$(CC) -O2 -w -o $@ tests/difftest.c $(SRC) -lm

$(BUILD)/difftest-scalar: tests/difftest.c $(SRC) src/lexnet.h $(MODEL) | $(BUILD)
	$(CC) -O2 -w -DLEXNET_NO_SIMD -o $@ tests/difftest.c $(SRC) -lm

TEST_FILES := model/lexnet.c model/lexnet_kernel.h src/*.c src/*.h tests/*.c

test-quick: $(BUILD)/difftest $(BUILD)/difftest-scalar
	$(BUILD)/difftest --quick $(TEST_FILES)
	$(BUILD)/difftest-scalar --quick --seed 7 $(TEST_FILES)

test: $(BUILD)/difftest $(BUILD)/difftest-scalar
	$(BUILD)/difftest $(TEST_FILES) $(wildcard $(SDK_INCLUDE)/*.h)
	$(BUILD)/difftest-scalar --seed 7 $(TEST_FILES)

# benchmark corpus: the macOS SDK headers concatenated (about 4 MB), or any file via CORPUS=
SDK_INCLUDE := $(shell xcrun --show-sdk-path 2>/dev/null)/usr/include
corpus: $(CORPUS)
$(CORPUS): $(MODEL) | $(BUILD)
	cat $(SDK_INCLUDE)/*.h > $@ 2>/dev/null || cat model/lexnet.c src/*.c > $@

bench: all $(CORPUS)
	bench/bench.sh $(CORPUS) $(REPS)

clean:
	rm -rf $(BUILD)
