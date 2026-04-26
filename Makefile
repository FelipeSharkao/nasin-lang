LIB_DIR = $(CURDIR)/library

ENV ?= prod
VALID_ENVS = prod dev

ifeq ($(filter $(ENV), $(VALID_ENVS)),)
$(error ENV must be one of $(VALID_ENVS))
endif

ifeq ($(ENV), prod)
RUST_OUT_DIR = target/release
RUSTFLAGS += --release --locked
else
RUST_OUT_DIR = target/debug
GENERATE_GRAMMAR ?= true
endif

.PHONY: all
all: bin/nasin

.PHONY: clean
clean:
	rm -rf bin
	rm -f tree-sitter-nasin/nasin.so
	cargo clean

.PHONY: test
test: bin/nasin
	./rere.py replay tests/_test.list

.PHONY: record-test
record-test: bin/nasin
	./rere.py record tests/_test.list

RUST_OUT = $(RUST_OUT_DIR)/nasin
bin/nasin: $(RUST_OUT)
	mkdir -p bin && cp $(RUST_OUT) bin/nasin

RUST_SRC = $(shell find src/ -type f -name '*.rs')
$(RUST_OUT): Cargo.toml $(RUST_SRC) tree-sitter-nasin/src/parser.c
	LIB_DIR=$(LIB_DIR) cargo build $(RUSTFLAGS)

tree-sitter-nasin/src/parser.c: tree-sitter-nasin/grammar.js tree-sitter-nasin/package.json
ifeq ($(GENERATE_GRAMMAR), true)
	cd tree-sitter-nasin        \
	&& bun install              \
	&& bun tree-sitter generate \
	&& bun tree-sitter build
endif
