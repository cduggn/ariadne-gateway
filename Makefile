# make check     format, lint and tests, as CI runs them
# make parity    the same scripted traffic through the Go gateway and this one; fails on any observable difference
# make image     the multi-arch image (IMAGE=docker.io/<you>/ariadne-gateway TAG=...)
ARIADNE ?= .cache/ariadne
IMAGE   ?= docker.io/cdugga/ariadne-gateway
TAG     ?= $(shell git rev-parse --short HEAD)
OUT     := .cache/parity

.PHONY: check parity image

check:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --locked -- -D warnings
	cargo test --workspace --locked

$(ARIADNE):
	git clone --quiet https://github.com/cduggn/Ariadne $(ARIADNE)
	git -C $(ARIADNE) checkout --quiet $${ARIADNE_COMMIT:-12204502aa0eb0e68bc97f14d0e720a0e5c2ed19}

parity: $(ARIADNE)
	@mkdir -p $(OUT)/go
	cd $(ARIADNE)/gateway && go build -o $(CURDIR)/$(OUT)/go/ ./cmd/gateway ./cmd/fakevllm ./cmd/fakeoverflow
	cargo build --release --locked --bins
	python3 parity/parity.py observe --gateway $(OUT)/go/gateway --fakevllm $(OUT)/go/fakevllm \
	  --fakeoverflow $(OUT)/go/fakeoverflow --out $(OUT)/go.json --port-base 18100
	python3 parity/parity.py observe --gateway target/release/gateway --fakevllm $(OUT)/go/fakevllm \
	  --fakeoverflow $(OUT)/go/fakeoverflow --out $(OUT)/rust.json --port-base 18300
	python3 parity/parity.py observe --gateway target/release/gateway --fakevllm target/release/fakevllm \
	  --fakeoverflow target/release/fakeoverflow --out $(OUT)/rust-fakes.json --port-base 18500
	python3 parity/parity.py diff $(OUT)/go.json $(OUT)/rust.json
	python3 parity/parity.py diff $(OUT)/go.json $(OUT)/rust-fakes.json

image:
	docker buildx build --platform linux/amd64,linux/arm64 -t $(IMAGE):$(TAG) --push .
