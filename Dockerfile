# The gateway image: a static musl binary on distroless, with no shell and no package manager.
#   docker buildx build --platform linux/amd64,linux/arm64 -t <repo>:<tag> --push .
# Each platform builds natively on Alpine, so a cross-platform build runs the other architecture under emulation.
FROM rust:1.98-alpine@sha256:7cc1c22d77d9432f7fe012a70e6d3e555af54c2a6832700ed7d553f1769ae89f AS build
RUN apk add --no-cache musl-dev make perl cmake clang
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --bin gateway && cp target/release/gateway /gateway

FROM gcr.io/distroless/static-debian12:nonroot@sha256:afa5c872c891853ca7fcf1f12c3edb23f7eeef36189728842dd51042ff57f7ab
COPY --from=build /gateway /gateway
USER 65532:65532
EXPOSE 8000
ENTRYPOINT ["/gateway"]
