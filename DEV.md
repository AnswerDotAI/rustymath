# Development

## Commands

```bash
maturin develop && pytest -q
ship-rs-build
```

## Layout

The root package is a pure Rust crate, published to crates.io. Rust users who depend on it never build PyO3. The Python extension is the `py` package. It depends on the root crate, is never published, and maturin builds it through `[tool.maturin].manifest-path`.

## Versioning

The canonical version lives in `[workspace.package]` in `Cargo.toml`, which both crates inherit. `pyproject.toml` gets the Python package version from Cargo via `dynamic = ["version"]`.

## Build profiles

uv builds and `maturin develop --release` use the incremental `release` profile for fast local iteration. CI builds distributed wheels with `dist`, which enables full LTO and one codegen unit, disables incremental compilation, and strips the result.

## Release

1. Run `maturin develop && pytest -q`.
2. Confirm the release version in `Cargo.toml` (`[workspace.package].version`).
3. Run `ship-release`.

`ship-release` pushes a `v<version>` tag, which starts the release workflow in GitHub Actions, then bumps the version.
