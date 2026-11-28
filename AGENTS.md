# AGENTS.md

Operating policy for humans and coding agents working on `baml-ml`.

## Trunk-based workflow

This repository uses **trunk-based development** on `main` with mandatory pull requests:

- **Never push feature commits directly to `main`.** Work on a short-lived branch (`vibe/<short-slug>-<id>`, `feature/...`, `fix/...`), open a pull request, and merge or discard the branch the same day — no long-lived branches and no multi-story mega-commits. One story or focused increment per PR; keep PRs small and atomic.
- **Keep `main` green at all times.** Never merge broken code; if a `main` CI check turns red, fix forward immediately on a new PR, or push a `git revert` of the offending commit (reverts are the only direct pushes allowed on `main`).
- **Run local verification before opening a PR** (and re-run before merging if `main` moved):
  - Always: `BAML_TELEMETRY=off "$BAML_CLI" check` and `BAML_TELEMETRY=off "$BAML_CLI" test --max-concurrency 1 --test-timeout 5m`.
  - When touching traces, loaders, or observation policies: run `experiments/imported_inference/run.sh` and `experiments/native_ml_observability/run.sh` (or `experiments/adaptive_shadow/run.py` when relevant).
  - When touching standalone CLI entry points (`baml_src/surface/cli.baml`, `baml_src/runtime/shadow.baml`, `baml.toml`): verify `experiments/standalone/run.sh` or confirm the `standalone-package` CI job passes on `main`.
- **No force-pushes to `main`.**

## Stories and Kanban (`status/*` labels)

- Backlog stories live in GitHub Issues with label `status/todo`.
- Respect explicit issue dependencies (`Depends on #...`) before starting a story.
- Move the issue label from `status/todo` to `status/in-progress` when starting work.
- Reference `Closes #<number>` in the PR body (and commit message) and move the label to `status/done` once the PR is merged and green on `main`.
- Respect explicit **Out of scope** sections in each story; open a new `status/todo` issue instead of expanding scope mid-commit.

## Engineering invariants

- **Pinned toolchain**: BAML `0.26.0` (verify archive SHA-256) and `scikit-learn 1.8.0` for independent oracle fixtures.
- **Inference-first, data-only**: training stays external; never execute pickle or arbitrary model code in the runtime.
- **Strict rejection over silent approximation**: reject unsupported estimator variants, invalid dimensions, non-finite parameters, unknown policy fields, or schema/feature/class mismatches explicitly.
- **Backward compatibility**: preserve existing `schema_version` artifact contracts, `BatchOutput` v1, and existing `ObservationMode` (`Timing`, `Semantic`, `Full`, `Adaptive`, `BAML_TELEMETRY=off`) semantics unless an ADR explicitly supersedes them.
- **Native BAML pitfalls**: float comparisons use a total order — always use `IsNaN` / `IsFinite` rather than `x != x`; BAML classes/arrays are mutable by reference — never expose factory-owned fitted state or caller-mutable policies inside loaded handles.
- **Honest evidence**: never make unmeasured performance or privacy claims; whenever `baml_src/*.baml` changes affect a committed benchmark report, re-run the benchmark or note clearly which commit/hashes the report covers.

## Releases

- Tagging an annotated `v*` tag on a green `main` commit (`git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`) automatically triggers `.github/workflows/release.yml`, which builds, offline-validates, checksums, and attaches `linux-x86_64`, `linux-arm64`, and `macos-arm64` archives to the GitHub Release.
