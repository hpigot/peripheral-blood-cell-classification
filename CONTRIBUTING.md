# Contributing

This is a solo project, but it follows a team workflow on purpose: every
change is reviewable, tested and traceable to an issue.

## Setup

```bash
uv sync --extra cpu --extra export --extra viz   # or cu126 instead of cpu
uv run pre-commit install
```

`pre-commit install` refuses to run if you have a global `core.hooksPath`.
In that case, have your global pre-commit hook hand over to
`.git/hooks/pre-commit`, and put this line in that file:

```sh
exec uv run --frozen pre-commit run --hook-stage pre-commit
```

## Workflow

1. Every change starts from an issue. Phase 1 (desktop prototype) and
   Phase 2 (edge) are milestones.
2. Branch from `main` as `<issue>-<short-slug>`, e.g. `12-temperature-scaling`.
3. Open a pull request, even when working alone. It says what changed, why,
   and how it was checked, and it closes its issue (`Closes #12`).
4. `main` is protected: CI must pass, and PRs are squash- or rebase-merged,
   so history stays linear with no merge commits.

## Commits

- Imperative subject, at most 50 characters, no trailing period
  ("Add temperature scaling", not "Added temperature scaling.").
- Add a body, wrapped at 72, only when the diff doesn't show *why*.
- One logical change per commit, and every commit passes lint and tests.

## Checks

The same checks run in pre-commit and CI:

| Check | Command |
|---|---|
| Lint and format | `uv run ruff check .` and `uv run ruff format .` |
| Types | `uv run mypy` |
| Tests | `uv run pytest --cov` |
| Secrets | gitleaks (pre-commit hook, full-history scan in CI) |
| No data or weights | pre-commit hook: only `data/README.md` and `data/splits.csv` may be committed under `data/` |
| C++ | CI builds `edge/cpp` on x86 Linux and runs `edge/parity.sh`: model input and predictions must match Python's |

Tests use synthetic data only. CI never downloads the dataset or
pretrained weights.

## Decisions and releases

- Real design choices get a short ADR in `docs/decisions/`, numbered in
  order: context, decision, consequences.
- Each phase ends with a tagged release (`v0.1.0` for Phase 1, `v0.2.0` for
  Phase 2) and an entry in `CHANGELOG.md`.
