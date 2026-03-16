# Zennit CRP

## Project overview

The goal of this repo is to refactor zennit-crp (https://github.com/rachtibat/zennit-crp) to be more in line with the original architecture of zennit (https://github.com/chr5tphr/zennit). 

## Environment

- The environment is managed by `uv`.

## Coding guidelines

- Make sure to follow the coding practices from Zennit.
- Make good use of Python built-ins:
    - Always use `pathlib` for paths.
    - Use `functools`. In particular, use `cache` when relevant.
    - Use `itertools` when relevant.
    - Use `contextlib`. In particular, use `suppress` and `ExitStack` when relevant.
    - Use `match` with guards when relevant.
    - Use `:=` when relevant.
    - Use `typing.Protocols` when relevant.
- Always use best practices when adding or changing code.
- Always check for bad code smells after adding or changing code.
- Add or update docstrings for code you change, even if nobody asks.
- Add comments throughout the code when code blocks are long.
- Use `ruff` for formatting and checking.
- Ensure `README.md` is up-to-date.

## Testing instructions

- Use `pytest` for testing.
- Place tests under `tests/`.
- Add or update tests for code you change, even if nobody asks.
- Ensure tests are relevant and cover the most important edge cases.
- Fix any test or error until the whole suite is green.
- If any warnings appear, verify whether they are relevant. If so, fix them, otherwise, ignore them.

## Commits

- When asked to commit, always commit in logical batches with a descriptive message.