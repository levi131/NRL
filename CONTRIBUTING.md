# Contributing to NativeRL

Thanks for your interest in contributing! This file contains a short guide to help you get started.

Running tests

We provide a helper script to run tests from the project root. The script sets the repository root on `PYTHONPATH` so tests import the local package correctly.

Run all tests:

```bash
bash scripts/run_tests.sh
```

Run a single test file or pass pytest options:

```bash
bash scripts/run_tests.sh tests/common/test_logger.py -q
```

Development workflow

- Open an issue to discuss larger changes before implementing.
- Create a branch named `feat/your-feature` or `fix/your-bug`.
- Add or update tests when you add functionality.
- Run tests locally with the script above before opening a PR.
- Follow the project's coding style and commit message conventions.

Thanks — contributions are welcome!
