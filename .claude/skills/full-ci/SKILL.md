---
name: full-ci
description: Run full CI pipeline — lint, tests, selfcheck, placeholder scan
---

Run full CI in order. Stop on first failure.

1. **Lint**: `ruff check src/ && ruff format --check src/`
2. **Tests**: `python -m pytest tests/ -v --tb=short`
3. **Selfcheck**: `python -m bacttraitcluster.cli --self-check`
4. **Placeholder scan**: `grep -rn '<PKG>\|<TOOL\|{PKG}\|TODO\|FIXME\|HACK\|XXX' src/ tests/ CLAUDE.md .claude/ .github/ --include='*.py' --include='*.md' --include='*.json' --include='*.yaml'`

Report each step PASS/FAIL.
