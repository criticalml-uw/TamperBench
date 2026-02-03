# Tom's scripts

## Scripts import pattern

Scripts in this directory that import from other scripts (e.g.,
`from scripts.user.tomtseng.mmlu_pro_llm_judge.judge import ...`) need to add
the repo root to `sys.path` before those imports:

```python
import sys

from tamperbench.utils import get_repo_root

# Add repo root to path so we can import from scripts.*
# We do this because scripts/ isn't in the distributed package.
sys.path.insert(0, str(get_repo_root()))

from scripts.user.tomtseng.other_script import something  # Now this works
```

### Why is this necessary?

`scripts/` is not part of this codebase's package, so it's not directly
importable. If the repo root were in `sys.path`, it would be importable, but
when we run `python scripts/user/tomtseng/foo/bar.py`, Python sets `sys.path[0]`
to the script's directory (`scripts/user/tomtseng/foo/`), not the repo root.

### Alternatives

#### Run with `python -m scripts.user.tomtseng.foo.bar`

Running with `-m` keeps the repo root in `sys.path[0]`, so imports work. But
it's less convenient than `python path/to/script.py` and easy to forget.

#### Register `scripts/` as part of the package

We could add `scripts/` to the package via the `[tool.setuptools.packages.find]`
section in `pyproject.toml`. This would make `scripts.*` importable from
anywhere without the `sys.path` hack. However, it would also include `scripts/`
in the distributed package when publishing to PyPI, adding bloat and potentially
exposing dev-only dependencies.

It would also pollute the namespace, adding a `scripts` package that could
conflict with users' own code or other packages. If we add `scripts/` to the
package, I suggest nesting it under `src/tamperbench/scripts` so that it doesn't
pollute the namespace, and adding an underscore prefix to any scripts that
library users should ignore (e.g., move `scripts/user` to
`src/tamperbench/scripts/_user`).
