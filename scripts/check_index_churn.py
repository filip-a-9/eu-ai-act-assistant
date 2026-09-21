"""Refuse a commit that carries index churn but no rebuild.

The prebuilt index is committed on purpose, so that a host deploying straight
from the repository has no build step. The consequence is that it is not
read-only in practice: opening and querying the collection makes Chroma rewrite
headers and segment bookkeeping, so every `scripts/query.py`, `scripts/ask.py`
or `evals/run_eval.py` run leaves the binaries dirty. Measured after a handful
of queries, the difference was 372 bytes spread across ~13 MB.

Committing that costs a fresh ~13 MB blob in history, because git stores
binaries whole and cannot delta them usefully. Worse, the churn is
indistinguishable from a real rebuild in `git status`, so a genuinely stale
index and a merely-queried one look identical.

The distinction this hook draws is the one `index_manifest.json` already
exists to answer. A real rebuild rewrites it, with a new `built_at` and a
`chunks_sha256` for the corpus the vectors were built from. Querying does not
touch it. So: index binaries staged without the manifest means churn, and the
commit is refused.

Invoked by .pre-commit-config.yaml. It asks git what is staged rather than
reading the paths pre-commit would pass it, because pre-commit partitions the
file list across parallel invocations: a run of `pre-commit run --all-files`
here handed one invocation two of the six index files and no manifest, which as
an argv-based check would reject a genuine rebuild whenever the manifest landed
in a different partition. `git diff --cached` sees the whole commit every time.

Asking git also gets the right semantics for free. It lists only paths whose
staged content differs from HEAD, so `git add data/index/` on a merely-queried
index stages the manifest without listing it -- unchanged content is not a
rebuild, and the hook still refuses.

Run by python3 rather than the project venv on purpose: it needs nothing but
the standard library, and a hook that required .venv would block every commit
on a fresh clone.
"""

from __future__ import annotations

import subprocess
import sys

INDEX_PREFIX = "data/index/"
MANIFEST = "data/index/index_manifest.json"


def staged_paths() -> list[str]:
    """Paths whose staged content differs from HEAD, as forward-slash strings."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [path for path in result.stdout.split("\0") if path]


def main() -> int:
    staged = staged_paths()
    paths = [path for path in staged if path.startswith(INDEX_PREFIX)]
    if not paths:
        return 0
    if MANIFEST in staged:
        return 0

    print("refusing to commit index churn.\n")
    print("These index files are staged, but index_manifest.json is not:\n")
    for path in sorted(paths):
        print(f"    {path}")
    print(
        "\nQuerying the index rewrites its bookkeeping without changing what it\n"
        "contains, so this is almost certainly the residue of a query rather\n"
        "than a rebuild. Committing it writes another ~13 MB blob into history.\n"
        "\nTo discard it:\n"
        "\n    git restore --staged data/index/ && git restore data/index/\n"
        "\nIf this really is a rebuild, scripts/build_index.py rewrites\n"
        "index_manifest.json too -- stage it with the rest and this passes."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
