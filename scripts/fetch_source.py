"""Download the EU AI Act source HTML into data/raw/.

Offline step, run by hand. Downloads only -- no parsing, no network access at
app startup.

Two documents are fetched, because neither alone is sufficient:

* The **consolidated** text (CELEX 02024R1689-20260727) is the law currently in
  force, after Regulation (EU) 2026/1744 amended the Act. Answers about
  obligations must come from this.
* The **as-published** OJ text still carries all 180 recitals, which EUR-Lex
  drops from consolidated versions. Recitals are where the Act explains its own
  intent, so they stay in the corpus, sourced from here.

Usage:
    python scripts/fetch_source.py
    python scripts/fetch_source.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import raw_dir  # noqa: E402

# EUR-Lex rejects requests with the default python-requests user agent.
USER_AGENT = "eu-ai-act-rag/0.1 (portfolio project; contact via repository)"
TIMEOUT_SECONDS = 60

SOURCES = [
    {
        "name": "consolidated",
        "celex": "02024R1689-20260727",
        "filename": "consolidated_02024R1689-20260727.html",
        "role": "articles and annexes -- law in force as of 2026-07-27",
        "url": (
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
            "?uri=CELEX%3A02024R1689-20260727"
        ),
    },
    {
        "name": "oj_as_published",
        "celex": "32024R1689",
        "filename": "oj_32024R1689.html",
        "role": "recitals only -- dropped from consolidated versions",
        "url": (
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
            "?uri=OJ:L_202401689"
        ),
    },
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def is_current(source: dict, target: Path, manifest: dict) -> bool:
    """True when the file on disk matches what the manifest recorded."""
    entry = manifest.get(source["name"])
    if entry is None or not target.exists():
        return False
    return entry.get("sha256") == sha256_of(target)


def fetch(source: dict, target: Path) -> dict:
    response = requests.get(
        source["url"],
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    target.write_bytes(response.content)
    return {
        "url": source["url"],
        "celex": source["celex"],
        "role": source["role"],
        "filename": source["filename"],
        "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "bytes": len(response.content),
        "sha256": sha256_of(target),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the local copy matches the manifest",
    )
    args = parser.parse_args()

    destination = raw_dir()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    manifest = load_manifest(manifest_path)

    for source in SOURCES:
        target = destination / source["filename"]
        if not args.force and is_current(source, target, manifest):
            print(f"skip   {source['name']:<16} {target.name} (unchanged)")
            continue
        entry = fetch(source, target)
        manifest[source["name"]] = entry
        print(
            f"fetch  {source['name']:<16} {target.name} "
            f"{entry['bytes']:,} bytes  sha256:{entry['sha256'][:12]}"
        )

    # newline="\n" so the manifest is byte-identical on Windows and Linux;
    # without it write_text() emits CRLF here and the file is perpetually
    # "modified" against the LF copy git stores.
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nmanifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
