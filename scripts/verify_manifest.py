#!/usr/bin/env python3
"""Verify every SHA-256 digest recorded in the frozen release manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def main() -> None:
    """Fail if any versioned release file differs from its recorded digest."""

    entries = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split(maxsplit=1)
        path = ROOT / relative_path.removeprefix("./")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(
                f"SHA-256 mismatch for {relative_path}: "
                f"expected {expected}, got {actual}"
            )
        entries += 1
    print(f"manifest_status=PASS entries={entries}")


if __name__ == "__main__":
    main()
