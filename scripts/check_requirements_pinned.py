from __future__ import annotations

from pathlib import Path


REQUIREMENT_FILES = [
    Path("requirements.txt"),
    Path("app/requirements.txt"),
    Path("ml_worker/requirements.txt"),
]

ALLOWED_PREFIXES = ("-r ", "--requirement ", "--index-url ", "--extra-index-url ")


def is_pinned_requirement(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    if stripped.startswith(ALLOWED_PREFIXES):
        return True
    if " @ " in stripped:
        return True
    return "==" in stripped


def main() -> int:
    violations: list[str] = []
    for path in REQUIREMENT_FILES:
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not is_pinned_requirement(line):
                violations.append(f"{path}:{line_number}: {line.strip()}")

    if violations:
        print("Unpinned requirements found:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("All requirements are pinned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
