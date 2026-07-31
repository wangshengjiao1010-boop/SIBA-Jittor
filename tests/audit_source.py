#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
from pathlib import Path


TRACKED_SUFFIXES = {".py", ".md", ".txt"}
FRAMEWORK_METHOD_MAP = {"forward": "execute"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def symbols(path):
    if path.suffix != ".py":
        return {"classes": [], "functions": []}
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    classes = []
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
    return {
        "classes": sorted(classes),
        "functions": sorted(functions),
    }


def scan(root, relative_paths=None):
    manifest = {}
    if relative_paths is None:
        paths = sorted(item for item in root.rglob("*") if item.is_file())
    else:
        paths = [root / relative for relative in sorted(relative_paths)]
    for path in paths:
        if not path.is_file():
            continue
        if ".git" in path.parts or path.suffix.lower() not in TRACKED_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8-sig")
        manifest[relative] = {
            "bytes": path.stat().st_size,
            "lines": len(text.splitlines()),
            "sha256": sha256(path),
            **symbols(path),
        }
    return manifest


def compare(official, mirror):
    official_files = {path for path in official if path.endswith(".py")}
    mirror_files = {path for path in mirror if path.endswith(".py")}
    common_python = sorted(
        path for path in official_files & mirror_files if path.endswith(".py")
    )
    symbol_differences = {}
    for path in common_python:
        differences = {}
        for key in ("classes", "functions"):
            official_symbols = set(official[path][key])
            mirror_symbols = set(mirror[path][key])
            mapped_symbols = {
                symbol
                for symbol in official_symbols
                if FRAMEWORK_METHOD_MAP.get(symbol) in mirror_symbols
            }
            missing = sorted(official_symbols - mirror_symbols - mapped_symbols)
            expected_mirror = official_symbols | {
                FRAMEWORK_METHOD_MAP[symbol]
                for symbol in mapped_symbols
            }
            extra = sorted(mirror_symbols - expected_mirror)
            if missing or extra:
                differences[key] = {"missing": missing, "extra": extra}
        if differences:
            symbol_differences[path] = differences
    return {
        "missing_files": sorted(official_files - mirror_files),
        "extra_files": sorted(mirror_files - official_files),
        "symbol_differences": symbol_differences,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--mirror", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official = scan(args.official.resolve())
    report = {"official": official}
    if args.mirror:
        mirror = scan(
            args.mirror.resolve(),
            [path for path in official if path.endswith(".py")],
        )
        report["mirror"] = mirror
        report["comparison"] = compare(official, mirror)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
